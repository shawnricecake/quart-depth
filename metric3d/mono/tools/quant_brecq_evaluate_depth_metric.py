import os
import os.path as osp
import cv2
import time
import sys
import re

CODE_SPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from mono.utils.logger import setup_logger
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data, calibration_with_one_batch
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data
from mono.model.decode_heads.RAFTDepthNormalDPTDecoder5 import LoRALinear, Conv2dLoRA, ConvTranspose2dLoRA
from mono.model.backbones.ViT_DINO_reg import Attention as EncoderAttention

from mono.utils.do_test import transform_test_data_scalecano

from mono.quant_fisher_gradient.state import enable_calibration_woquantization, enable_quantization, disable_all
from mono.quant_fisher_gradient.quantized_module import QuantizedLayer, QuantizedBlock, PreQuantizedLayer, \
    QuantizedMatMul, QuantEncoderAttention
from mono.quant_fisher_gradient.fake_quant import QuantizeBase
from mono.quant_fisher_gradient.observer import ObserverBase, AvgMSEObserver, AvgMinMaxObserver
from mono.quant_fisher_gradient.recon import reconstruction
import mono.quant_fisher_gradient.utils as brecq_util

from mono.quant_fq.layers import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'],
                        default='None', help='job launcher')
    parser.add_argument('--test_data_path_1', default='None', type=str, help='the path of test data')
    parser.add_argument('--test_data_path_2', default='None', type=str, help='the path of test data')
    parser.add_argument('--train_data_path', default='None', type=str, help='the path of test data')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size for inference')

    # xuan: add
    parser.add_argument('--cali_data_path', default='None', type=str, help='the path of test data')
    parser.add_argument('--quant_config_path', default="", type=str, help='quant config path')
    parser.add_argument('--save_path', default="", type=str, help='save path')

    args = parser.parse_args()
    return args


def main(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # show_dir is determined in this priority: CLI > segment in file > filename
    if args.show_dir is not None:
        # update configs according to CLI args if args.show_dir is not None
        cfg.show_dir = args.show_dir
    else:
        # use condig filename + timestamp as default show_dir if args.show_dir is None
        cfg.show_dir = osp.join('./show_dirs',
                                osp.splitext(osp.basename(args.config))[0],
                                args.timestamp)

    # ckpt path
    if args.load_from is None:
        raise RuntimeError('Please set model path!')
    cfg.load_from = args.load_from
    cfg.batch_size = args.batch_size

    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)

    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)

    # init the logger before other steps
    cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # init distributed env dirst, since logger depends on the dist info
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_env(args.launcher, cfg)
    logger.info(f'Distributed training: {cfg.distributed}')

    # dump config
    cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))

    # load dataset
    cali_data_path = args.cali_data_path
    test_data_path_1 = args.test_data_path_1
    test_data_path_2 = args.test_data_path_2
    cali_data = load_from_annos(cali_data_path)
    test_data_1 = load_from_annos(test_data_path_1)
    test_data_2 = load_from_annos(test_data_path_2)

    main_worker(cfg, cali_data, test_data_1, test_data_2, args)


def main_worker(cfg: dict, cali_data: list, test_data_1: list, test_data_2: list, args):
    q_config = brecq_util.parse_config(args.quant_config_path)
    use_brecq = True  # todo

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    vit_name = re.search(r'vit_\w+', cfg.load_from).group(0)
    save_path = args.save_path + "/" + \
                "brecq-{}-w_{}_{}_{}bit-a_{}_{}_{}bit.pth".format(
                    vit_name,
                    q_config.w_qconfig.observer,
                    "sym" if q_config.w_qconfig.symmetric else "asym",
                    q_config.w_qconfig.bit,
                    q_config.a_qconfig.observer,
                    "sym" if q_config.a_qconfig.symmetric else "asym",
                    q_config.a_qconfig.bit,
                )

    # build model
    model = get_configured_monodepth_model(cfg).eval().cuda()
    model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)

    # ================= Quantization Preparation ===================
    model = quantize_model(model, q_config).eval().cuda()
    fp_model = copy.deepcopy(model)

    cali_data = cali_data[: q_config.calibration_data_num]
    print("Calibrating...")
    calibrate(model, cali_data, cfg)

    # # calibrate_softmax(model, cali_data, cfg)

    recon_model(model, fp_model, cali_data, q_config.recon, brecq=use_brecq, cfg=cfg)

    enable_quantization(model)
    # ================= Quantization Preparation ===================

    # Evaluation
    print("Evaluating on {}".format(args.test_data_path_1))
    do_scalecano_test_with_custom_data(
        model.eval().cuda(),
        cfg,
        test_data_1,
        cfg.distributed,
        cfg.batch_size,
    )
    print("Evaluating on {}".format(args.test_data_path_2))
    do_scalecano_test_with_custom_data(
        model.eval().cuda(),
        cfg,
        test_data_2,
        cfg.distributed,
        cfg.batch_size,
    )

    # Save quantized model
    torch.save(model.state_dict(), save_path)


def quantize_model(model, config_quant):
    def replace_module(module, w_qconfig, a_qconfig, special_config, q_output_or_input=True):
        for name, child_module in module.named_children():
            if 'embed' in name:
                continue
            elif isinstance(child_module, (EncoderAttention)):
                setattr(module, name, QuantEncoderAttention(child_module, w_qconfig, a_qconfig, special_config))
            elif isinstance(child_module, (nn.Linear, LoRALinear)):
                setattr(module, name, PreQuantizedLayer(child_module, None, w_qconfig, a_qconfig, q_output_or_input))
            elif isinstance(child_module, (nn.Conv2d, Conv2dLoRA)):  # ConvTranspose2dLoRA
                setattr(module, name, PreQuantizedLayer(child_module, None, w_qconfig, a_qconfig, q_output_or_input))
            elif isinstance(child_module, nn.Identity):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, special_config, q_output_or_input)

    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, config_quant.special)

    return model


@torch.no_grad()
def calibrate(model, cali_data, cfg):
    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')

    # metric3d adds: ====================================================================
    with torch.no_grad():
        for i in tqdm(range(0, len(cali_data), 1), desc='Calibrating for activation:', ncols=100):
            calibration_with_one_batch(cfg, model, cali_data[i: i + 1])
    # ========================================================================================

    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')

    # metric3d adds: ====================================================================
    with torch.no_grad():
        for i in tqdm(range(0, len(cali_data), 1), desc='Calibrating for weight:', ncols=100):
            calibration_with_one_batch(cfg, model, cali_data[i: i + 1])
    # ========================================================================================


@torch.no_grad()
def calibrate_softmax(model, cali_data, cfg):
    for name, module in model.named_modules():
        if isinstance(module, (QAct, QIntSoftmax)):
            module.calibrate = True

    with torch.no_grad():
        for i in tqdm(range(0, len(cali_data), 1), desc='Calibrating:'):
            if i == len(cali_data) - 1:
                # Open last calibration mode:
                for m in model.modules():
                    if isinstance(module, (QAct, QIntSoftmax)):
                        m.last_calibrate = True

            calibration_with_one_batch(cfg, model, cali_data[i: i + 1])

    # Close calibration mode:
    for m in model.modules():
        if isinstance(module, (QAct, QIntSoftmax)):
            m.calibrate = False

    # Open quant mode:
    for name, m in model.named_modules():
        if isinstance(module, (QAct, QIntSoftmax)):
            m.quant = True


def recon_model(model, fp_model, cali_data, recon_config, brecq=False, cfg=None):
    if len(cali_data) > 8:
        recon_config.keep_gpu = False

    if brecq:
        enable_quantization(model, 'weight_fake_quant')
    else:
        enable_quantization(model)

    def _recon_model(module, fp_module, previous_name=""):
        for name, child_module in module.named_children():
            current_name = previous_name + '.' + name if previous_name else name
            if isinstance(child_module, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer,
                                         QuantizedMatMul)):  # and current_name == 'depth_model.decoder.token2feature.read_1.sample':
                print(f'Reconstructing {current_name} ...')
                reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, recon_config, cfg)
            else:
                _recon_model(child_module, getattr(fp_module, name), current_name)

    # Start reconstruction
    _recon_model(model, fp_model)


if __name__ == '__main__':
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    main(args)