import os
import os.path as osp
import cv2
import time
import sys

CODE_SPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

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

from mono.quant_compensation.quant import add_actquant, ActQuantWrapper, Quantizer
from mono.quant_compensation.compensation import Compensation
from mono.model.decode_heads.RAFTDepthNormalDPTDecoder5 import LoRALinear, Conv2dLoRA


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
    parser.add_argument('--test_data_path', default='None', type=str, help='the path of test data')
    parser.add_argument('--train_data_path', default='None', type=str, help='the path of test data')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size for inference')
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
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    # train_data = load_from_annos(train_data_path)
    test_data = load_from_annos(test_data_path)
    # todo: create train data .json
    train_data = test_data

    main_worker(cfg, train_data, test_data)


def main_worker(cfg: dict, train_data: list, test_data: list):
    save_path = "/home/xuans/sensei-fs-link/code/depth-project/depth-anything/quant-depth-estimation/my-clean-code/metric3d/quantized_model/" + \
                "quantized_model-giant-w4.pth"

    # ================= Quantization Configuration ===================
    w_quant = True
    w_bit = 4
    w_sym = False
    w_minmax = True
    w_perchannel = True

    a_quant = False
    a_bit = 8
    a_sym = True
    a_minmax = True
    a_perchannel = True

    parallel_columns = 32
    calibrate_batches = 64
    calibrate_batch_size = 1
    nrounds = 1
    rel_damp = 0.01
    # ================= Quantization Configuration ===================

    # build model
    model_dense = get_configured_monodepth_model(cfg).eval().cuda()
    model_quant = get_configured_monodepth_model(cfg).eval().cuda()

    # load ckpt
    model_dense, _, _, _ = load_ckpt(cfg.load_from, model_dense, strict_match=False)
    model_quant, _, _, _ = load_ckpt(
        cfg.load_from,
        model_quant, strict_match=False
    )

    # ================= Quantization Preparation ===================
    update_list = [
        nn.Conv2d, nn.Linear,
        # ActQuantWrapper,
        LoRALinear, Conv2dLoRA
    ]

    if a_quant:
        add_actquant(model_quant)

    layers_dense = {}
    layers_quant = {}
    for name, module in model_dense.named_modules():
        if 'embed' in name:
            continue
        if type(module) in update_list:
            layers_dense[name] = module
    for name, module in model_quant.named_modules():
        if 'embed' in name:
            continue
        if type(module) in update_list:
            layers_quant[name] = module

    compensation = {}
    for name in layers_quant:
        layer = layers_quant[name]
        # if isinstance(layer, ActQuantWrapper):
        #     layer = layer.module
        compensation[name] = Compensation(layer, rel_damp=rel_damp)
        # if isinstance(layer, ActQuantWrapper) and a_quant:
        #     layers_quant[name].quantizer.configure(
        #         a_bit, perchannel=a_perchannel, sym=a_sym, mse=not a_minmax
        #     )
        if w_quant:
            compensation[name].quantizer = Quantizer()
            compensation[name].quantizer.configure(
                w_bit, perchannel=w_perchannel, sym=w_sym, mse=not w_minmax
            )

    # quant and update weight:
    # if w_quant:
    #     print('Quantizing weights ...')
    #
    #     def add_batch(name):
    #         def tmp(layer, inp, out):
    #             compensation[name].add_batch(inp[0].data, out.data)
    #         return tmp
    #
    #     handles = []
    #     for name in compensation:
    #         handles.append(layers_dense[name].register_forward_hook(add_batch(name)))
    #     with torch.no_grad():
    #         for i in range(nrounds):
    #             for j in range(0, len(train_data), calibrate_batch_size):
    #                 if j >= calibrate_batches:
    #                     break
    #                 calibration_with_one_batch(cfg, model_dense, train_data[j: j + calibrate_batch_size])
    #     for h in handles:
    #         h.remove()
    #
    #     for name in compensation:
    #         print('Quantizing {}...'.format(name))
    #         compensation[name].quantize(parallel=parallel_columns)
    #         compensation[name].free()

    # quant and update weight layerly:
    if w_quant:
        print('Quantizing weights ...')

        for name in compensation:

            def add_batch(name):
                def tmp(layer, inp, out):
                    compensation[name].add_batch(inp[0].data, out.data)

                return tmp

            handle = layers_dense[name].register_forward_hook(add_batch(name))
            with torch.no_grad():
                for i in range(nrounds):
                    for j in range(0, len(train_data), calibrate_batch_size):
                        if j >= calibrate_batches:
                            break
                        calibration_with_one_batch(cfg, model_dense, train_data[j: j + calibrate_batch_size])
            handle.remove()

            print('Quantizing {}...'.format(name))
            compensation[name].quantize(parallel=parallel_columns)
            compensation[name].free()

    # # quant activation:
    # if a_quant:
    #     print('Quantizing activations ...')
    #
    #     def init_actquant(name):
    #         def tmp(layer, inp, out):
    #             layers_quant[name].quantizer.find_params(inp[0].data)
    #         return tmp
    #
    #     handles = []
    #     for name in layers_dense:
    #         handles.append(layers_dense[name].register_forward_hook(init_actquant(name)))
    #     with torch.no_grad():
    #         for j in range(0, len(train_data), calibrate_batch_size):
    #             if j >= calibrate_batches:
    #                 break
    #             calibration_with_one_batch(cfg, model_dense, train_data[j: j + calibrate_batch_size])
    #     for h in handles:
    #         h.remove()
    # ================= Quantization Preparation ===================

    # Save quantized model
    torch.save(model_quant.state_dict(), save_path)

    # Evaluation
    do_scalecano_test_with_custom_data(
        model_quant,
        cfg,
        test_data,
        cfg.distributed,
        cfg.batch_size,
    )


if __name__ == '__main__':
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    main(args)
