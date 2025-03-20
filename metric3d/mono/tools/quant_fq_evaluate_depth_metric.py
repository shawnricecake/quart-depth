import os
import os.path as osp
import cv2
import time
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import torch
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

from mono.quant_fq.quant_config import Quant_Config
from mono.quant_fq.layers import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear, QConvTranspose2d

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
    parser.add_argument('--num_calibration_sample', default=16, type=int, help='num calibration samples')
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
    # todo: create train data .json, random sample with args.num_calibration_sample
    train_data = test_data[:32]

    main_worker(cfg, train_data, test_data)


def main_worker(cfg: dict, train_data: list, test_data: list):
    # todo: we may not quantize convtranspose2d, layernorm, and softmax
    # quant_list = [QConv2d, QConvTranspose2d, QLinear, QAct, QIntSoftmax]
    quant_list = [QConv2d, QLinear, QAct]

    # build quant config
    quant_cfg = Quant_Config(
        w_bit='uint4', observer_w='minmax', quantizer_w='uniform', calibration_mode_w='channel_wise',
        a_bit='uint8', observer_a='minmax', quantizer_a='uniform', calibration_mode_a='channel_wise',

        use_int_softmax=True, softmax_bit='uint8', softmax_observer='minmax', softmax_quantizer='log2',
        calibration_mode_s='channel_wise',  # layer_wise

        use_int_norm=False, norm_observer='ptf', norm_calibration='channel_wise',
    )
    print('quant weight: ', quant_cfg.BIT_TYPE_W.bits, quant_cfg.OBSERVER_W, quant_cfg.QUANTIZER_W, quant_cfg.CALIBRATION_MODE_W)
    print('quant activation: ', quant_cfg.BIT_TYPE_A.bits, quant_cfg.OBSERVER_A, quant_cfg.QUANTIZER_A, quant_cfg.CALIBRATION_MODE_A)

    # build model
    model = get_configured_monodepth_model(
        cfg,
        # quant:
        quant=False,
        calibrate=False,
        quant_cfg=quant_cfg,
    )
    model = model.eval()
    model = model.cuda()
        
    # load ckpt
    model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=True, quant=True)

    # ================= Quantization Preparation ===================
    # Open calibration mode:
    for m in model.modules():
        if type(m) in quant_list:
            m.calibrate = True

    with torch.no_grad():
        for i in tqdm(range(0, len(train_data), 1), desc='Calibrating:'):
            if i == len(train_data) - 1:
                # Open last calibration mode:
                for m in model.modules():
                    if type(m) in quant_list:
                        m.last_calibrate = True

            calibration_with_one_batch(cfg, model, train_data[i: i + 1])

    # Close calibration mode:
    for m in model.modules():
        if type(m) in quant_list:
            m.calibrate = False
    # Open quant mode:
    for m in model.modules():
        if type(m) in quant_list:
            m.quant = True
        # if quant_cfg.INT_NORM:
        #     if type(m) in [QIntLayerNorm]:
        #         m.mode = 'int'  # todo: may not quant layernorm
    # ================= Quantization Preparation ===================

    # Evaluation
    do_scalecano_test_with_custom_data(
        model, 
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