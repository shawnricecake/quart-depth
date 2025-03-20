import os
import os.path as osp
import cv2
import time
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from mono.utils.logger import setup_logger
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data

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
    test_data_path = args.test_data_path
    if not os.path.isabs(test_data_path):
        test_data_path = osp.join(CODE_SPACE, test_data_path)

    if 'json' in test_data_path:
        test_data = load_from_annos(test_data_path)
    else:
        test_data = load_data(args.test_data_path)

    main_worker(cfg, test_data)


def main_worker(cfg: dict, test_data: list):
    # todo: build quant config with no quant

    # build model
    model = get_configured_monodepth_model(cfg, )
    model = model.eval()
    model = model.cuda()
        
    # load ckpt
    model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    
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