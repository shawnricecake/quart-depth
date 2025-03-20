import os
import sys

CODE_SPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse

from mmengine import Config, DictAction
from mono.model.monodepth_model import get_configured_monodepth_model


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

    # build model
    model = get_configured_monodepth_model(cfg).eval().cuda()

    print(model)


if __name__ == '__main__':
    args = parse_args()
    main(args)