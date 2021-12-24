import argparse

from ProtoPNet import train

# Script options
parser = argparse.ArgumentParser(description='Train a new ProtoPNet model', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
parser.add_argument('--gpus', type=str, default='0', help='list of gpus to use, e.g. 0,1,2 (default: %(default)s)', metavar='GPUS')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to use for data loading (default: %(default)s)', metavar='N')
parser.add_argument('--batch_size', type=int, default=20, help='input batch size (default: %(default)s)', metavar='B')
parser.add_argument('--seed', type=int, default=0, help='random seed to use (default: %(default)s)', metavar='S')
args = parser.parse_args()

# Start training
train.run(args.gpus, args.num_workers, args.batch_size, args.seed)
