import argparse
from datetime import datetime

from ppnet import train

# Script options
parser = argparse.ArgumentParser(description='Train a new ProtoPNet model', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))

parser.add_argument('--exp', type=str, default=datetime.now().strftime('%Y%m%dT%H%M%S'), help='id of the current experiment (default: %(default)s)')
parser.add_argument('--data_path', type=str, default= './datasets/cub200/', help='path of the dataset to use for training and evaluation (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=500, help='number of training epochs (default: %(default)s)')
parser.add_argument('--warm_epochs', type=int, default=5, help='number of warming epochs (default: %(default)s)')
parser.add_argument('--push_start', type=int, default=10, help='epoch in which to start pushing prototypes  (default: %(default)s)')
parser.add_argument('--push_interval', type=int, default=20, help='epoch interval in which push prototypes  (default: %(default)s)')

parser.add_argument('--num_classes', type=int, default=200, help='number of classes (default: %(default)s)')
parser.add_argument('--img_size', type=int, default=224, help='resize dimension for training images (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=20, help='input batch size (default: %(default)s)')

parser.add_argument('--prototype_activation_function', type=str, default='log', choices=['log', 'linear'], help='activation function for the last prototype (default: %(default)s)')
parser.add_argument('--add_on_layers', type=str, default='regular', choices=['regular', 'bottleneck'], help='add-on layers type (default: %(default)s)')

parser.add_argument('--gpus', type=str, default='0', help='list of gpus to use, e.g. 0,1,2 (default: %(default)s)')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to use for data loading (default: %(default)s)')
parser.add_argument('--seed', type=int, default=0, help='random seed to use (default: %(default)s)')

args = parser.parse_args()

# Start training
train(args)
