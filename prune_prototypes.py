import argparse

from ppnet.prune import run_pruning

# Script arguments
parser = argparse.ArgumentParser(description='Prune the learned prototypes from a trained model', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))

parser.add_argument('--model', type=str, required=True, help='path of the trained model')
parser.add_argument('--dataset', type=str, required=True, help='path of the dataset used for training')

parser.add_argument('--batch_size', type=int, default=32, help='input batch size (default: %(default)s)')
parser.add_argument('--prune_threshold', type=int, default=3, help='resize dimension for training images (default: %(default)s)')
parser.add_argument('--k_nearest', type=int, default=6, help='number of patches to compare for each prototype (default: %(default)s)')
parser.add_argument('--n_iter', type=int, default=10, help='number of iterations to run for last layer optimization after pruning. If 0, do not optimize last layer. (default: %(default)s)')

parser.add_argument('--gpus', type=str, default='0', help='list of gpus to use, e.g. 0,1,2 (default: %(default)s)')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to use for data loading (default: %(default)s)')


if __name__ == '__main__':
    # Start training
    args = parser.parse_args()
    run_pruning(args)
