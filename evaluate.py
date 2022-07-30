import argparse

from ppnet import global_analysis, local_analysis, alignment_analysis


# General script arguments
parser = argparse.ArgumentParser(description='Evaluate prototypes activations of a trained model', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
parser.add_argument('--model', type=str, required=True, help='path of a trained model to use for evaluation')
parser.add_argument('--gpus', type=str, default='0', help='list of gpus to use, e.g. 0,1,2 (default: %(default)s)')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to use for data loading (default: %(default)s)')
parser.add_argument('--out', '-o', type=str, default='analysis', help='output path for saving the evaluation results (default: %(default)s)')
subparsers = parser.add_subparsers()


# Global analysis arguments
global_parser = subparsers.add_parser('global', help='run global analysis')
global_parser.add_argument('--dataset', type=str, required=True, help='path of the dataset to use for evaluation')
global_parser.add_argument('--top_imgs', type=int, default=5, help='number of most activated images to be displayed for each prototype (default: %(default)s)')
global_parser.set_defaults(func=global_analysis.run_analysis)


# Local analysis arguments
local_parser = subparsers.add_parser('local', help='run local prototypes analysis')
local_parser.add_argument('--img', type=str, required=True, help='path of an image or a directory of a split in the dataset to use for evaluation. If a directory is given, the analysis will be performed on a random subset of images in the directory.')
local_parser.add_argument('--top_prototypes', type=int, default=10, help='number of most activated prototypes to be displayed (default: %(default)s)')
local_parser.add_argument('--top_classes', type=int, default=5, help='number of most activated classes for which display the top prototypes (default: %(default)s)')
local_parser.set_defaults(func=local_analysis.run_analysis)


# Alignment score arguments
local_parser = subparsers.add_parser('alignment', help='run alignment analysis of prototypes')
local_parser.add_argument('--dataset', type=str, required=True, help='path of the dataset to use for evaluation')
local_parser.set_defaults(func=alignment_analysis.run_analysis)


if __name__ == '__main__':
    # Start evaluation
    args = parser.parse_args()
    if 'func' in args:
        args.func(args)
    else:
        raise RuntimeError('Specify analysis type: "global" or "local".')
