import argparse

from ProtoPNet import train

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')  # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()

train.run(args.gpuid)
