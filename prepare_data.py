import os
import shutil
import gdown
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

CUB200_URL = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'  # Data taken from https://www.vision.caltech.edu/datasets/cub_200_2011/
CELEB_A_HQ_URL = 'https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv'  # Data taken from http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html

DATASETS_FOLDER = './datasets'


def download_dataset(url: str, filename: str):
    """Download dataset from gdrive."""
    out_path = os.path.join(DATASETS_FOLDER, os.path.splitext(filename)[0], filename)
    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdown.download(url, out_path, quiet=False)
    return out_path


def unpack_dataset(filepath: str):
    """Unpack dataset."""
    filename, extension = os.path.splitext(os.path.basename(filepath))
    out_path = os.path.join(DATASETS_FOLDER, filename, 'original')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        gdown.extractall(filepath, out_path)
    return out_path


def generate_cub200():
    # Download and unpack dataset
    dataset_path = download_dataset(CUB200_URL, 'cub200.tgz')
    original_path = unpack_dataset(dataset_path)
    # Crop images
    print('Cropping images...')
    img_count = 0
    with open(f'{original_path}/CUB_200_2011/images.txt') as f:
        for line in f:
            img_count += 1
    with open(f'{original_path}/CUB_200_2011/images.txt') as images_file, \
            open(f'{original_path}/CUB_200_2011/bounding_boxes.txt') as bboxes_file, \
            open(f'{dataoriginal_path_path}/CUB_200_2011/train_test_split.txt') as split_file:
        for images_line, bboxes_line, split_line in tqdm(zip(images_file, bboxes_file, split_file), total=img_count):
            # Read lines
            id1, path = images_line.strip().split(' ')
            id2, x, y, w, h = [int(float(x)) for x in bboxes_line.strip().split(' ')]
            id3, is_training = split_line.strip().split(' ')
            if int(id1) != int(id2) or int(id1) != int(id3):
                raise ValueError(f'Ids in images.txt and bounding_boxes.txt do not match for {id1}, {id2} and {id3}.')
            # Crop image
            img = Image.open(f'{original_path}/CUB_200_2011/images/{path}')
            cropped = img.crop((x, y, x + w, y + h))
            # Save image
            parent_folder = path.split('/')[0]
            out_path = f'{dataset_path}/{"train" if bool(int(is_training)) else "test"}/{parent_folder}'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            cropped.save(f'{out_path}/{id1}.jpg')


def generate_celeb_a():
    # Download and unpack dataset
    dataset_path = download_dataset(CELEB_A_HQ_URL, 'celeb_a.zip')
    original_path = unpack_dataset(dataset_path)
    # Read attribute annotations
    print('Reading attribute annotations...')
    attributes = pd.read_csv(f'{original_path}/CelebAMask-HQ-attribute-anno.txt', skiprows=1, sep='\s+', header=None,)



parser = argparse.ArgumentParser(description='Download and prepare the dataset')
parser.add_argument('dataset', type=str, choices=['cub200', 'celeb_a'], help='The dataset to download %(choices)s')

if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Generating {args.dataset} dataset...')
    if args.dataset == 'cub200':
        generate_cub200()
    elif args.dataset == 'celeb_a':
        generate_celeb_a()
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
