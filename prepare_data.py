import os
import shutil
import gdown
import tarfile
from tqdm import tqdm
from PIL import Image

CUB200_URL = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'  # Data taken from https://www.vision.caltech.edu/datasets/cub_200_2011/
CELEB_A_HQ_URL = 'https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv'  # Data taken from http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
# CELEB_A_NORMAL_URL = 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'  # Data taken from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

DATASETS_FOLDER = './datasets'
TMP_FOLDER = f'{DATASETS_FOLDER}/tmp'


def download_dataset(url: str, filename: str):
    """Download dataset from gdrive."""
    out_path = os.path.join(TMP_FOLDER, filename)
    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path))
        gdown.download(url, out_path, quiet=False)
    return out_path


def unpack_dataset(filepath: str):
    """Unpack dataset."""
    out_path = os.path.join(DATASETS_FOLDER, os.path.basename(filepath).split('.')[0], 'original')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        with tarfile.open(filepath) as file:
            file.extractall(out_path)
    return out_path


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

# Clean up temp folder
shutil.rmtree(TMP_FOLDER)
