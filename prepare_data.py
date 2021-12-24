import os
import shutil
import gdown
import tarfile
from tqdm import tqdm
from PIL import Image

# Data taken from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
CUB200_URL = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'

DATASETS_FOLDER = './datasets'
TMP_FOLDER = f'{DATASETS_FOLDER}/tmp'
OUTPUT_FOLDER = f'{DATASETS_FOLDER}/cub200_cropped'

# Download dataset
os.makedirs(TMP_FOLDER, exist_ok=True)
gdown.download(CUB200_URL, f'{DATASETS_FOLDER}/tmp/data.tgz', quiet=False)

# Unpack dataset
print('Unpacking dataset...')
with tarfile.open(f'{DATASETS_FOLDER}/tmp/data.tgz') as file:
    file.extractall(f'{DATASETS_FOLDER}/tmp')

# Crop images
print('Cropping images...')
os.makedirs(f'{OUTPUT_FOLDER}/train_cropped', exist_ok=True)
img_count = 0
with open(f'{DATASETS_FOLDER}/tmp/CUB_200_2011/images.txt') as f:
    for line in f:
        img_count += 1
with open(f'{DATASETS_FOLDER}/tmp/CUB_200_2011/images.txt') as images_file, \
        open(f'{DATASETS_FOLDER}/tmp/CUB_200_2011/bounding_boxes.txt') as bboxes_file, \
        open(f'{DATASETS_FOLDER}/tmp/CUB_200_2011/train_test_split.txt') as split_file:
    for images_line, bboxes_line, split_line in tqdm(zip(images_file, bboxes_file, split_file), total=img_count):
        # Read lines
        id1, path = images_line.strip().split(' ')
        id2, x, y, w, h = [int(float(x)) for x in bboxes_line.strip().split(' ')]
        id3, is_training = split_line.strip().split(' ')
        if int(id1) != int(id2) or int(id1) != int(id3):
            raise ValueError(f'Ids in images.txt and bounding_boxes.txt do not match for {id1}, {id2} and {id3}.')
        # Crop image
        img = Image.open(f'{DATASETS_FOLDER}/tmp/CUB_200_2011/images/{path}')
        cropped = img.crop((x, y, x + w, y + h))
        # Save image
        parent_folder = path.split('/')[0]
        out_path = f'{OUTPUT_FOLDER}/{"train_cropped" if bool(int(is_training)) else "test_cropped"}/{parent_folder}'
        os.makedirs(f'{out_path}', exist_ok=True)
        cropped.save(f'{out_path}/{id1}.jpg')

# Clean up
shutil.rmtree(TMP_FOLDER)
