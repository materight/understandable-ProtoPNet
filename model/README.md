# ProtoPNet

Original repository: https://github.com/cfchen-duke/ProtoPNet

This code package implements the prototypical part network (ProtoPNet) from the paper "This Looks Like That: Deep Learning for Interpretable Image Recognition" (to appear at NeurIPS 2019), by Chaofan Chen* (Duke University), Oscar Li* (Duke University), Chaofan Tao (Duke University), Alina Jade Barnett (Duke University), Jonathan Su (MIT Lincoln Laboratory), and Cynthia Rudin (Duke University) (* denotes equal contribution).

This code package was SOLELY developed by the authors at Duke University, and licensed under MIT License (see LICENSE for more information regarding the use and the distribution of this code package).

**Prerequisites:** PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor). \
**Recommended hardware:** 4 NVIDIA Tesla P-100 GPUs or 8 NVIDIA Tesla K-80 GPUs.

## Dataset preparation
1. Download the dataset `CUB_200_2011.tgz` from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
2. Unpack `CUB_200_2011.tgz`.
3. Crop the images using information from `bounding_boxes.txt` (included in the dataset).
4. Split the cropped images into training and test sets, using `train_test_split.txt` (included in the dataset).
5. Put the cropped training images in the directory `./datasets/cub200_cropped/train_cropped/`.
6. Put the cropped test images in the directory `./datasets/cub200_cropped/test_cropped/`.
7. Augment the training set using `img_aug.py` (included in this code package). This will create an augmented training set in the following directory: `./datasets/cub200_cropped/train_cropped_augmented/`.

## Training the model
1. In `settings.py`, provide the appropriate strings for `data_path`, `train_dir`, `test_dir`, `train_push_dir`:
    - `data_path` is where the dataset resides. If you followed the instructions for preparing the data, it should be `./datasets/cub200_cropped/`.
    - `train_dir` is the directory containing the augmented training set. If you followed the instructions for preparing the data, it should be `data_path` + `train_cropped_augmented/`.
    - `test_dir` is the directory containing the test set. If you followed the instructions for preparing the data, it should be `data_path` + `test_cropped/`.
    - `train_push_dir` is the directory containing the original (unaugmented) training set. If you followed the instructions for preparing the data, it should be `data_path` + `train_cropped/`.
2. Run `main.py`.

## Find nearest prototypes to a test image
1. Run `local_analysis.py` and supply the following arguments:
    - `-gpuid` is the GPU device ID(s) you want to use (optional, default '0').
    - `-modeldir` is the directory containing the model you want to analyze.
    - `-model` is the filename of the saved model you want to analyze.
    - `-imgdir` is the directory containing the image you want to analyze.
    - `-img` is the filename of the image you want to analyze.
    - `-imgclass` is the (0-based) index of the correct class of the image.

## Find the nearest patches to each prototype
1. Run `global_analysis.py` and supply the following arguments:
    - `-gpuid` is the GPU device ID(s) you want to use (optional, default '0').
    - `-modeldir` is the directory containing the model you want to analyze.
    - `-model` is the filename of the saved model you want to analyze.

## Prune the prototypes from a saved model
1. Run `run_pruning.py` and supply the following arguments:
    - `-gpuid` is the GPU device ID(s) you want to use (optional, default '0').
    - `-modeldir` is the directory containing the model you want to prune prototypes from.
    - `-model` is the filename of the saved model you want to prune prototypes from.

**Note:** the prototypes in the model must already have been projected (pushed) onto the nearest latent training patches, before running this script.

## Combine several ProtoPNet models
1. Run the Jupyter Notebook `combine_models.ipynb`.
