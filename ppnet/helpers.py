import os
import torch
import random
import numpy as np


def pairwise_dist(embeddings, squared=False):
    """Compute pairwise distances betweeen embeddings efficently."""
    dot_prod = embeddings.mm(embeddings.T)
    square_norm = dot_prod.diag()
    dist = square_norm.unsqueeze(0) - 2*dot_prod + square_norm.unsqueeze(1) # Squared pairwise distances
    dist = dist.clamp(min=0) # Some values might be negative due to numerical instability. Set distances to >=0
    if not squared:  # Compute sqrt of distances
        mask = (dist == 0).float() # 1 in the positions where dist==0, otherwise 0
        dist = dist + 1e-16 * mask  # Because the gradient of sqrt is infinite when dist==0, add a small epsilon where dist==0
        dist = torch.sqrt(dist)
        dist = dist * (1 - mask) # Correct the epsilon added: set the distances on the mask to be 0
    return dist


def make_one_hot(target, target_one_hot):
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
