import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def save_alignment_matrix(fname, alignment_matrix):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(-alignment_matrix.astype(float))
    ax.set_yticks(range(len(alignment_matrix.index)))
    ax.set_yticklabels(alignment_matrix.index)
    ax.set_xticks(range(len(alignment_matrix.columns)))
    ax.set_xticklabels(alignment_matrix.columns)
    for i in range(len(alignment_matrix.index)):
        for j in range(len(alignment_matrix.columns)):
            ax.text(j, i, int(alignment_matrix.iloc[i, j]), ha='center', va='center', color='w', size='small')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    fig.savefig(fname)


def alignment_score(part_locs, high_act_patch_indices):
    """Compute alignemnt score between a vector fo part location and a bound box of an highly activated pathch."""
    high_act_y, high_act_x = np.mean(high_act_patch_indices[0:2], dtype=int), np.mean(high_act_patch_indices[2:4], dtype=int)
    dist = ((part_locs['x'] - high_act_x) ** 2 + (part_locs['y'] - high_act_y) ** 2) ** .5  # Euclidean distances
    return dist
