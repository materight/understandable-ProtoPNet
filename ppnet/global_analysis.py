import os
import re
from argparse import Namespace
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .helpers import makedir
from .find_nearest import find_k_nearest_patches_to_prototypes
from .log import create_logger
from .preprocess import preprocess_input_function


def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), str(index), 'prototype-img-original.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def run_analysis(args: Namespace):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('GPUs:', os.environ['CUDA_VISIBLE_DEVICES'])
    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    model_path = os.path.abspath(args.model)  # ./saved_models/vgg19/003/checkpoints/10_18push0.7822.pth
    model_base_architecture, experiment_run, _, model_name = re.split(r'\\|/', model_path)[-4:]
    start_epoch_number = int(re.search(r'\d+', model_name).group(0))

    # load the model
    save_analysis_path = os.path.join(args.output, 'global', model_base_architecture, experiment_run, model_name)
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))
    
    log(f'load model from: {args.model}')
    log(f'model epoch: {start_epoch_number}')
    log(f'model base architecture: {model_base_architecture}')
    log(f'experiment run: {experiment_run}')

    ppnet = torch.load(args.model)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size

    # load the data
    batch_size = 100

    # train set: do not normalize
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=False)

    # test set: do not normalize
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=False)

    root_dir_for_saving_train_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'train')
    root_dir_for_saving_test_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'test')
    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_test_images)

    # save prototypes in original images
    load_img_dir = os.path.join(os.path.dirname(args.model), 'img')
    assert os.path.exists(load_img_dir), f'Folder "{load_img_dir}" does not exist'
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}', f'bb{start_epoch_number}.npy'))

    for j in range(ppnet.num_prototypes):
        makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
        makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(root_dir_for_saving_train_images, str(j), 'prototype_in_original_pimg.png'),
            epoch=start_epoch_number,
            index=j,
            bbox_height_start=prototype_info[j][1],
            bbox_height_end=prototype_info[j][2],
            bbox_width_start=prototype_info[j][3],
            bbox_width_end=prototype_info[j][4],
            color=(0, 255, 255)
        )
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(root_dir_for_saving_test_images, str(j), 'prototype_in_original_pimg.png'),
            epoch=start_epoch_number,
            index=j,
            bbox_height_start=prototype_info[j][1],
            bbox_height_end=prototype_info[j][2],
            bbox_width_start=prototype_info[j][3],
            bbox_width_end=prototype_info[j][4],
            color=(0, 255, 255)
        )

    find_k_nearest_patches_to_prototypes(
        dataloader=train_loader,  # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
        k=args.top_patches + 1,
        preprocess_input_function=preprocess_input_function,  # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=log)
    find_k_nearest_patches_to_prototypes(
        dataloader=test_loader,  # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
        k=args.top_patches,
        preprocess_input_function=preprocess_input_function,  # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=log)

    logclose()