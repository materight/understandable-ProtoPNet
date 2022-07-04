import os
import shutil
import re
from argparse import Namespace
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets

from .helpers import set_seed
from . import model, push, prune, train_and_test as tnt, save
from .log import create_logger
from .preprocess import mean, std, preprocess_input_function


def train(args: Namespace):
    # Set default values
    prototype_shape = (2000, 128, 1, 1)
    joint_optimizer_lrs = dict(
        features=1e-4,
        add_on_layers=3e-3,
        prototype_vectors=3e-3
    )
    warm_optimizer_lrs = dict(
        add_on_layers=3e-3,
        prototype_vectors=3e-3
    )
    last_layer_optimizer_lr = 1e-4
    coefs = dict(
        crs_ent=1,
        clst=0.8,
        sep=-0.08,
        l1=1e-4,
    )

    # Init environment
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('GPUs:', os.environ['CUDA_VISIBLE_DEVICES'])

    base_architecture_type = re.match('^[a-z]*', args.architecture).group(0)
    model_dir = os.path.join('./saved_models', args.architecture, args.exp_name)
    if os.path.exists(model_dir):
        print(f'Model directory "{args.exp_name}" already exists, overwriting...')
    else:    
        os.makedirs(model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'ppnet', base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'ppnet', 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'ppnet', 'train_and_test.py'), dst=model_dir)
    with open(os.path.join(model_dir, 'args.yaml'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    img_dir = os.path.join(model_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    train_dir = os.path.join(args.data_path, 'train')
    test_dir = os.path.join(args.data_path, 'test')

    normalize = T.Normalize(mean=mean, std=std)

    # all datasets
    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        T.Compose([
            T.RandomAffine(degrees=15, shear=10),
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_dir,
        T.Compose([
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        T.Compose([
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(args.batch_size))

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=args.architecture,
                                  pretrained=True, img_size=args.img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=args.num_classes,
                                  prototype_activation_function=args.prototype_activation_function,
                                  add_on_layers_type=args.add_on_layers)
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},  # bias are now also being regularized
         {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
            {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
         ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=args.step_size, gamma=0.1)

    warm_optimizer_specs = \
        [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
         ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # train the model
    log('\nStart training')
    for epoch in range(args.epochs + 1):
        log('epoch: \t{0}'.format(epoch))

        if epoch < args.warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
            joint_lr_scheduler.step()

        if epoch % args.push_interval == 0:
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                        target_accu=0.70, log=log, epoch=epoch)

        if epoch > 0 and epoch % args.push_interval == 0:
            push.push_prototypes(
                train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,  # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
                epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=0.70, log=log)

            if args.prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=0.70, log=log)
        log('------------------------------------------\n')
    logclose()
