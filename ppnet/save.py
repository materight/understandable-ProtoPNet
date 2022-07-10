import os
import torch


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print, epoch=None):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu or (epoch is not None and epoch % 50 == 0):
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model, f=os.path.join(model_dir, 'checkpoints', (model_name + '{0:.4f}.pth').format(accu*100)))
