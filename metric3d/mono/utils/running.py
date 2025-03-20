import os
import torch
import torch.nn as nn
from mono.utils.comm import main_process
import copy
import inspect
import logging
import glob


index_mapping = {
    'normal_predictor': {
        '0': '0',
        '2': '3',
        '4': '6',
        '6': '9',
    },
    'depth_regressor': {
        '0': '0',
        '2': '3',
    }
}


def load_ckpt(load_path, model, optimizer=None, scheduler=None, strict_match=True, loss_scaler=None, quant=False):
    """
    Load the check point for resuming training or finetuning.
    """
    logger = logging.getLogger()
    if os.path.isfile(load_path):
        if main_process():
            logger.info(f"Loading weight '{load_path}'")
        checkpoint = torch.load(load_path, map_location="cpu")

        if quant:
            if 'model_state_dict' in checkpoint:
                old_state_dict = checkpoint['model_state_dict']
            else:
                old_state_dict = checkpoint
            new_state_dict = {}
            for key in old_state_dict.keys():
                if 'depth_model.decoder.token2feature.read_0.sample.0.weight' == key:
                    new_state_dict['depth_model.decoder.token2feature.read_0.sample.weight'] = old_state_dict[key]
                elif 'depth_model.decoder.token2feature.read_0.sample.0.bias' == key:
                    new_state_dict['depth_model.decoder.token2feature.read_0.sample.bias'] = old_state_dict[key]
                elif 'normal_predictor' in key:
                    parts = key.split('.')
                    module_idx = parts[-2]
                    if module_idx in index_mapping['normal_predictor']:
                        new_module_idx = index_mapping['normal_predictor'][module_idx]
                        new_key = key.replace(module_idx, new_module_idx)
                        new_state_dict[new_key] = old_state_dict[key]
                    else:
                        continue
                elif 'depth_regressor' in key:
                    parts = key.split('.')
                    module_idx = parts[-2]
                    if module_idx in index_mapping['depth_regressor']:
                        new_module_idx = index_mapping['depth_regressor'][module_idx]
                        new_key = key.replace(module_idx, new_module_idx)
                        new_state_dict[new_key] = old_state_dict[key]
                    else:
                        continue
                elif 'depth_model.decoder.context_zqr_convs.' in key:
                    parts = key.split('.')
                    module_idx = parts[-2]
                    new_module_idx = module_idx + '.1'
                    new_key = key.replace(module_idx, new_module_idx)
                    new_state_dict[new_key] = old_state_dict[key]
                else:
                    new_state_dict[key] = old_state_dict[key]
        else:
            if 'model_state_dict' in checkpoint:
                new_state_dict = checkpoint['model_state_dict']
            else:
                new_state_dict = checkpoint

        model.load_state_dict(new_state_dict, strict=strict_match)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if loss_scaler is not None and 'scaler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scaler'])
        del checkpoint
        if main_process():
            logger.info(f"Successfully loaded weight: '{load_path}'")
            if scheduler is not None and optimizer is not None:
                logger.info(f"Resume training from: '{load_path}'")
    else:
        if main_process():
            raise RuntimeError(f"No weight found at '{load_path}'")
    return model, optimizer, scheduler, loss_scaler



def save_ckpt(cfg, model, optimizer, scheduler, curr_iter=0, curr_epoch=None, loss_scaler=None):
    """
    Save the model, optimizer, lr scheduler.
    """
    logger = logging.getLogger()

    if 'IterBasedRunner' in cfg.runner.type:
        max_iters = cfg.runner.max_iters
    elif 'EpochBasedRunner' in cfg.runner.type:
        max_iters = cfg.runner.max_epochs
    else:
        raise TypeError(f'{cfg.runner.type} is not supported')

    ckpt = dict(
        model_state_dict=model.module.state_dict(),
        optimizer=optimizer.state_dict(),
        max_iter=cfg.runner.max_iters if 'max_iters' in cfg.runner \
            else cfg.runner.max_epochs,
        scheduler=scheduler.state_dict(),
    )

    if loss_scaler is not None:
        ckpt.update(dict(scaler=loss_scaler.state_dict()))
    
    ckpt_dir = os.path.join(cfg.work_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    save_name = os.path.join(ckpt_dir, 'step%08d.pth' %curr_iter)
    saved_ckpts = glob.glob(ckpt_dir + '/step*.pth')
    torch.save(ckpt, save_name)

    # keep the last 8 ckpts
    if len(saved_ckpts) > 20:
        saved_ckpts.sort()
        os.remove(saved_ckpts.pop(0))
    
    logger.info(f'Save model: {save_name}')
