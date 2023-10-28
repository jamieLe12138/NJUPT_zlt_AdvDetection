# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/misc.py


import numpy as np
import random
import math
import os
import sys
import shutil
import warnings
#import seaborn as sns
import matplotlib.pyplot as plt
from os.path import dirname, abspath, exists, join
from scipy import linalg
from datetime import datetime
from tqdm import tqdm
from itertools import chain
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image



class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


class Adaptive_Augment(object):
    def __init__(self, prev_ada_p, ada_target, ada_length, batch_size, rank):
        self.prev_ada_p = prev_ada_p
        self.ada_target = ada_target
        self.ada_length = ada_length
        self.batch_size = batch_size
        self.rank = rank

        self.ada_aug_step = self.ada_target/self.ada_length


    def initialize(self):
        self.ada_augment = torch.tensor([0.0, 0.0], device = self.rank)
        if self.prev_ada_p is not None:
            self.ada_aug_p = self.prev_ada_p
        else:
            self.ada_aug_p = 0.0
        return self.ada_aug_p


    def update(self, logits):
        ada_aug_data = torch.tensor((torch.sign(logits).sum().item(), logits.shape[0]), device=self.rank)
        self.ada_augment += ada_aug_data
        if self.ada_augment[1] > (self.batch_size*4 - 1):
            authen_out_signs, num_outputs = self.ada_augment.tolist()
            r_t_stat = authen_out_signs/num_outputs
            sign = 1 if r_t_stat > self.ada_target else -1
            self.ada_aug_p += sign*self.ada_aug_step*num_outputs
            self.ada_aug_p = min(1.0, max(0.0, self.ada_aug_p))
            self.ada_augment.mul_(0.0)
        return self.ada_aug_p


def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    kwargs = flatten_dict(kwargs)
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(train_configs, model_configs):
    cfgs = {}
    for k, v in chain(train_configs.items(), model_configs.items()):
        cfgs[k] = v

    class cfg_container: pass
    cfg_container.train_configs = train_configs
    cfg_container.model_configs = model_configs
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container


# fix python, numpy, torch seed
def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def setup(rank, world_size, backend="nccl"):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        # initialize the process group
        dist.init_process_group(backend,
                                init_method="tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']),
                                rank=rank,
                                world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def count_parameters(module):
    return 'Number of parameters: {}'.format(sum([p.data.nelement() for p in module.parameters()]))


def define_sampler(dataset_name, conditional_strategy, batch_size, num_classes):
    if conditional_strategy != "no":
        if dataset_name == "cifar10" or batch_size >= num_classes*8:
            sampler = "class_order_all"
        else:
            sampler = "class_order_some"
    else:
        sampler = "default"
    return sampler


def check_flags(train_configs, model_configs, n_gpus):
    if model_configs['train']['model']['architecture'] == "dcgan":
        assert model_configs['data_processing']['img_size'] == 32, "Sry,\
            StudioGAN does not support dcgan models for generation of images larger than 32 resolution."

    if train_configs['freeze_layers'] > -1:
        assert train_configs['checkpoint_folder'] is not None,\
            "Freezing discriminator needs a pre-trained model."

    if train_configs['distributed_data_parallel']:
        msg = "StudioGAN does not support image visualization, k_nearest_neighbor, interpolation, frequency, and tsne analysis with DDP. " +\
            "Please change DDP with a single GPU training or DataParallel instead."
        assert train_configs['image_visualization'] + train_configs['k_nearest_neighbor'] + train_configs['interpolation'] +\
            train_configs['frequency_analysis'] + train_configs['tsne_analysis'] == 0, msg

    if model_configs['train']['model']['conditional_strategy'] in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
        assert not train_configs['distributed_data_parallel'], \
        "StudioGAN does not support DDP training for NT_Xent_GAN, Proxy_NCA_GAN, and ContraGAN"

    if train_configs['train']*train_configs['standing_statistics']:
        print("When training, StudioGAN does not apply standing_statistics for evaluation. " + \
              "After training is done, StudioGAN will accumulate batchnorm statistics and evaluate the trained model")

    if model_configs['train']['model']['conditional_strategy'] == "ContraGAN":
        assert model_configs['train']['loss_function']['tempering_type'] == "constant" or \
            model_configs['train']['loss_function']['tempering_type'] == "continuous" or \
            model_configs['train']['loss_function']['tempering_type'] == "discrete", \
            "Tempering_type should be one of constant, continuous, or discrete."

    if model_configs['train']['model']['pos_collected_numerator']:
        assert model_configs['train']['model']['conditional_strategy'] == "ContraGAN", \
            "Pos_collected_numerator option is not appliable except for ContraGAN."

    if train_configs['distributed_data_parallel']:
        msg = 'Evaluation results of the image generation with DDP are not exact. ' + \
            'Please use a single GPU training mode or DataParallel for exact evluation.'
        warnings.warn(msg)

    if model_configs['data_processing']['dataset_name'] == 'cifar10':
        assert train_configs['eval_type'] in ['train', 'test'], "Cifar10 does not contain dataset for validation."

    elif model_configs['data_processing']['dataset_name'] in ['imagenet', 'tiny_imagenet', 'custom']:
        assert train_configs['eval_type'] == 'train' or train_configs['eval_type'] == 'valid', \
            "StudioGAN dose not support the evalutation protocol that uses the test dataset on imagenet, tiny imagenet, and custom datasets"

    assert train_configs['bn_stat_OnTheFly']*train_configs['standing_statistics'] == 0, \
        "You can't turn on train_statistics and standing_statistics simultaneously."

    assert model_configs['train']['optimization']['batch_size'] % n_gpus == 0, \
        "Batch_size should be divided by the number of gpus."

    assert int(model_configs['train']['training_and_sampling_setting']['diff_aug']) * \
        int(model_configs['train']['training_and_sampling_setting']['ada']) == 0, \
        "You can't simultaneously apply Differentiable Augmentation (DiffAug) and Adaptive Discriminator Augmentation (ADA)."

    assert int(train_configs['mixed_precision'])*int(model_configs['train']['loss_function']['gradient_penalty_for_dis']) == 0, \
        "You can't simultaneously apply mixed precision training (mpc) and Gradient Penalty for WGAN-GP."

    assert int(train_configs['mixed_precision'])*int(model_configs['train']['loss_function']['deep_regret_analysis_for_dis']) == 0, \
        "You can't simultaneously apply mixed precision training (mpc) and Deep Regret Analysis for DRAGAN."

    assert int(model_configs['train']['loss_function']['cr'])*int(model_configs['train']['loss_function']['bcr']) == 0 and \
        int(model_configs['train']['loss_function']['cr'])*int(model_configs['train']['loss_function']['zcr']) == 0, \
        "You can't simultaneously turn on Consistency Reg. (CR) and Improved Consistency Reg. (ICR)."

    assert int(model_configs['train']['loss_function']['gradient_penalty_for_dis'])* \
    int(model_configs['train']['loss_function']['deep_regret_analysis_for_dis']) == 0, \
        "You can't simultaneously apply Gradient Penalty (GP) and Deep Regret Analysis (DRA)."


# Convenience utility to switch off requires_grad
def toggle_grad(model, on, freeze_layers=-1):
    try:
        if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
            num_blocks = len(model.module.in_dims)
        else:
            num_blocks = len(model.in_dims)

        assert freeze_layers < num_blocks,\
            "can't not freeze the {fl}th block > total {nb} blocks.".format(fl=freeze_layers, nb=num_blocks)

        if freeze_layers == -1:
            for name, param in model.named_parameters():
                param.requires_grad = on
        else:
            for name, param in model.named_parameters():
                param.requires_grad = on
                for layer in range(freeze_layers):
                    block = "blocks.{layer}".format(layer=layer)
                    if block in name:
                        param.requires_grad = False
    except:
        for name, param in model.named_parameters():
            param.requires_grad = on


def set_bn_train(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()

def untrack_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = False

def track_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = True


def set_deterministic_op_train(m):
    if isinstance(m, torch.nn.modules.conv.Conv2d):
        m.train()

    if isinstance(m, torch.nn.modules.conv.ConvTranspose2d):
        m.train()

    if isinstance(m, torch.nn.modules.linear.Linear):
        m.train()

    if isinstance(m, torch.nn.modules.Embedding):
        m.train()


def reset_bn_stat(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.reset_running_stats()


def elapsed_time(start_time):
    now = datetime.now()
    elapsed = now - start_time
    return str(elapsed).split('.')[0]  # remove milliseconds


def reshape_weight_to_matrix(weight):
    weight_mat = weight
    dim =0
    if dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def find_string(list_, string):
    for i, s in enumerate(list_):
        if string == s:
            return i


def find_and_remove(path):
    if os.path.isfile(path):
        os.remove(path)


def calculate_all_sn(model):
    sigmas = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "bn" not in name and "shared" not in name and "deconv" not in name:
                if "blocks" in name:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'blocks')
                    block_idx = int(splited_name[int(idx+1)])
                    module_idx = int(splited_name[int(idx+2)])
                    operation_name = splited_name[idx+3]
                    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
                        operations = model.module.blocks[block_idx][module_idx]
                    else:
                        operations = model.blocks[block_idx][module_idx]
                    operation = getattr(operations, operation_name)
                else:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'module') if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel) else -1
                    operation_name = splited_name[idx+1]
                    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
                        operation = getattr(model.module, operation_name)
                    else:
                        operation = getattr(model, operation_name)

                weight_orig = reshape_weight_to_matrix(operation.weight_orig)
                weight_u = operation.weight_u
                weight_v = operation.weight_v
                sigmas[name] = torch.dot(weight_u, torch.mv(weight_orig, weight_v))
    return sigmas


def plot_img_canvas(images, save_path, nrow, logger, logging=True):
    directory = dirname(save_path)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_image(images, save_path, padding=0, nrow=nrow)
    if logging: logger.info("Saved image to {}".format(save_path))


def plot_pr_curve(precision, recall, run_name, logger, logging=True):
    directory = join('./figures', run_name)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "pr_curve.png")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(recall, precision)
    ax.grid(True)
    ax.set_xlabel('Recall (Higher is better)', fontsize=15)
    ax.set_ylabel('Precision (Higher is better)', fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path)
    if logging: logger.info("Save image to {}".format(save_path))
    return fig


def plot_spectrum_image(real_spectrum, fake_spectrum, run_name, logger, logging=True):
    directory = join('./figures', run_name)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "dfft_spectrum.png")

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(real_spectrum, cmap='viridis')
    ax1.set_title("Spectrum of real images")

    ax2.imshow(fake_spectrum, cmap='viridis')
    ax2.set_title("Spectrum of fake images")
    fig.savefig(save_path)
    if logging: logger.info("Save image to {}".format(save_path))




class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
    # def __call__(self, module, module_in, module_out):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []


def calculate_ortho_reg(m, rank):
    with torch.enable_grad():
        reg = 1e-6
        param_flat = m.view(m.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0]).to(rank)
        ortho_loss = reg * sym.abs().sum()
    return ortho_loss
