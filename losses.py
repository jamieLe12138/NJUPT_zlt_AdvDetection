import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import autograd


def loss_dcgan_dis(dis_out_real, dis_out_fake):
    device = dis_out_real.get_device()
    ones = torch.ones_like(dis_out_real, device=device, requires_grad=False)
    dis_loss = -torch.mean(nn.LogSigmoid()(dis_out_real) + nn.LogSigmoid()(ones - dis_out_fake))
    return dis_loss


def loss_dcgan_gen(gen_out_fake):
    return -torch.mean(nn.LogSigmoid()(gen_out_fake))


def loss_lsgan_dis(dis_out_real, dis_out_fake):
    dis_loss = 0.5*(dis_out_real - torch.ones_like(dis_out_real))**2 + 0.5*(dis_out_fake)**2
    return dis_loss.mean()


def loss_lsgan_gen(dis_out_fake):
    gen_loss = 0.5*(dis_out_fake - torch.ones_like(dis_out_fake))**2
    return gen_loss.mean()


def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))


def loss_hinge_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)


def loss_wgan_dis(dis_out_real, dis_out_fake):
    return torch.mean(dis_out_fake - dis_out_real)


def loss_wgan_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)

def loss_recon(recon_x,x,mean,logstd,device):
    MSECriterion = nn.MSELoss().to(device)
    MSE = MSECriterion(recon_x,x)
    logvar = 2 * logstd
    KLD = -0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - torch.exp(logvar))
    return MSE+KLD
