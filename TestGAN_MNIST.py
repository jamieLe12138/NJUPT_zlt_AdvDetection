# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from Generator_MNIST import generator_mnist
import os
batch_size=128
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 创建对象
G = generator_mnist(z_dimension=100)
G.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/generator_CGAN_mnist.pth'))
G = G.to(device)

outputs = []
for num in range(10):
    label = torch.Tensor([num]).repeat(10).long()
    label_onehot = torch.zeros((10,10))
    label_onehot[torch.arange(10),label]=1
    z = torch.randn((10,100),device=device)
    z = torch.cat([z,label_onehot.to(device)],1)
    outputs.append(G(z).view(z.shape[0],1,28,28))
outputs = torch.cat(outputs)
img = make_grid(outputs,nrow=10,normalize=False).clamp(0,1).detach().cpu().numpy()
plt.imshow(np.transpose(img,(1,2,0)),interpolation='nearest')
plt.show()
