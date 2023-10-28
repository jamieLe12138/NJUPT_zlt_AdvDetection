import numpy as np
import os
import glob
import random
from scipy import ndimage
from sklearn.manifold import TSNE
from os.path import join
from tqdm import tqdm
import torch
from torch.nn import DataParallel
import torch.distributed as dist
import torch.autograd
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torchvision import transforms		
from data_utils.load_dataset import *
import torch.optim as optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from studioGAN.biggan_utils import *
from studioGAN.CVAE_BigGAN import Encoder_cifar10,Generator_cifar10,Discriminator_cifar10,VAE
import studioGAN.losses as losses
from studioGAN.Misc import *
from studioGAN.diff_aug import *
import studioGAN.pytorch_ssim as ssim_package
# 创建文件夹
if not os.path.exists('./img_CVAE-GAN_Cifar10'):
    os.mkdir('./img_CVAE-GAN_Cifar10')
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size =128
z_dimension=80


train_dataset = LoadDataset("cifar10", "E:/Project/ModelAndDataset/data", train=True, download=True, resize_size=32,
								random_flip=True)
# img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# cifar10 = datasets.CIFAR10(
#     root="E:/Project/ModelAndDataset/data", train=True, transform=img_transform, download=True
# )
# # 创建一个包含前6400张图像的子集
# subset_indices = range(12800)  # 选择所需数量的图像
# dataset = Subset(cifar10, subset_indices)

# data loader 数据载入
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True)

train_iter=iter(train_dataloader)
# 初始化变分自编码器，分类器与判别器
Encoder=Encoder_cifar10(device=device)
Gen=Generator_cifar10(device=device)
Dis=Discriminator_cifar10(device=device)
Gen_copy=Generator_cifar10(device=device)
Gen_ema=ema(Gen, Gen_copy,0.9999,1000)
vae = VAE().to(device)

#  ========================加载预训练模型  ==============================      
Encoder.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Encoder.pth'))
Gen.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Decoder.pth'))
Dis.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Discriminator.pth'))
vae.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Vae.pth'))
# torch.save(Encoder.state_dict(), 'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Encoder.pth')
# torch.save(vae.state_dict(), 'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Vae.pth')
# torch.save(Gen.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Decoder.pth')
# torch.save(Dis.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Discriminator.pth')

if not os.path.exists('./img_CVAE-GAN_Cifar10'):
        os.mkdir('./img_CVAE-GAN_Cifar10')
cudnn.benchmark=True

# 初始化优化器
G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), 0.0002, [0.5, 0.999], eps=1e-6)
D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), 0.0002, [0.5, 0.999], eps=1e-6)
opt_encoder = torch.optim.Adam([{'params':Encoder.parameters()},{'params':vae.parameters()}], 0.0002, [0.5, 0.999], eps=1e-6)
		
# 初始化损失函数
D_loss = losses.loss_hinge_dis
g_loss = losses.loss_hinge_gen
ssim=ssim_package.SSIM().to(device)
		


Dis.train()
Encoder.train()
Gen.train()
step_count=0
total_step=2000
d_steps_per_iter=3
g_steps_per_iter=1

# 训练cVAE-GAN
while step_count <= total_step:
	print("Current step_count:",step_count)
	toggle_grad(Dis, on=True, freeze_layers=-1)
	toggle_grad(Gen, on=False, freeze_layers=-1)
	# ================== TRAIN D ================== #
	for step_index in range(d_steps_per_iter):
		# 清空编码器和判别器的梯度
		opt_encoder.zero_grad()
		D_optimizer.zero_grad()	
		# 处理真实数据
		try:
			real_images, real_labels = next(train_iter)
		except StopIteration:
			train_iter = iter(train_dataloader)
			real_images, real_labels = next(train_iter)
		real_images=real_images.to(device)
		real_labels=real_labels.to(device)

		real_images_for_gen = real_images
		real_labels_for_gen = real_labels
		# 对真实数据应用数据增强
		real_images = DiffAugment(real_images, policy="color,translation,cutout")
		real_images_without_noise = real_images
		# 随机噪声注入
		flag = np.random.random()
		if flag > 0.2:
			epsilon = np.random.random()*0.51
			real_images = torch.empty_like(real_images, dtype=real_images.dtype).uniform_(-epsilon, +epsilon) + real_images
			real_images = torch.clamp(real_images, min=-1, max=1)
		# 通过编码器和VAE获取潜在表示		
		latent_i = Encoder(real_images)
		z_mean, z_log_var, zs = vae(latent_i)
		# 获取用于生成的潜在表示
		latent_i_ori = Encoder(real_images_for_gen)
		z_mean_ori, z_log_var_ori, zs_ori = vae(latent_i_ori)
		# 生成假数据
		fake_labels = real_labels
		fake_images = Gen(zs, fake_labels)
		# 创建错误标签
		Int_Modi = random.randint(1, 9)
		wrong_labels = ((real_labels + Int_Modi) % 10).to(device)
		# 用错误标签生成假数据
		fake_images_wrong_labels_ori  = Gen(zs_ori, wrong_labels)
		fake_images_2d_ori = Gen(zs_ori, real_labels)
		# 对生成的假数据应用数据增强
		fake_images = DiffAugment(fake_images, policy="color,translation,cutout")
						
		# 判别器的前向传播				
		dis_out_real = Dis(real_images_without_noise, real_labels)# compare fake with clean images
		dis_out_fake = Dis(fake_images, fake_labels)
		# 计算判别器的损失		
		dis_acml_loss = D_loss(dis_out_real, dis_out_fake)
		dis_acml_loss_watch = dis_acml_loss
		# 反向传播和更新判别器参数
		print("Dis Step{} dis_acml_loss: {}".format(step_index,dis_acml_loss))
		dis_acml_loss.backward()
		D_optimizer.step()

	toggle_grad(Dis, False, freeze_layers=-1)
	toggle_grad(Gen, True, freeze_layers=-1)
	# ================== TRAIN G ================== #
	for step_index in range(g_steps_per_iter):
		opt_encoder.zero_grad()
		G_optimizer.zero_grad()

		try:
			 # 尝试从数据加载器中获取下一个数据批次
			real_images, real_labels = next(train_iter)
		except StopIteration:
			# 如果没有更多的数据批次可用，重新初始化数据加载器并获取下一个数据批次
			train_iter = iter(train_dataloader)
			real_images, real_labels = next(train_iter)
		real_images=real_images.to(device)
		real_labels=real_labels.to(device)

		# 添加噪音到真实图像
		real_images_without_noise = real_images
		flag = np.random.random()
		if flag > 0.2:
			epsilon = np.random.random()
			real_images = torch.empty_like(real_images, dtype=real_images.dtype).uniform_(-epsilon, +epsilon) + real_images
			real_images = torch.clamp(real_images, min=-1, max=1)
			# 通过编码器获取潜在表示	
			latent_i = Encoder(real_images)
			# print("real_images min:", real_images.min())
			z_mean, z_log_var, zs = vae(latent_i)
			fake_labels = real_labels
			# 使用生成器生成虚假图像,对虚假图像进行数据增强
			fake_images_ori = Gen(zs, fake_labels)
			fake_images = DiffAugment(fake_images_ori, policy="color,translation,cutout")
			# 使用判别器评估虚假图像
			dis_out_fake = Dis(fake_images, fake_labels)
							
			 # 计算结构相似度损失、潜在空间损失、L2损失和L1损失			
			loss_img_ssim = -torch.log(ssim(fake_images_ori, real_images_without_noise) + 1e-15)
			loss_lat = 0.5 * (z_log_var.exp() + z_mean ** 2 - 1 - z_log_var).mean()
			loss_img_l2 = torch.abs((fake_images_ori - real_images_without_noise)**2).mean()#torch.nn.MSELoss(fake_images_ori, real_images).mean() #t
			loss_img_l1 = torch.abs(fake_images_ori - real_images_without_noise).mean()
			# 计算生成器的损失项			
			gen_acml_loss_watch_g_loss = g_loss(dis_out_fake)
			gen_l2_loss_watch = loss_img_l2
			gen_ssim_loss_watch = 5*loss_img_ssim
			gen_lat_loss = loss_lat
			gen_l1_loss = loss_img_l1

			gen_acml_loss = g_loss(dis_out_fake) + loss_img_l2 + 5 * loss_img_ssim + loss_lat + loss_img_l1 
			print("Gen Step{} gen_acml_loss: {}".format(step_index,gen_acml_loss))					
			gen_acml_loss.backward()
				
			G_optimizer.step()
			opt_encoder.step()				
			Gen_ema.update(step_count)

	step_count=step_count+1		
	if (step_count+1)%50==0:
		train_iter = iter(train_dataloader)
		real_images, real_labels = next(train_iter)
		real_images0=real_images.cpu()
		make_grid(real_images0, nrow=8, normalize=True).detach()
		save_image(real_images, './img_CVAE-GAN_Cifar10/real_images{}.png'.format(step_count+1)) 
		z= Encoder(real_images.to(device))
		output = Gen(z,real_labels)
		fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
		save_image(fake_images, './img_CVAE-GAN_Cifar10/fake_images-{}.png'.format(step_count + 1))
	if(step_count+1)%200==0:
		torch.save(Encoder.state_dict(), 'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Encoder.pth')
		torch.save(vae.state_dict(), 'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Vae.pth')
		torch.save(Gen.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Decoder.pth')
		torch.save(Dis.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Discriminator.pth')



				

		