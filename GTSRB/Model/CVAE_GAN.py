import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import numpy as np

import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")

import os
from os.path import join
from Model.model_options import snconv2d,snlinear,sndeconv2d,Self_Attn,ConditionalBatchNorm2d_for_skip_and_shared
import pytorch_ssim as ssim_package

class DISCRIMINATOR(nn.Module):
	def __init__(self, imSize, fSize=2, numLabels=1,self_attn=False,d_spectral_norm=False,device='cpu',):
		super(DISCRIMINATOR, self).__init__()
		#define layers here
		self.device=device
		self.fSize = fSize
		self.imSize = imSize
		self.attn=self_attn
		self.d_spectral_norm=d_spectral_norm
		inSize = imSize // ( 2 ** 4)
		self.numLabels = numLabels
		if d_spectral_norm:
			self.dis1=snconv2d(3, fSize, 5, stride=2, padding=2)
			self.dis2=snconv2d(fSize, fSize * 2, 5, stride=2, padding=2)
			self.dis3 = snconv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
			self.dis4 = snconv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
			self.dis5 = snlinear((fSize * 8) * inSize * inSize, numLabels)

		else:
			self.dis1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
			self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
			self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
			self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
			self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, numLabels)
		if self_attn:
			self.attn_block=Self_Attn(fSize * 8,spectral_norm=False)

	def discriminate(self, x):
		x=x.to(self.device)
		x = F.relu(self.dis1(x))
		x = F.relu(self.dis2(x))
		x = F.relu(self.dis3(x))
		x = F.relu(self.dis4(x))
		if self.attn:
			x=self.attn_block(x)
		x = x.view(x.size(0), -1)
		if self.numLabels == 1:
			x = F.sigmoid(self.dis5(x))
		else:
			x = F.softmax(self.dis5(x))
		
		return x

	def forward(self, x):
		# the outputs needed for training
		return self.discriminate(x)


	def save_params(self, modelDir,class_num):
		print ('saving params...')
		torch.save(self.state_dict(), join(modelDir,'Discriminator_GTSRB_{}.pth'.format(class_num)))


	def load_params(self, modelDir,class_num):
		print ('loading params...')
		self.load_state_dict(torch.load(join(modelDir,'Discriminator_GTSRB_{}.pth'.format(class_num))))



class CVAE(nn.Module):

	def __init__(self,nz, imSize,
			  	in_channel=3,
				fSize=2, 
				numLabels=2,
				enc_self_attn=False,
				dec_self_attn=False,
				g_spectral_norm=False,
				CBN=False,device='cpu'):
		super(CVAE, self).__init__()
		#define layers here
		self.device=device
		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		inSize = imSize // (2 ** 4)
		self.inSize = inSize
		self.numLabels = numLabels
		# 使用自注意力块
		self.enc_attn=enc_self_attn
		self.dec_attn=dec_self_attn
		self.g_spectral_norm=g_spectral_norm

		# 使用条件归一化
		self.CBN=CBN
		
		self.enc1 = nn.Conv2d(in_channel, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		# 编码器使用自注意力机制
		if self.enc_attn:
			self.enc_attn_block=Self_Attn(fSize * 8,spectral_norm=False)

		self.encLogVar = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encMu = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encY = nn.Linear((fSize * 8) * inSize * inSize, numLabels)
		
		self.dec1 = nn.Linear(nz+numLabels, (fSize * 8) * inSize * inSize)

		# 解码器使用谱归一化
		if self.g_spectral_norm:
			self.dec2 = sndeconv2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
			self.dec3 = sndeconv2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
			self.dec4 = sndeconv2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
			self.dec5 = sndeconv2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)
		else:
			self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
			self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
			self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
			self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)
		# 解码器使用自注意力机制
		if self.dec_attn:
			self.dec_attn_block=Self_Attn(fSize*4 ,spectral_norm=False)

		# 条件归一化层
		if self.CBN:
			self.dec2b=ConditionalBatchNorm2d_for_skip_and_shared(num_features=fSize * 4,
														 		  z_dims_after_concat=self.nz+self.numLabels,
																  spectral_norm=False)
			self.dec3b=ConditionalBatchNorm2d_for_skip_and_shared(num_features=fSize * 2,
														 		  z_dims_after_concat=self.nz+self.numLabels,
																  spectral_norm=False)
			self.dec4b=ConditionalBatchNorm2d_for_skip_and_shared(num_features=fSize,
														 		  z_dims_after_concat=self.nz+self.numLabels,
																  spectral_norm=False)
		else:
			self.dec2b = nn.BatchNorm2d(fSize * 4)
			self.dec3b = nn.BatchNorm2d(fSize * 2)
			self.dec4b = nn.BatchNorm2d(fSize)

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x=x.to(self.device)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		if self.enc_attn:
			x = self.enc_attn_block(x)
		x = x.view(x.size(0), -1)
		mu = self.encMu(x)  #no relu - mean may be negative
		log_var = self.encLogVar(x) #no relu - log_var may be negative
		y = F.softmax(self.encY(x.detach()))
		
		return mu, log_var, y

	def re_param(self, mu, log_var):
		#do the re-parameterising here
		sigma = torch.exp(log_var/2)  #sigma = exp(log_var/2) #torch.exp(log_var/2)
		eps =torch.randn(sigma.size(0), self.nz).to(self.device)
	
		return mu + sigma * eps  #eps.mul(simga)._add(mu)

	def decode(self, y, z):
		y=y.to(self.device)
		z=z.to(self.device)
		#define the decoder here
		if self.CBN:
			z= torch.cat([y,z], dim=1)
			z0=z
			z = F.relu(self.dec1(z))
			z = z.view(z.size(0), -1, self.inSize, self.inSize)
			z = F.relu(self.dec2b(self.dec2(z),z0))
			if self.dec_attn:
				z=self.dec_attn_block(z)
			
			z = F.relu(self.dec3b(self.dec3(z),z0))			
			z = F.relu(self.dec4b(self.dec4(z),z0))
			
			z = F.sigmoid(self.dec5(z))

		else:
			z = torch.cat([y,z], dim=1)
			z = F.relu(self.dec1(z))
			z = z.view(z.size(0), -1, self.inSize, self.inSize)
			z = F.relu(self.dec2b(self.dec2(z)))
			
			z = F.relu(self.dec3b(self.dec3(z)))
			z = F.relu(self.dec4b(self.dec4(z)))
			if self.dec_attn:
				z=self.dec_attn_block(z)
			z = F.sigmoid(self.dec5(z))

		return z

	def loss(self, rec_x, x, mu, logVar):
		#ssim=ssim_package.SSIM().to(self.device)
		#Total loss is BCE(x, rec_x) + KL
		BCE = F.binary_cross_entropy(rec_x, x, size_average=False)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		#MSE = F.mse_loss(rec_x, x, size_average=False)
		#SSIM= 5*(1-ssim(rec_x,x))/2
		#(might be able to use nn.NLLLoss2d())
		KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar) #0.5 * sum(1 + log(var) - mu^2 - var)
		# return (BCE+MSE) / (x.size(2) ** 2)+SSIM, KL / mu.size(1)
		return BCE/ (x.size(2) ** 2), KL / mu.size(1)
	
	# def caculate_difference(self,x,y,class_nums):
	# 	x=x.to(self.device)
	# 	y=y.to(self.device)
	# 	mu, log_var, rec_y = self.encode(x)
	# 	z = self.re_param(mu, log_var)
	# 	# 解码器重构x
	# 	rec_x = self.decode(rec_y, z)
	# 	# 解码器用标签重构x
	# 	one_hot_y= torch.eye(class_nums)[torch.LongTensor(y.data.cpu().numpy())].type_as(z)
	# 	dec_x = self.decode(one_hot_y,z)
	# 	diff=rec_x-dec_x
	# 	max_min_diff=(diff - diff.min()) / (diff.max() - diff.min()).detach()
	# 	return max_min_diff
	def caculate_difference(self,x,y,class_nums):
		x=x.to(self.device)
		y=y.to(self.device)
		mu, log_var, rec_y = self.encode(x)
		z = self.re_param(mu, log_var)
		# 解码器用标签重构x
		one_hot_y= torch.eye(class_nums)[torch.LongTensor(y.data.cpu().numpy())].type_as(z)
		dec_x = self.decode(one_hot_y,z)
		diff=x-dec_x
		max_min_diff=(diff - diff.min()) / (diff.max() - diff.min()).detach()
		return max_min_diff



	def forward(self, x):
		# the outputs needed for training
		x=x.to(self.device)
		mu, log_var, y = self.encode(x)
		z = self.re_param(mu, log_var)
		reconstruction = self.decode(y, z)

		return reconstruction, mu, log_var, y

	def save_params(self, modelDir,class_num):
		print ('saving params...')
		torch.save(self.state_dict(), join(modelDir, 'cVAE_GAN_GTSRB_{}.pth'.format(class_num)))


	def load_params(self, modelDir,class_num):
		print ('loading params...')
		self.load_state_dict(torch.load(join(modelDir, 'cVAE_GAN_GTSRB_{}.pth'.format(class_num))))





















