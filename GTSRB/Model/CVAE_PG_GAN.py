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
from torchvision import transforms

class Gen_Block(nn.Module):
	def __init__(self,c_in,c_out,k_size, stride, pad,nz,numLabels):
		super(Gen_Block,self).__init__()
		self.nz=nz
		self.numLabels=numLabels
		# self.upsample=nn.Upsample(scale_factor=2, mode='nearest')
		self.layer1=nn.Sequential(
					snconv2d(c_in, c_out, k_size, stride, pad),
					nn.ReLU()
                    )
		self.layer2=nn.Sequential(
					snconv2d(c_out, c_out, k_size, stride, pad),
					nn.ReLU()
                    )
		self.cbn1=ConditionalBatchNorm2d_for_skip_and_shared(num_features=c_out,
														 	z_dims_after_concat=self.nz+self.numLabels,
															spectral_norm=False)
		self.cbn2=ConditionalBatchNorm2d_for_skip_and_shared(num_features=c_out,
														 		  z_dims_after_concat=self.nz+self.numLabels,
																  spectral_norm=False)
		self.self_attn=Self_Attn(c_out,False)
	def forward(self,x,y):
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		x=self.layer1(x)
		x=self.cbn1(x,y)
		x=self.layer2(x)
		x=self.cbn2(x,y)
		x=self.self_attn(x)
		return x

class TO_RGB_Block(nn.Module):
	def __init__(self,c_in):
		super(TO_RGB_Block,self).__init__()
		self.c_in=c_in
		self.layer=nn.Sequential(
					snconv2d(c_in, 3, 3, 1, 1),
					nn.ReLU()
                    )
	def forward(self,x):
		x=self.layer(x)
		return x
class CVAE(nn.Module):
	def __init__(self,nz, imSize,block_num,
			  	in_channel=3,
				fSize=64, 
				numLabels=2,
				device='cpu'):
		super(CVAE, self).__init__()
		self.device=device
		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		self.block_num= block_num
		self.in_channel=in_channel
		inSize = imSize // (2 ** block_num)
		self.inSize = inSize
		self.numLabels = numLabels
		self.encLogVar = nn.Linear(2**(self.block_num-1)*fSize* inSize * inSize, nz)
		self.encMu = nn.Linear(2**(self.block_num-1)*fSize*inSize * inSize, nz)
		self.encY = nn.Linear(2**(self.block_num-1)*fSize*inSize * inSize, numLabels)
		self.enc_layers=nn.ModuleList()
		for i in range(self.block_num):
			if i==0:
				self.enc_layers.append(nn.Conv2d(self.in_channel,self.fSize,5,2,2))
			else:
				self.enc_layers.append(nn.Conv2d(self.fSize*(2**(i-1)),self.fSize*(2**i),5,2,2))   
		self.enc_attn_block=Self_Attn(self.fSize * (2**(self.block_num-1)),spectral_norm=False)
		self.dec_fc=nn.Linear(nz+numLabels,2**(self.block_num-1)*fSize* inSize * inSize)
		self.dec_blocks=nn.ModuleList()
		self.toRGB_blocks=nn.ModuleList()
		for i in range(self.block_num):	
			if i!=self.block_num-1:
				genblock=Gen_Block(c_in=2**(self.block_num-i-1)*fSize,
					  			c_out=2**(self.block_num-i-2)*fSize,
								k_size=3,
								stride=1,
								pad=1,
								nz=self.nz,
								numLabels=self.numLabels)
				self.toRGB_blocks.append(TO_RGB_Block(c_in=2**(self.block_num-i-2)*fSize))
			else:
				genblock=Gen_Block(c_in=2**(self.block_num-i-1)*fSize,
					  			c_out=2**(self.block_num-i-1)*fSize,
								k_size=3,
								stride=1,
								pad=1,
								nz=self.nz,
								numLabels=self.numLabels)
				self.toRGB_blocks.append(TO_RGB_Block(c_in=2**(self.block_num-i-1)*fSize))
			self.dec_blocks.append(genblock)	

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x=x.to(self.device)
		for layer in self.enc_layers:
			x=F.relu(layer(x))
		x = self.enc_attn_block(x)
		x = x.view(x.size(0), -1)
		print(x.shape)
		mu = self.encMu(x)  #no relu - mean may be negative
		log_var = self.encLogVar(x) #no relu - log_var may be negative
		y = F.softmax(self.encY(x.detach()))
		return mu, log_var, y

	def re_param(self, mu, log_var):
		#do the re-parameterising here
		sigma = torch.exp(log_var/2)  #sigma = exp(log_var/2) #torch.exp(log_var/2)
		eps =torch.randn(sigma.size(0), self.nz).to(self.device)
		return mu + sigma * eps  #eps.mul(simga)._add(mu)

	def decode(self, y, z, stage,alpha):
		y=y.to(self.device)
		z=z.to(self.device)
		z=torch.concat([z,y],dim=1)
		z0=z
		z=self.dec_fc(z)
		z=z.view(z.size(0), -1, self.inSize, self.inSize)
		#第一阶段和最后阶段训练
		if stage==1 :
			z=self.dec_blocks[0](z,z0)
			#print("Decoder{} output:{}".format(stage,z.shape))
			z=self.toRGB_blocks[0](z)
		#中间阶段
		else:
			#高分辨率层输出
			z_high=z
			for i in range(stage):
				z_high=self.dec_blocks[i](z_high,z0)
				#print("Decoder{} output:{}".format(i,z_high.shape))
			z_high=self.toRGB_blocks[stage-1](z_high)
			#低分辨率层输出和上采样
			z_low=z
			for i in range(stage-1):
				z_low=self.dec_blocks[i](z_low,z0)
			z_low=self.toRGB_blocks[stage-2](z_low)
			z_low= F.interpolate(z_low, scale_factor=2, mode='nearest')
			z=(1-alpha)*z_low+alpha*z_high
		return z
	
	def loss(self, rec_x, x, mu, logVar):
		#ssim=ssim_package.SSIM().to(self.device)
		BCE = F.binary_cross_entropy(rec_x, x, size_average=False)
		#MSE = F.mse_loss(rec_x, x, size_average=False)
		#SSIM= 100*(1-ssim(rec_x,x))/2
		KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar) #0.5 * sum(1 + log(var) - mu^2 - var)
		#return BCE/ (x.size(2) ** 2)+SSIM, KL / mu.size(1)
		return BCE/ (x.size(2) ** 2), KL / mu.size(1)
	def caculate_difference(self,x,y,class_nums,stage,alpha):
		x=x.to(self.device)
		y=y.to(self.device)
		mu, log_var, rec_y = self.encode(x)
		z = self.re_param(mu, log_var)
		# 解码器重构x
		rec_x = self.decode(rec_y, z,stage,alpha)
		# 解码器用标签重构x
		one_hot_y= torch.eye(class_nums)[torch.LongTensor(y.data.cpu().numpy())].type_as(z)
		dec_x = self.decode(one_hot_y,z,stage,alpha)
		diff=rec_x-dec_x
		#max_min_diff=(diff - diff.min()) / (diff.max() - diff.min()).detach()
		#return max_min_diff
		return 10*diff

	def forward(self, x,stage,alpha):
		# the outputs needed for training
		x=x.to(self.device)
		mu, log_var, y = self.encode(x)
		z = self.re_param(mu, log_var)
		reconstruction = self.decode(y,z,stage,alpha)

		return reconstruction, mu, log_var, y

	def save_params(self, modelDir,class_num):
		print ('saving params...')
		torch.save(self.state_dict(), join(modelDir, 'cVAE_PG_GAN_GTSRB_{}.pth'.format(class_num)))

	def load_params(self, modelDir,class_num):
		print ('loading params...')
		self.load_state_dict(torch.load(join(modelDir, 'cVAE_PG_GAN_GTSRB_{}.pth'.format(class_num))))

class Dis_Block(nn.Module):
	def __init__(self,c_in,c_out,k_size,stride,pad):
		super(Dis_Block,self).__init__()
		self.layer1=nn.Sequential(
					snconv2d(c_in, c_out, k_size, stride, pad),
					nn.LeakyReLU(0.2)
                    )
		self.layer2=nn.Sequential(
					snconv2d(c_out, c_out, k_size, stride, pad),
					nn.LeakyReLU(0.2)
                    )
	def forward(self,x):
		x=F.avg_pool2d(x, kernel_size=2, stride=2)
		x=self.layer1(x)
		x=self.layer2(x)
		return x

class FROM_RGB_Block(nn.Module):
	def __init__(self,c_out):
		super(FROM_RGB_Block,self).__init__()
		self.c_out=c_out
		self.layer=nn.Sequential(
					snconv2d(3,c_out, 3, 1, 1),
					nn.LeakyReLU(0.2)
                    )
	def forward(self,x):
		x=self.layer(x)
		return x


class DISCRIMINATOR(nn.Module):
	def __init__(self, imSize, block_num ,fSize=64,numLabels=1,device='cpu',):
		super(DISCRIMINATOR, self).__init__()
		#define layers here
		self.device=device
		self.fSize = fSize
		self.imSize = imSize
		self.block_num=block_num
		outSize = imSize // ( 2 ** block_num)
		self.numLabels = numLabels
		self.dis_blocks=nn.ModuleList()
		self.from_rgb_blocks=nn.ModuleList()
		self.dis_fc=nn.Linear((2**(self.block_num-1))*fSize*outSize*outSize,1)
		for i in range(self.block_num):	
			if i!=0:
				self.from_rgb_blocks.append(FROM_RGB_Block(2**(i-1)*fSize))
				disblock=Dis_Block(c_in=2**(i-1)*fSize,
					  			c_out=2**(i)*fSize,
								k_size=3,
								stride=1,
								pad=1
								)
				self.dis_blocks.append(disblock)
				
			else:
				self.from_rgb_blocks.append(FROM_RGB_Block(2**(i)*fSize))				
				disblock=Dis_Block(c_in=2**(i)*fSize,
					  			c_out=2**(i)*fSize,
								k_size=3,
								stride=1,
								pad=1
								)
				self.dis_blocks.append(disblock)

	def discriminate(self,x,stage,alpha):
		x=x.to(self.device)
		# print("000000000",x)
		if stage==1 :
			x=self.from_rgb_blocks[-1](x)
			x=self.dis_blocks[-1](x)
		else:
			x_high=x
			x_low=x
			# 计算当前层的输出
			x_low=self.from_rgb_blocks[self.block_num-stage](x_low)
			#print("From_rgb{} out:{}".format(self.block_num-stage,x_low.shape))
			x_low=self.dis_blocks[self.block_num-stage](x_low)
			#print("Disblock{} out:{}".format(self.block_num-stage,x_low.shape))
			# 使用下一层的输出
			x_high=F.avg_pool2d(x_high, kernel_size=2, stride=2)
			x_high=self.from_rgb_blocks[self.block_num-stage+1](x_high)
			#print("From_rgb{} out:{}".format(self.block_num-stage+1,x_high.shape))
			x=x_low*alpha+x_high*(1-alpha)
			#print(x.shape)
			for i in range(self.block_num-stage+1,self.block_num):
				x=self.dis_blocks[i](x)
				#print("Disblock{} out:{}".format(i,x.shape))
		print("==========",x.shape)
		x=x.view(x.size(0),-1)
		x=F.sigmoid(self.dis_fc(x))
		return x

	def forward(self, x,stage,alpha):
		# the outputs needed for training
		return self.discriminate(x,stage,alpha)


	def save_params(self, modelDir,class_num):
		print ('saving params...')
		torch.save(self.state_dict(), join(modelDir,'Discriminator_PG_GAN_GTSRB_{}.pth'.format(class_num)))


	def load_params(self, modelDir,class_num):
		print ('loading params...')
		self.load_state_dict(torch.load(join(modelDir,'Discriminator_PG_GAN_GTSRB_{}.pth'.format(class_num))))
























