#conditional VAE+GAN trained on smile/no smile faces -- info seperation!
import sys
sys.path.append("E:/Project/ZLTProgram/")

from dataload import CELEBA
from function import make_new_folder, plot_losses, vae_loss_fn, save_input_args, \
is_ready_to_stop_pretraining, sample_z, class_loss_fn, label_switch, plot_norm_losses #, one_hot
from models import CVAE, DISCRIMINATOR

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce

from torchvision import transforms
from torchvision.utils import make_grid, save_image

import numpy as np
import os
from os.path import join
from PIL import Image

import matplotlib 
from matplotlib import pyplot as plt
from time import time

# 数据集存储目录
root='E:/Project/ModelAndDataset/data'
attr_name="Male"
batch_size=64
nz=100
fsize=64
lr=2e-4
alpha=1e-3
momentum=0.9
maxEpochs=10
gamma=1
rho=1
delta=1
outdir='E:/Project/ModelAndDataset/model/CelebA'

####### Data set #######
print ('Prepare data loaders...')
transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
trainDataset = CELEBA(root=root, train=True, train_ratio=0.7,transform=transforms.ToTensor(),label=attr_name)

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

testDataset = CELEBA(root=root, train=False,train_ratio=0.7 ,transform=transforms.ToTensor(),label=attr_name)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)
print ('Data loaders ready.')


####### Create model #######
cvae = CVAE(nz=nz, imSize=64, fSize=fsize)
dis = DISCRIMINATOR(imSize=64, fSize=fsize)
#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


cvae.to(device)
dis.to(device)
print (cvae)
print (dis)

####### Define optimizer #######
optimizerCVAE = optim.RMSprop(cvae.parameters(), lr=lr)  #specify the params that are being upated
optimizerDIS = optim.RMSprop(dis.parameters(), lr=lr, alpha=momentum)

####### Create a new folder to save results and model info #######
exDir = 'E:/Project/ZLTProgram/Images/cvae_gan'
print ('Outputs will be saved to:',exDir)


losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'test_bce':[], 'class':[], 'test_class':[]}
Ns = len(trainLoader)*batch_size  #no samples
Nb = len(trainLoader)  #no batches
####### Start Training #######
for epoch in range(maxEpochs):
	cvae.train()
	dis.train()

	epochLoss = 0
	epochLoss_kl = 0
	epochLoss_bce = 0
	epochLoss_dis = 0
	epochLoss_gen = 0
	epochLoss_class = 0

	TIME = time()

	for i, data in enumerate(trainLoader, 0):
			
		x, y = data
		x, y = x.to(device), y.to(device)
		rec_x, outMu, outLogVar, predY = cvae(x)
		#VAE loss
		# 使用重构图片和原图片计算损失
		bceLoss, klLoss = cvae.loss(rec_x=rec_x, x=x, mu=outMu, logVar=outLogVar)
		vaeLoss = bceLoss + alpha*klLoss

		#Classification loss  #not on reconstructed sample
		#计算编码器分类损失	
		classLoss = class_loss_fn(pred=predY, target=y) 
		vaeLoss += rho * classLoss
			
		#DIS loss
		# 计算判别器损失
		# 真实图片预测
		predict_Xreal = dis(x)
		# 生成图片预测
		predict_XRec = dis(rec_x.detach())
		zRand = sample_z(x.size(0), nz, device)
		# 生成独热编码矩阵
		yRand = torch.eye(2)[torch.LongTensor(y.data.cpu().numpy())].type_as(zRand)

		predict_XRand = dis(cvae.decode(yRand, zRand).detach())
		fakeLabel = torch.Tensor(predict_Xreal.size()).zero_().type_as(predict_Xreal)
		realLabel = torch.Tensor(predict_Xreal.size()).fill_(1).type_as(predict_Xreal)
		disLoss = 0.3 * (bce(predict_Xreal, realLabel, size_average=False) + \
			bce(predict_XRec, fakeLabel, size_average=False) + \
			bce(predict_XRand, fakeLabel, size_average=False)) / predict_Xreal.size(1)


		#GEN loss，计算生成器与判别器对抗损失
		predict_XRec = dis(rec_x)
		predict_XRand = dis(cvae.decode(yRand, zRand))
		genLoss = 0.5 * (bce(predict_XRec, realLabel,size_average=False) +\
			bce(predict_XRand, realLabel, size_average=False)) / predict_XRec.size(1)

		#include the GENloss (the encoder loss) with the VAE loss
		vaeLoss += delta * genLoss

		#zero the grads - otherwise they will be acculated
		#fill in grads and do updates:
		optimizerCVAE.zero_grad()
		vaeLoss.backward() #fill in grads
		optimizerCVAE.step()

		optimizerDIS.zero_grad()
		disLoss.backward()
		optimizerDIS.step()

		epochLoss += vaeLoss.item()
		epochLoss_kl += klLoss.item()
		epochLoss_bce += bceLoss.item()
		epochLoss_gen += genLoss.item()
		epochLoss_dis += disLoss.item()
		epochLoss_class += classLoss.item()


		if i%100 == 0:
			i+=1
			print ('[%d, %d] loss: %0.5f, bce: %0.5f, alpha*kl: %0.5f, gen: %0.5f, dis: %0.5f, class: %0.5f, time: %0.3f' % \
		 	(epoch, i, epochLoss/i, epochLoss_bce/i,alpha*epochLoss_kl/i, epochLoss_gen/i, epochLoss_dis/i, \
			epochLoss_class/i, time() - TIME))


	#generate samples after each 10 epochs
	if epoch % 10 == 0:
		cvae.eval()
		dis.eval()

		#Load test data
		testIter = iter(testLoader)
		xTest, yTest = next(testIter)
		yTest = yTest
		
		xTest = xTest.to(device).data
		yTest = yTest.to(device)
		outputs, outMu, outLogVar, outY = cvae(xTest)

		print ('saving a set of samples')
		
		z = torch.randn(xTest.size(0), nz).to(device)
		

		y_Positive = torch.eye(2)[torch.LongTensor(np.ones(yTest.size()).astype(int))].type_as(z)
		samples = cvae.decode(y_Positive, z).cpu()
		save_image(samples.data, join(exDir,attr_name+'_epoch'+str(epoch)+'.png'))

		y_Negative = torch.eye(2)[torch.LongTensor(np.zeros(yTest.size()).astype(int))].type_as(z)
		samples = cvae.decode(y_Negative, z).cpu()
		save_image(samples.data, join(exDir,"not_"+attr_name+'_epoch'+str(epoch)+'.png'))

		#check reconstructions after each 10 epochs
		outputs, outMu, outLogVar, outY = cvae(xTest)

		bceLossTest, klLossTest = vae_loss_fn(rec_x=outputs, x=xTest, mu=outMu, logVar=outLogVar)
		maxVal, predLabel = torch.max(outY, 1)
		classScoreTest = torch.eq(predLabel, yTest).float().sum()/yTest.size(0)
		print ('classification test:', classScoreTest.item())

		save_image(xTest, join(exDir,'input.png'))
		save_image(outputs.data, join(exDir,'output_'+str(epoch)+'.png'))

		label_switch(xTest, yTest, cvae, exDir=exDir)

		cvae.save_params(exDir=exDir)

		losses['total'].append(epochLoss/Ns)
		losses['kl'].append(epochLoss_kl/Ns)
		losses['bce'].append(epochLoss_bce/Ns)
		losses['test_bce'].append((bceLossTest).item()/xTest.size(0)) #append every epoch
		losses['dis'].append(epochLoss_dis/Ns)
		losses['gen'].append(epochLoss_gen/Ns)
		losses['class'].append(epochLoss_class/Ns)
		losses['test_class'].append(classScoreTest.item())

		if epoch > 1:
			plot_losses(losses, exDir, epochs=epoch+1)
			plot_norm_losses(losses, exDir, epochs=1+epoch)




