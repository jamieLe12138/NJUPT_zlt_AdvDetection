#conditional VAE+GAN trained on smile/no smile faces -- info seperation!
import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")

from Utils.MISC import drawGTSRBImages
from Model.function import make_new_folder, plot_losses, vae_loss_fn, save_input_args, \
is_ready_to_stop_pretraining, sample_z, class_loss_fn, label_switch, plot_norm_losses #, one_hot
from Model.CVAE_GAN import CVAE,DISCRIMINATOR

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

from torchvision.datasets import GTSRB
from Utils.MISC import *
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小为统一大小
    transforms.ToTensor(),
])
def train_cVAE_GAN( selected_classes,
                    root="F:\ModelAndDataset\data",
                    train_batch_size=64,
                    test_batch_size=64,
                    nz=100,
                    fsize=64,
                    lr=2e-4,
                    alpha=1e-3,
                    momentum=0.9,
                    Epochs=30,
                    gamma=1,
                    rho=1,
                    delta=1,
                    save_model_dir='F:/ModelAndDataset/model/GTSRB/cVAE_GAN',# 模型参数存放目录
                    load_model=False,
                    result_dir = 'E:/Project/ZLTProgram/Images/cvae_gan_gtsrb'# 实验结果存放目录
                    ):
	
    torch.cuda.empty_cache()
    print ('Results will be saved to:',result_dir)
    ####### Data set #######
    print ('Prepare data loaders...')
    train_class_mapping,train_class_name_mapping,trainLoader=loadData_selected_labels(root=root,
                                                                                       selected_classes=selected_classes,
                                                                                       batch_size=train_batch_size,
                                                                                       train=True)
    test_class_mapping,test_class_name_mapping,testLoader=loadData_selected_labels(root=root,
                                                                                    selected_classes=selected_classes,
                                                                                    batch_size=test_batch_size,
                                                                                    train=False)
    print("Mapping")
    print(train_class_mapping)
    print(test_class_name_mapping)
    print ('Data loaders ready.')
    #GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:",device)
    ####### Create model #######
    cvae = CVAE(nz=nz,
			 imSize=64,
			 enc_self_attn=True,
			 dec_self_attn=True,
			 g_spectral_norm=False,
			 CBN=True,
			 fSize=fsize,
             numLabels=len(selected_classes),
			 device=device)
    dis = DISCRIMINATOR(imSize=64,
					    self_attn=False,
					    d_spectral_norm=True,
					    fSize=fsize,
					    device=device)
    if load_model:
        print("Load Pretrained Models!")
        cvae.load_params(modelDir=save_model_dir,class_num=len(selected_classes))
        dis.load_params(modelDir=save_model_dir,class_num=len(selected_classes))

    cvae.to(device)
    dis.to(device)

    print (cvae)
    print (dis)
    ####### Define optimizer #######
    optimizerCVAE = optim.RMSprop(cvae.parameters(), lr=lr)  #specify the params that are being upated
    optimizerDIS = optim.RMSprop(dis.parameters(), lr=lr, alpha=momentum)

    losses = {'total':[], 'kl':[], 'rec':[], 'dis':[], 'gen':[], 'test_rec':[], 'class':[],'test_class':[]}
    Ns = len(trainLoader)*train_batch_size  #no samples
    Nb = len(trainLoader)  #no batches
    ####### Start Training #######
    for epoch in range(Epochs):
        cvae.train()
        dis.train()
        
        epochLoss = 0
        epochLoss_kl = 0
        epochLoss_rec = 0
        epochLoss_dis = 0
        epochLoss_gen = 0
        epochLoss_class = 0
        TIME = time()

        for i, data in enumerate(trainLoader, 0):	
            x, y = data
            y=mapping_labels(train_class_mapping,y)
            x, y = x.to(device), y.to(device)
           
            rec_x, outMu, outLogVar, predY = cvae(x)
            z = cvae.re_param(outMu, outLogVar)
		    #VAE loss
		    # 使用重构图片和原图片计算损失
            rec_Loss, klLoss = cvae.loss(rec_x=rec_x, x=x, mu=outMu, logVar=outLogVar)
            vaeLoss = rec_Loss + alpha*klLoss
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
            sample_batch_size=x.size(0)
            zRand = sample_z(sample_batch_size, nz)
            # 生成独热编码矩阵
            yRand = torch.eye(len(selected_classes))[torch.LongTensor(y.data.cpu().numpy())].type_as(zRand)

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
			bce(predict_XRand, realLabel, size_average=False)) / predict_XRand.size(1)


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
            epochLoss_rec += rec_Loss.item()
            epochLoss_gen += genLoss.item()
            epochLoss_dis += disLoss.item()
		
            epochLoss_class += classLoss.item()


            if i%25 == 0:
                i+=1
                print ('[%d, %d] loss: %0.5f, rec: %0.5f, alpha*kl: %0.5f, gen: %0.5f, dis: %0.5f,class: %0.5f, time: %0.3f' % \
		 	    (epoch, i, epochLoss/i, epochLoss_rec/i ,alpha*epochLoss_kl/i, epochLoss_gen/i, epochLoss_dis/i, \
			    epochLoss_class/i, time() - TIME))
		
        cvae.eval()
        dis.eval()
	    # 创建测试结果目录
        if os.path.exists(result_dir)==False:
            os.mkdir(result_dir)

	    #Load test data
        testIter = iter(testLoader)
        xTest, yTest = next(testIter)
        yTest=mapping_labels(train_class_mapping,yTest)
		
        xTest = xTest.to(device).data
        yTest = yTest.to(device)
        outputs, outMu, outLogVar, outY = cvae(xTest)
        if (epoch+1)%5==0:
            drawGTSRBImages(xTest.cpu().numpy(),
			    	        yTest.cpu(),
                            test_class_name_mapping,
				            save_path=join(result_dir,'input.png'),
                            overwrite=True
				            )
            drawGTSRBImages(outputs.cpu().detach().numpy(),
			    	        yTest.cpu(),
                            test_class_name_mapping,
				            save_path=join(result_dir,"GTSRB_{}_output{}.png".format(len(selected_classes),epoch)),
                            overwrite=True
				            )

        (recLossTest), klLossTest = vae_loss_fn(rec_x=outputs, x=xTest, mu=outMu, logVar=outLogVar)
        maxVal, predLabel = torch.max(outY, 1)
        classScoreTest = torch.eq(predLabel, yTest).float().sum()/yTest.size(0)
        print ('classification test:', classScoreTest.item())

	    # 创建模型参数目录
        if os.path.exists(save_model_dir)==False:
            os.mkdir(save_model_dir)
        cvae.save_params(modelDir=save_model_dir,class_num=len(selected_classes))
        dis.save_params(modelDir=save_model_dir,class_num=len(selected_classes))
	
        losses['total'].append(epochLoss/Ns)
        losses['kl'].append(epochLoss_kl/Ns)
        losses['rec'].append(epochLoss_rec/Ns)
        losses['test_rec'].append((recLossTest).item()/xTest.size(0)) #append every epoch
        losses['dis'].append(epochLoss_dis/Ns)
        losses['gen'].append(epochLoss_gen/Ns)
        losses['class'].append(epochLoss_class/Ns)
        losses['test_class'].append(classScoreTest.item())

        if epoch > 1:
            plot_losses(losses, result_dir, epochs=epoch+1)
            plot_norm_losses(losses, result_dir, epochs=1+epoch)




