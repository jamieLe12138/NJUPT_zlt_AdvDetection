#conditional VAE+GAN trained on smile/no smile faces -- info seperation!
import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")

from Utils.MISC import drawGTSRBImages
from Model.function import make_new_folder, plot_losses, vae_loss_fn, save_input_args, \
is_ready_to_stop_pretraining, sample_z, class_loss_fn, label_switch, plot_norm_losses #, one_hot
#from Model.CVAE_GAN import CVAE,DISCRIMINATOR
from Model.CVAE_PG_GAN import CVAE,DISCRIMINATOR
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
from torchvision.transforms import InterpolationMode
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
                    beta=1e-3,
                    momentum=0.9,
                    stages=4,
                    epochs_per_stage=20,
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
    #device = 'cpu'
    print("device:",device)
    cvae_pg_gan=CVAE(nz=100,
			 imSize=64,
             block_num=stages,
             in_channel=3,
			 fSize=fsize,
             numLabels=len(selected_classes),
			 device=device)
    dis= DISCRIMINATOR(imSize=64,
                    block_num=stages,
                    fSize=fsize,
                    numLabels=1,
                    device=device)
    if load_model:
        print("Load Pretrained Models!")
        cvae_pg_gan.load_params(modelDir=save_model_dir,class_num=len(selected_classes))
        dis.load_params(modelDir=save_model_dir,class_num=len(selected_classes))

    cvae_pg_gan.to(device)
    dis.to(device)

    print (cvae_pg_gan)
    print (dis)
    ####### Define optimizer #######
    optimizerCVAE = optim.RMSprop(cvae_pg_gan.parameters(), lr=lr)  #specify the params that are being upated
    optimizerDIS = optim.RMSprop(dis.parameters(), lr=lr, alpha=momentum)

    losses = {'total':[], 'kl':[], 'rec':[], 'dis':[], 'gen':[], 'test_rec':[], 'class':[],'test_class':[]}
    Ns = len(trainLoader)*train_batch_size  #no samples
    Nb = len(trainLoader)  #no batches
    ####### Start Training #######
    for current_stage in range(1,stages+2):
        cvae_pg_gan.train()
        dis.train()
        for epoch in range(epochs_per_stage):    
            epochLoss = 0
            epochLoss_kl = 0
            epochLoss_rec = 0
            epochLoss_dis = 0
            epochLoss_gen = 0
            epochLoss_class = 0
            TIME = time()
            if current_stage==1 or current_stage==stages+1:
                alpha=1
            else:
                alpha=epoch/epochs_per_stage
            print("Alpha:",alpha)
            for i, data in enumerate(trainLoader, 0):	
                x, y = data
                y=mapping_labels(train_class_mapping,y)
                x, y = x.to(device), y.to(device)
                
                
                rec_x, outMu, outLogVar, predY = cvae_pg_gan(x,current_stage,beta)
                
                reshape_x=transforms.Resize((rec_x.size(2), rec_x.size(3)), InterpolationMode.BILINEAR)(x).to(device)
                
                #rec_x = torch.clamp(rec_x, min=0, max=1)  
                reshape_x = torch.clamp(reshape_x, min=0, max=1)  
                z = cvae_pg_gan.re_param(outMu, outLogVar)
		        #VAE loss
		        # 使用重构图片和原图片计算损失
                rec_Loss, klLoss = cvae_pg_gan.loss(rec_x=rec_x, x=reshape_x, mu=outMu, logVar=outLogVar)
                vaeLoss = rec_Loss + beta*klLoss
                #Classification loss  #not on reconstructed sample
		        #计算编码器分类损失	
                classLoss = class_loss_fn(pred=predY, target=y) 
                vaeLoss += rho * classLoss
			
                #DIS loss
		        # 计算判别器损失
		        # 真实图片预测
                # print("reshape_x:",reshape_x.shape)
                # predict_Xreal = dis(reshape_x,current_stage,alpha)
	            # 生成图片预测
                predict_XRec = dis(rec_x.detach(),current_stage,alpha)
                #print("reshape_x:",reshape_x.shape)
                predict_Xreal = dis(reshape_x,current_stage,alpha)
                sample_batch_size=x.size(0)
                zRand = sample_z(sample_batch_size, nz)
                # 生成独热编码矩阵
                yRand = torch.eye(len(selected_classes))[torch.LongTensor(y.data.cpu().numpy())].type_as(zRand)

                predict_XRand = dis(cvae_pg_gan.decode(yRand, zRand,current_stage,alpha).detach(),current_stage,alpha)
                fakeLabel = torch.Tensor(predict_Xreal.size()).zero_().type_as(predict_Xreal)
                realLabel = torch.Tensor(predict_Xreal.size()).fill_(1).type_as(predict_Xreal)
                disLoss = 0.3 * (bce(predict_Xreal, realLabel, size_average=False) + \
			    bce(predict_XRec, fakeLabel, size_average=False) + \
			    bce(predict_XRand, fakeLabel, size_average=False)) / predict_Xreal.size(1)
		        #GEN loss，计算生成器与判别器对抗损失
                predict_XRec = dis(rec_x,current_stage,alpha)
                predict_XRand = dis(cvae_pg_gan.decode(yRand, zRand,current_stage,alpha),current_stage,alpha)
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
                    print ('[%d,%d,%d] loss: %0.5f, rec: %0.5f,beta*kl: %0.5f, gen: %0.5f, dis: %0.5f,class: %0.5f, time: %0.3f' % \
		 	        (current_stage,epoch,i,epochLoss/i, epochLoss_rec/i ,beta*epochLoss_kl/i, epochLoss_gen/i, epochLoss_dis/i, \
			        epochLoss_class/i, time() - TIME))
		
            cvae_pg_gan.eval()
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
            outputs, outMu, outLogVar, outY = cvae_pg_gan(xTest,current_stage,alpha)
            # yDiff = torch.randint(0, len(selected_classes)-1, yTest.shape)

            # diff1=cvae_pg_gan.caculate_difference(xTest,yTest,len(selected_classes),current_stage,beta)
            # diff2=cvae_pg_gan.caculate_difference(xTest,yDiff,len(selected_classes),current_stage,beta)  

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
                # drawGTSRBImages(diff1.cpu().detach().numpy(),
                #     	          yTest.cpu(),
                #                 test_class_name_mapping,
                # 	            save_path=join(result_dir,"GTSRB_{}_diffsame{}.png".format(len(selected_classes),epoch)),
                #                 overwrite=True
                # 	            )
                # drawGTSRBImages(diff2.cpu().detach().numpy(),
                #     	        yDiff.cpu(),
                #                 test_class_name_mapping,
                # 	            save_path=join(result_dir,"GTSRB_{}_diffrand{}.png".format(len(selected_classes),epoch)),
                #                 overwrite=True
                # 	            )
            reshape_xTest=transforms.Resize((outputs.size(2), outputs.size(3)), InterpolationMode.BILINEAR)(xTest).to(device)
            reshape_xTest = torch.clamp(reshape_xTest, min=0, max=1)


            (recLossTest), klLossTest = cvae_pg_gan.loss(rec_x=outputs, x=reshape_xTest, mu=outMu, logVar=outLogVar)
            
            maxVal, predLabel = torch.max(outY, 1)
            classScoreTest = torch.eq(predLabel, yTest).float().sum()/yTest.size(0)
            print ('classification test:', classScoreTest.item())

	        # 创建模型参数目录
            if os.path.exists(save_model_dir)==False:
                os.mkdir(save_model_dir)
            cvae_pg_gan.save_params(modelDir=save_model_dir,class_num=len(selected_classes))
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



