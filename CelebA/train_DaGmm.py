import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from time import time
import datetime
import sys
sys.path.append("E:/Project/ZLTProgram/CelebA")
from Utils.dataload import *
from Model.DAGMM import *
from Model.CVAE_GAN_AdvancedV2 import *
import matplotlib.pyplot as plt
from Utils.AE_Util import Adversarial_Examples_Generator
from Model import Target_model 
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import *
from torch.utils.data import TensorDataset, DataLoader
attr_name='Smiling'
batch_size=64
trainDataset = CELEBA(root='E:/Project/ModelAndDataset/data', train=True,train_ratio=0.7 ,transform=transforms.ToTensor(),label=attr_name)
trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=False)
num_epochs=3
iter_ctr = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# DAGMM模型
dagmm=DAGMM(3,64,64)
dagmm.to(device)
optimizer=torch.optim.Adam(dagmm.parameters(), lr=1e-6)
load_model=True

# CVAE_GAN模型
nz=100
fsize=64

cvae = CVAE(nz=nz,
			 imSize=64,
			 enc_self_attn=True,
			 dec_self_attn=True,
			 g_spectral_norm=False,
			 CBN=True,
			 fSize=fsize,
			 device=device)
cvae.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae.to(device)

losses = {'epoch_total_loss':[], 'epoch_sample_energy':[], 'epoch_recon_error':[], 'epoch_cov_diag':[]}

TIME = time()
# ============training=====================
if load_model:
    dagmm.load_params('E:\Project\ModelAndDataset\model\CelebA\DAGMM')
else:
    for epoch in range(0, num_epochs):
    
        dagmm.train()
        cvae.eval()
        epoch_total_loss=0
        epoch_sample_energy=0
        epoch_recon_error=0
        epoch_cov_diag=0
        for i, (x, y) in enumerate(trainLoader):
            difference=cvae.caculate_difference(x,y)
            with torch.no_grad():
                difference=difference.to(device)
            enc, dec, dagmm_z, gamma = dagmm(difference)
            # print(enc.shape)
            # print(dec.shape)
            # 计算损失
            total_loss, sample_energy, recon_error, cov_diag =dagmm.loss_function(difference, dec, dagmm_z, gamma, lambda_energy=0.1, lambda_cov_diag=0.005)
            epoch_total_loss+=total_loss
            epoch_sample_energy+=sample_energy
            epoch_recon_error+=recon_error
            epoch_cov_diag+=cov_diag
            # 反向传播
            dagmm.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(dagmm.parameters(), 5)
            optimizer.step()
            if i%100 == 0:
                i+=1
                print ('[{}, {}] total_loss: {}, sample_energy: {}, recon_error: {}, cov_diag: {},time: {}'.format
		 	    (epoch, i, epoch_total_loss/i,
                epoch_sample_energy/i,
                epoch_recon_error/i,
                epoch_cov_diag/i,
                time() - TIME
                ))
    dagmm.save_params('E:\Project\ModelAndDataset\model\CelebA\DAGMM')
dagmm.eval()
# ============test=====================
N = 0
# mu_sum = torch.zeros([2,3]).to(device)
# cov_sum = torch.zeros([2,3,3]).to(device)
# gamma_sum = torch.zeros([2]).to(device)

mu_sum=0
cov_sum=0
gamma_sum=0
for it, (x, y) in enumerate(trainLoader):
    #print("it:",it)
    difference=cvae.caculate_difference(x,y)
    with torch.no_grad():
        difference=difference.to(device)
        enc, dec, dagmm_z, gamma = dagmm(difference)
        phi, mu, cov = dagmm.compute_gmm_params(dagmm_z, gamma)  

        batch_gamma_sum = torch.sum(gamma, dim=0)    
        gamma_sum+=batch_gamma_sum
        mu_sum+=mu*batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
        cov_sum+=cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only    
        N += difference.size(0)
    if it%100==0:
        print("Batch:",it)
        print("gamma_sum :\n",gamma_sum)
        print("mu_sum :\n",mu_sum)
        print("cov_sum:\n",cov_sum)


            
train_phi = gamma_sum / N
train_mu = mu_sum / gamma_sum.unsqueeze(-1)
train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

print("N:",N)
print("phi :\n",train_phi)
print("mu :\n",train_mu)
print("cov :\n",train_cov)
print("=============Caculating train_energy=======================")
train_energy = []
train_labels = []
train_z = []
for it, (x,y) in enumerate(trainLoader):
    difference=cvae.caculate_difference(x,y)
    with torch.no_grad():
        difference=difference.to(device)
    enc, dec, dagmm_z, gamma = dagmm(difference)
    sample_energy, cov_diag = dagmm.compute_energy(dagmm_z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
    # 计算训练集能量        
    train_energy.append(sample_energy.data.cpu().numpy())
    train_z.append(dagmm_z.data.cpu().numpy())
    train_labels.append(torch.zeros_like(y).numpy())


train_energy = np.concatenate(train_energy,axis=0)
train_z = np.concatenate(train_z,axis=0)
train_labels = np.concatenate(train_labels,axis=0)


model_name='resnet18'
clip_values = (0.0, 1.0)
dagmm.eval()
# 定义模型
if model_name=='resnet18':
    target_model=Target_model.ResNet18(2)
elif model_name=='vgg19':
    target_model=Target_model.VGG_19(2)
elif model_name=='densenet169':
    target_model=Target_model.Densenet169(2)
elif model_name=='mobilenet':
    target_model=Target_model.MobileNet(2)
# 定义优化器
if model_name=='resnet18' or model_name=='densenet169':
    optimizer='Adam'
elif model_name=='vgg19' : 
    optimizer="SGD"
elif model_name=='mobilenet':
    optimizer="RMSprop"
#加载目标模型
target_model_dir="E:\Project\ModelAndDataset\model\CelebA"
target_model.load_state_dict(torch.load(join(target_model_dir,"CelebA_"+model_name+"_"+attr_name+".pth")))
estimator=PyTorchClassifier(model=target_model,loss=nn.CrossEntropyLoss(),
                                    optimizer=optimizer,
                                    input_shape=(3,64,64), nb_classes=2,clip_values=clip_values)
attacker=FastGradientMethod(estimator=estimator,eps=0.05)
# 对抗样本生成器
testDataset = CELEBA(root='E:/Project/ModelAndDataset/data', train=False,train_ratio=0.99,transform=transforms.ToTensor(),label=attr_name)
ae_generator=Adversarial_Examples_Generator(
            targetmodel=target_model,
            task=attr_name,
            dataset=testDataset,
            method=attacker,
            targeted=False,
            batch_size=64,
            save_dir=None,
            device=device
            )
print("Generating AEs!")
# 生成对抗样本
raw_imgs,adv_imgs,raw_labels,adv_labels=ae_generator.generate()

rawDataset=TensorDataset(raw_imgs,raw_labels)
advDataset=TensorDataset(adv_imgs,adv_labels)
rawLoader=DataLoader(dataset=rawDataset,batch_size=batch_size,shuffle=False)
advLoader=DataLoader(dataset=advDataset,batch_size=batch_size,shuffle=False)

test_energy = []
test_labels = []
test_z = []
print("=============Caculating test_energy=======================")
for (normal_imgs,normal_labels),(ae_imgs,ae_labels) in zip(rawLoader,advLoader):
    #重构与条件重构正常样本
    normal_diff=cvae.caculate_difference(normal_imgs,normal_labels)
    with torch.no_grad():
        normal_diff=normal_diff.to(device)
    # 计算正常样本能量
    enc, dec, dagmm_z, gamma = dagmm(normal_diff)
    sample_energy, cov_diag = dagmm.compute_energy(dagmm_z, size_average=False)
    test_energy.append(sample_energy.data.cpu().numpy())
    test_z.append(dagmm_z.data.cpu().numpy())
    test_labels.append(torch.zeros_like(normal_labels).cpu().numpy())
    #重构与条件重构对抗样本
    adv_diff=cvae.caculate_difference(ae_imgs,ae_labels)
    with torch.no_grad():
        adv_diff=adv_diff.to(device)
    # 计算对抗样本能量
    enc, dec, dagmm_z, gamma = dagmm(adv_diff)
    sample_energy, cov_diag = dagmm.compute_energy(dagmm_z, size_average=False)
    test_energy.append(sample_energy.data.cpu().numpy())
    test_z.append(dagmm_z.data.cpu().numpy())
    test_labels.append(torch.ones_like(ae_labels).cpu().numpy())

test_energy = np.concatenate(test_energy,axis=0)
test_z = np.concatenate(test_z,axis=0)
test_labels = np.concatenate(test_labels,axis=0)
# 计算combined_energy
combined_energy = np.concatenate([train_energy, test_energy], axis=0)
combined_labels = np.concatenate([train_labels, test_labels], axis=0)

thresh = np.percentile(combined_energy, 100 - 20)
print("Threshold :", thresh)

pred = (test_energy > thresh).astype(int)
gt = test_labels.astype(int)

from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

accuracy = accuracy_score(gt,pred)
precision, recall, f_score, support = prf(gt, pred, average='binary')

print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
        









        
            





        
