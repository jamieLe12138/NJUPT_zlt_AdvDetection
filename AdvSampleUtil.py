import art
from art.attacks.evasion import FastGradientMethod
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
class AdvSampleGenerator():
    def __init__(self,targetmodel,dataset,method,targeted,device):
        self.targetmodel=targetmodel
        self.dataset=dataset
        self.method=method
        self.targeted=targeted
        self.device=device
    def generate(self,num_samples):
        model=self.targetmodel
        device=self.device
        dataset=self.dataset
        attacker=self.method
        model.to(device)
        #初始化
        raw_imgs=torch.empty(0).to(device)
        adv_imgs=torch.empty(0).to(device)
        raw_labels = torch.empty(0).to(device)
        adv_labels = torch.empty(0).to(device)
        adv_probs = torch.empty(0).to(device)
        batch_size = 64  # 选择适当的批次大小
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, (data, label) in enumerate(loader):
            # 将数据和目标移到正确的设备
            data, label = data.to(device), label.to(device)
            data=data.detach().cpu().numpy()
            # 生成对抗样本
            adv_data = attacker.generate(data) 
            model.eval()
            with torch.no_grad():
                adv_data=torch.from_numpy(adv_data).to(device)
                adv_output=model(adv_data)
                raw_data=torch.from_numpy(data).to(device)
                raw_output=model(raw_data)
                # 从模型的输出中提取标签 
                _,raw_predicted = raw_output.max(1) #获取样本的原标签
                _, adv_predicted = adv_output.max(1)  # 获取每个样本的预测标签 
                print("raw_predicted:",raw_predicted)
                print("adv_predicted:",adv_predicted)
                #判断是否成功攻击
                comparison = torch.eq(raw_predicted, adv_predicted).to(device)
                print("comparison:",comparison)
                #添加成功攻击的样本
                raw_imgs=torch.cat([raw_imgs,raw_data[~comparison]],dim=0)
                adv_imgs=torch.cat([adv_imgs,adv_data[~comparison]],dim=0)
                raw_labels = torch.cat([raw_labels, raw_predicted[~comparison].to(device)], dim=0)
                adv_labels = torch.cat([adv_labels, adv_predicted[~comparison].to(device)], dim=0)
                adv_probs = torch.cat([adv_probs, torch.softmax(adv_output[~comparison], dim=1).to(device)], dim=0)
        #随机选取一定数量的对抗样本
        random_indices=random.sample(range(len(raw_imgs)),num_samples)
        advSamplePairs=[]
        for idx in random_indices:
            raw_img=raw_imgs[idx].squeeze().cpu().numpy()
            adv_img=adv_imgs[idx].squeeze().cpu().numpy()
            raw_label=raw_labels[idx].item()
            adv_label=adv_labels[idx].item()
            adv_prob=adv_probs[idx]
            newAdvSamplePair=AdvSamplePair(idx,raw_img,adv_img,raw_label,adv_label,adv_prob)
            advSamplePairs.append(newAdvSamplePair)
        return advSamplePairs




        

class AdvSamplePair():
    def __init__(self,id,raw_img,adv_img,raw_label,adv_label,adv_prob):
        self.id=id 
        self.raw_img = raw_img
        self.adv_img = adv_img
        self.raw_label = raw_label
        self.adv_label = adv_label
        self.adv_probs = adv_prob
    def save(self,raw_path,adv_path):
        raw_path=raw_path+"raw_"+str(id)+"_"+str(int(self.raw_label))+"_"+str(int(self.raw_label))+".png"
        adv_path=adv_path+"adv_"+str(id)+"_"+str(int(self.raw_label))+"_"+str(int(self.raw_label))+".png"
        plt.imsave(raw_path,self.raw_img,cmap='gray')
        plt.imsave(adv_path,self.adv_img,cmap='gray')

def EvaluateImage(img1,img2):
    mse=np.mean((img1 - img2) ** 2)
    L2_distance=np.linalg.norm(img1 - img2)
    ssim_value = ssim(img1, img2,data_range=1.0)
    return mse,L2_distance,ssim_value  

# 计算曼哈顿距离
def manhattan_distance(z1, z2):
    return torch.sum(torch.abs(z1 - z2))

# 计算欧氏距离
def euclidean_distance(z1, z2):
    return F.pairwise_distance(z1, z2)

# 计算余弦相似度
def cosine_similarity(z1, z2):
    return F.cosine_similarity(z1, z2, dim=0)

# 计算KL散度
def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q))


def DrawResult(raw_data,adv_data,title,xlabel,ylabel):
    plt.scatter(range(len(raw_data)), raw_data, color='blue', label='Raw', s=3)
    plt.scatter(range(len(adv_data)), adv_data, color='red', label='Adv', s=3)
    
    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 添加图例
    plt.legend()
    fig = plt.gcf()
    return fig

        



