import sys
sys.path.append("E:/Project/ZLTProgram/CelebA")
from Model.CVAE_GAN import CVAE 
from Model.CVAE_GAN_Advanced import CVAE as Advanced_CVAE

from Utils.dataload import CELEBA,CELEBA_Attack
from torchvision import transforms
import torch
import numpy as np
import os
from os.path import join

class CVAEGAN_Images_Generator():
    def __init__(self,
                 dataset,
                 gen_model_name,
                 gen_model_path="E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN",
                 attr_name='Smiling',
                 batch_num=150,
                 data_save_path='E:/Project/ModelAndDataset/data/celebA/Gen',
                 device='cpu'):
        self.gen_model_name=gen_model_name
        self.gen_model_path=gen_model_path
        self.attr_name=attr_name
        self.batch_num=batch_num
        self.data_save_path=data_save_path
        self.device=device

        self.dataLoader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        if self.gen_model_name=='model':
            print("Use model")
            self.gen_model=CVAE(nz=100,imSize=64,fSize=64,device=self.device)
            self.gen_model.load_params(join(self.gen_model_path,self.attr_name))
            self.gen_model.to(device=self.device)
        elif self.gen_model_name=='advanced_model':
            print("Use advanced_model")
            self.gen_model=Advanced_CVAE(nz=100,imSize=64,fSize=64,self_attn=True,CBN=True,device=self.device)
            self.gen_model.load_params(join(self.gen_model_path,self.attr_name))
            self.gen_model.to(device=self.device)
        
    def reconstruct(self):
        real_imgs_list=[]
        gen_imgs_list=[]
        label_list=[]
        for (i,(x,y)) in enumerate(self.dataLoader):
            if i<self.batch_num:
                x=x.to(self.device)
                y=y.to(self.device)
                # 添加生成数据与真实数据
                fake_x,_,_,_ = self.gen_model(x)
                real_imgs_list.append(x.cpu())
                gen_imgs_list.append(fake_x.detach().cpu())
                label_list.append(y.cpu())
                if i%10==0:
                    print("Reconstruct batch{}/{}".format(i,self.batch_num))            
            else:
                break
        real_imgs=torch.cat(real_imgs_list,dim=0)
        gen_imgs = torch.cat(gen_imgs_list, dim=0)
        labels=torch.cat(label_list,dim=0)

        if self.gen_model_name=="model":
            dirname='cVAEGAN'
        else:
            dirname='cVAEGAN_Advanced'
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(join(self.data_save_path,dirname)):
            os.makedirs(join(self.data_save_path,dirname))
        if not os.path.exists(join(self.data_save_path,"join")):
            os.makedirs(join(self.data_save_path,"join"))
            
        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"image"), gen_imgs.detach().numpy())
        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"label"),labels.detach().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"image"),real_imgs.detach().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"label"),labels.detach().numpy())

    def generate_with_label(self,reverse_label=False):
        real_imgs_list=[]
        gen_imgs_list=[]
        label_list=[]
        for (i,(x,y)) in enumerate(self.dataLoader):
            if i<self.batch_num:
                x=x.to(self.device)
                y=y.to(self.device)
                mu, log_var, _= self.gen_model.encode(x)
                z = self.gen_model.re_param(mu, log_var)
                #print("y",y)
                #反转标签
                if reverse_label:
                    reverse_y=y^1
                    #print("reverse_y",reverse_y)
                    one_hot_y = torch.eye(2)[torch.LongTensor(reverse_y.data.cpu().numpy())].type_as(z)
                else:
                    one_hot_y = torch.eye(2)[torch.LongTensor(y.data.cpu().numpy())].type_as(z)
                fake_x = self.gen_model.decode(one_hot_y, z)
                # 添加生成数据与真实数据
                real_imgs_list.append(x.cpu())
                gen_imgs_list.append(fake_x.detach().cpu())
                label_list.append(y.cpu())
                if i%10==0:
                    print("Generate with labels batch{}/{}".format(i,self.batch_num))            
            else:
                break    
        real_imgs=torch.cat(real_imgs_list,dim=0)
        gen_imgs = torch.cat(gen_imgs_list, dim=0)
        labels=torch.cat(label_list,dim=0)

        if self.gen_model_name=='model':
            dirname='cVAEGAN'
        else:
            dirname='cVAEGAN_Advanced'
        if reverse_label:
            dirname=dirname+'_ReverseLabel'
        else:
            pass
        print(dirname)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(join(self.data_save_path,dirname)):
            os.makedirs(join(self.data_save_path,dirname))
        if not os.path.exists(join(self.data_save_path,"real")):
            os.makedirs(join(self.data_save_path,"real"))

        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"image"), gen_imgs.detach().numpy())
        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"label"),labels.detach().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"image"),real_imgs.detach().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"label"),labels.detach().numpy())

        
        



