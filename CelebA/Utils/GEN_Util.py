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
                 gen_model_path="E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN",
                 attr_name='Smiling',
                 batch_num=150,
                 gen_model="advanced_model",
                 data_save_path='E:/Project/ModelAndDataset/data/celebA/Gen',
                 device='cpu'):
        
        self.gen_model_path=gen_model_path
        self.attr_name=attr_name
        self.batch_num=batch_num
        self.gen_model=gen_model
        self.data_save_path=data_save_path
        self.device=device

        self.dataLoader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        if self.gen_model=="model":
            self.gen_model=CVAE(nz=100,imSize=64,fSize=64,device=self.device)
            self.gen_model.load_params(join(self.gen_model_path,self.attr_name))
            self.gen_model.to(device=self.device)
        elif self.gen_model=="advanced_model":
            self.gen_model=Advanced_CVAE(nz=100,imSize=64,fSize=64,self_attn=True,CBN=True,device=self.device)
            self.gen_model.load_params(join(self.gen_model_path,self.attr_name))
            self.gen_model.to(device=self.device)
        
    def reconstruct(self):
        gen_imgs=torch.empty(0).cpu()
        gen_labels=torch.empty(0).cpu()
        for (i,(x,y)) in enumerate(self.dataLoader):
            if i<self.batch_num:
                x=x.to(self.device)
                y=y.to(self.device)
                fake_x,_,_,_ = self.gen_model(x)
                gen_imgs=torch.cat([gen_imgs,fake_x.detach().cpu()],dim=0)
                gen_labels=torch.cat([gen_labels,y.detach().cpu()],dim=0)
                if i%10==0:
                    print("Reconstruct batch{}/{}".format(i,self.batch_num))            
            else:
                break
        if self.gen_model=="model":
            dirname='cVAEGAN'
        else:
            dirname='cVAEGAN_Advanced'
        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"image"), gen_imgs.detach().numpy())
        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"label"), gen_labels.detach().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"image"),x.detach().cpu().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"label"),y.detach().cpu().numpy())

    def generate_with_label(self):
        gen_imgs=torch.empty(0).cpu()
        gen_labels=torch.empty(0).cpu()
        for (i,(x,y)) in enumerate(self.dataLoader):
            if i<self.batch_num:
                x=x.to(self.device)
                y=y.to(self.device)
                mu, log_var, _= self.gen_model.encode(x)
                z = self.gen_model.re_param(mu, log_var)
                y = torch.eye(2)[torch.LongTensor(y.data.cpu().numpy())].type_as(z)
                fake_x = self.gen_model.decode(y, z)
                gen_imgs=torch.cat([gen_imgs,fake_x.detach().cpu()],dim=0)
                gen_labels=torch.cat([gen_labels,y.detach().cpu()],dim=0)
                if i%10==0:
                    print("Generate with labels batch{}/{}".format(i,self.batch_num))            
            else:
                break
        if self.gen_model=="model":
            dirname='cVAEGAN'
        else:
            dirname='cVAEGAN_Advanced'
        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"image"), gen_imgs.detach().numpy())
        np.save(join(self.data_save_path,dirname,self.attr_name+"_"+"label"), gen_labels.detach().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"image"),x.detach().cpu().numpy())
        np.save(join(self.data_save_path,"real",self.attr_name+"_"+"label"),y.detach().cpu().numpy())

        
        



