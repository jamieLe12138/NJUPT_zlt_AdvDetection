from advanced_models import CVAE as CVAE2
from models import CVAE as CVAE

from dataload import CELEBA
from torchvision import transforms
import torch
import numpy as np
import os
from os.path import join
root='E:/Project/ModelAndDataset/data'
attr_name='Smiling'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_num=150
transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
testDataset = CELEBA(root=root, train=False, train_ratio=0.7,transform=transforms.ToTensor(),label=attr_name)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=64, shuffle=False)

cvae1=CVAE(nz=100,imSize=64,fSize=64,device=device)
cvae2=CVAE2(nz=100,imSize=64,fSize=64,self_attn=True,CBN=True,device=device)
cvae1.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae2.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae1.to(device)
cvae2.to(device)

gen_imgs1=torch.empty(0).cpu()
gen_labels1=torch.empty(0).cpu()
gen_imgs2=torch.empty(0).cpu()
gen_labels2=torch.empty(0).cpu()

for (i,(x,y)) in enumerate(testLoader):
    if i<batch_num:
        x=x.to(device)
        y=y.to(device)
        fake_x1,outMu,outLogVar,outY = cvae1(x)
        fake_x2,outMu,outLogVar,outY = cvae2(x)

        gen_imgs1=torch.cat([gen_imgs1,fake_x1.detach().cpu()],dim=0)
        gen_labels1=torch.cat([gen_labels1,y.detach().cpu()],dim=0)

        gen_imgs2=torch.cat([gen_imgs2,fake_x2.detach().cpu()],dim=0)
        gen_labels2=torch.cat([gen_labels2,y.detach().cpu()],dim=0)
        print("generate batch{}".format(i))
    else:
        break

np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN',attr_name+"_"+"image"), gen_imgs1.detach().numpy())
np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN',attr_name+"_"+"label"), gen_labels1.detach().numpy())

np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN_Advanced',attr_name+"_"+"image"), gen_imgs2.detach().numpy())
np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN_Advanced',attr_name+"_"+"label"), gen_labels2.detach().numpy())

np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/real',attr_name+"_"+"image"),x.detach().cpu().numpy())
np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/real',attr_name+"_"+"label"),y.detach().cpu().numpy())

