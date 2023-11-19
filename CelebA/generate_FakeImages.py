import sys
sys.path.append("E:/Project/ZLTProgram/CelebA")


from Model.CVAE_GAN import CVAE as CVAE
from Model.CVAE_GAN_Advanced import CVAE as CVAE2
from Model.CVAE_GAN_AdvancedV2 import CVAE as CVAE3
from Utils.dataload import CELEBA


from torchvision import transforms
import torch
import numpy as np
import os
from os.path import join
root='E:/Project/ModelAndDataset/data'
attr_name='Smiling'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_num=150
transform = transforms.Compose([transforms.ToTensor()])
testDataset = CELEBA(root=root, train=False, train_ratio=0.7,transform=transforms.ToTensor(),label=attr_name)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=64, shuffle=False)

# cvae1=CVAE(nz=100,imSize=64,fSize=64,device=device)
cvae2=CVAE2(nz=100,imSize=64,fSize=64,self_attn=True,CBN=True,device=device)
cvae3 = CVAE3(nz=100,
			 imSize=64,
			 enc_self_attn=True,
			 dec_self_attn=True,
			 g_spectral_norm=False,
			 CBN=True,
			 fSize=64,
			 device=device)

# cvae1.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae2.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae3.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
# cvae1.to(device)
cvae2.to(device)
cvae3.to(device)

gen_imgs3=torch.empty(0).cpu()
gen_labels3=torch.empty(0).cpu()
gen_imgs2=torch.empty(0).cpu()
gen_labels2=torch.empty(0).cpu()

for (i,(x,y)) in enumerate(testLoader):
    if i<batch_num:
        x=x.to(device)
        y=y.to(device)
        fake_x3,outMu,outLogVar,outY = cvae3(x)
        fake_x2,outMu,outLogVar,outY = cvae2(x)

        gen_imgs3=torch.cat([gen_imgs3,fake_x3.detach().cpu()],dim=0)
        gen_labels3=torch.cat([gen_labels3,y.detach().cpu()],dim=0)

        gen_imgs2=torch.cat([gen_imgs2,fake_x2.detach().cpu()],dim=0)
        gen_labels2=torch.cat([gen_labels2,y.detach().cpu()],dim=0)
        print("generate batch{}".format(i))
    else:
        break

np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN_AdvancedV2',attr_name+"_"+"image"), gen_imgs3.detach().numpy())
np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN_AdvancedV2',attr_name+"_"+"label"), gen_labels3.detach().numpy())

np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN_Advanced',attr_name+"_"+"image"), gen_imgs2.detach().numpy())
np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/cVAEGAN_Advanced',attr_name+"_"+"label"), gen_labels2.detach().numpy())

np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/real',attr_name+"_"+"image"),x.detach().cpu().numpy())
np.save(join('E:/Project/ModelAndDataset/data/celebA/Gen/real',attr_name+"_"+"label"),y.detach().cpu().numpy())

