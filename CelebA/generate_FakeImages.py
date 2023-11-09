from advanced_models import CVAE as CVAE2
from models import CVAE as CVAE

from dataload import CELEBA
from torchvision import transforms
import torch
root='E:/Project/ModelAndDataset/data'
attr_name='Smiling'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
testDataset = CELEBA(root=root, train=False, train_ratio=0.7,transform=transforms.ToTensor(),label=attr_name)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=64, shuffle=False)

cvae1=CVAE(nz=100,imSize=64,device=device)
cvae2=CVAE2(nz=100,imSize=64,self_attn=True,CBN=True,device=device)
cvae1.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae2.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae1.to(device)
cvae2.to(device)

for x,y in testLoader:
    fake_x1, outMu, outLogVar, outY = cvae1(x)