import os
import torch.autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import Subset
from cVAE_GAN_cifar10 import Encoder_cifar10,Decoder_cifar10,Discriminator_cifar10,loss_function
# 创建文件夹
if not os.path.exists('./img_CVAE-GAN_Cifar10'):
    os.mkdir('./img_CVAE-GAN_Cifar10')
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
num_epoch = 20
z_dimension=80

img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10 = datasets.CIFAR10(
    root="E:/Project/ModelAndDataset/data", train=True, transform=img_transform, download=True
)
# 创建一个包含前6400张图像的子集
subset_indices = range(6400)  # 选择所需数量的图像
dataset = Subset(cifar10, subset_indices)

# data loader 数据载入
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True
)
# 初始化变分自编码器，分类器与判别器
encoder=Encoder_cifar10(device=device)
decoder=Decoder_cifar10(device=device)
discriminator=Discriminator_cifar10(device=device)

if not os.path.exists('./img_CVAE-GAN_Cifar10'):
        os.mkdir('./img_CVAE-GAN_Cifar10')
cudnn.benchmark=True

# 初始化优化器
optimizerVAE = optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0001)
# 初始化损失函数
criterion_BCE = nn.BCEWithLogitsLoss().to(device)
# 训练cVAE-GAN
for epoch in range(num_epoch):
      for i,(data,label) in enumerate(dataloader,0):
            data=data.to(device)
            label=label.to(device)
            batch_size=data.shape[0]
            # 训练判别器
            real_output=discriminator(data,label)
            real_output=real_output.squeeze()
            #print(real_output.shape)
            real_label=torch.ones(batch_size).to(device)
           # print(real_label.shape)
            fake_label=torch.zeros(batch_size).to(device)
            d_loss_real=criterion_BCE(real_output,real_label)
            # VAE重构图片
            z,mean,logstd = encoder(data)
            fake_data=decoder(z,label)
            fake_output=discriminator(fake_data,label)
            fake_output=fake_output.squeeze()
            d_loss_fake=criterion_BCE(fake_output,fake_label)
            d_loss=d_loss_real+d_loss_fake

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            
            #训练VAE
            recon_data =decoder(z,label)
            # 计算重构图片与潜在向量之间的损失 
            vae_loss_recon=loss_function(recon_x=recon_data,
                                         x=data,
                                         mean=mean,
                                         logstd=logstd,
                                         device=device
                                         )
            # 计算判别器损失
            output=discriminator(recon_data,label)
            output=output.squeeze()
            real_label = torch.ones(batch_size).to(device)
            vae_loss_d = criterion_BCE(output,real_label)

           
            # 反向传播更新VAE参数
            optimizerVAE.zero_grad()
            vae_loss = vae_loss_recon + vae_loss_d
            vae_loss.backward()
            optimizerVAE.step()
            

            #训练分类器
            print('[%d/%d][%d/%d] Loss_D: %.4f  Loss_G: %.4f'
                      % (epoch, num_epoch, i, len(dataloader),
                         d_loss.item(),vae_loss.item()))
               
            if epoch==0:
                real_images = make_grid(data.cpu(), nrow=8, normalize=True).detach()
                save_image(real_images, './img_CVAE-GAN_Cifar10/real_images.png') 
            if i == len(dataloader)-1:
                sample = torch.randn(data.shape[0],z_dimension).to(device)
                output = decoder(sample,label)
                fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                save_image(fake_images, './img_CVAE-GAN_Cifar10/fake_images-{}.png'.format(epoch + 1))
torch.save(encoder.state_dict(), 'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Encoder.pth')
torch.save(decoder.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Decoder.pth')
torch.save(discriminator.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Discriminator.pth')