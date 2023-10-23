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
from CVAE_BigGAN import Encoder_cifar10,Generator_cifar10,Discriminator_cifar10
import losses
# 创建文件夹
if not os.path.exists('./img_CVAE-GAN_Cifar10'):
    os.mkdir('./img_CVAE-GAN_Cifar10')
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size =128
num_epoch = 20
z_dimension=80

img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10 = datasets.CIFAR10(
    root="E:/Project/ModelAndDataset/data", train=True, transform=img_transform, download=True
)
# 创建一个包含前6400张图像的子集
subset_indices = range(12800)  # 选择所需数量的图像
dataset = Subset(cifar10, subset_indices)

# data loader 数据载入
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True
)
# 初始化变分自编码器，分类器与判别器
encoder=Encoder_cifar10(device=device)
decoder=Generator_cifar10(device=device)
discriminator=Discriminator_cifar10(device=device)

" ========================加载预训练模型  ==============================      "
# encoder.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Encoder.pth'))
# decoder.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Decoder.pth'))
# discriminator.load_state_dict(torch.load('E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Discriminator.pth'))

if not os.path.exists('./img_CVAE-GAN_Cifar10'):
        os.mkdir('./img_CVAE-GAN_Cifar10')
cudnn.benchmark=True

# 初始化优化器
optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.00002)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=0.00002)
optimizerD = optim.Adam(discriminator.parameters(), lr=0.00002)
# 初始化损失函数
d_loss = losses.loss_hinge_dis
g_loss = losses.loss_hinge_gen

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
            d_loss_real=d_loss(real_output,real_label)
            
            fake_label=torch.zeros(batch_size).to(device)
            # VAE重构图片
            z,_,_ = encoder(data)
            # z=torch.randn(batch_size,80).to(device)
            fake_data=decoder(z,label)
            fake_output=discriminator(fake_data,label)
            fake_output=fake_output.squeeze()
            d_loss_fake=d_loss(fake_output,fake_label)
            d_losses=d_loss_real+d_loss_fake
            optimizerD.zero_grad()
            d_losses.backward()
            optimizerD.step()
            #print("Discriminator_Loss",d_loss)
            
            #训练VAE
            z2,mean,logstd= encoder(data)
            recon_data =decoder(z2,label)
            # 计算重构图片与潜在向量之间的损失 
            vae_loss_recon=losses.loss_recon(recon_x=recon_data,
                                         x=data,
                                         mean=mean,
                                         logstd=logstd,
                                         device=device
                                         )
            #vae_loss_recon=g_loss()

            # 计算判别器损失
            output=discriminator(recon_data,label)
            output=output.squeeze()
            real_label = torch.ones(batch_size).to(device)
            vae_loss_d = d_loss(output,real_label)
            vae_loss=(vae_loss_recon+vae_loss_d)
            #print("VAE_Loss:",vae_loss)
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            # 反向传播更新VAE参数
            vae_loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()
            
            
           
            #训练分类器
            if i%50==0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_VAE: %.4f'
                      % (epoch, num_epoch, i, len(dataloader),
                         d_losses.item(),vae_loss.item()))
               
            if epoch==0:
                real_images = make_grid(data.cpu(), nrow=8, normalize=True).detach()
                save_image(real_images, './img_CVAE-GAN_Cifar10/real_images.png') 
            if i == len(dataloader)-1:
                z3,_,_ = encoder(data)
                output = decoder(z3,label)
                fake_images = make_grid(output.cpu(), nrow=16, normalize=True).detach()
                save_image(fake_images, './img_CVAE-GAN_Cifar10/fake_images-{}.png'.format(epoch + 1))
                real_images = make_grid(data.cpu(), nrow=16, normalize=True).detach()
                save_image(real_images, './img_CVAE-GAN_Cifar10/real_images-{}.png'.format(epoch + 1))
torch.save(encoder.state_dict(), 'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Encoder.pth')
torch.save(decoder.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Decoder.pth')
torch.save(discriminator.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Cifar10-Discriminator.pth')