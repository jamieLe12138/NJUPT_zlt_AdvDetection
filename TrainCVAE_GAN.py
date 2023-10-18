import os
import torch.autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
from torchvision.utils import save_image
from cVAE_GAN import VAE_MNIST,Discriminator_MNIST,loss_function 
from torchvision.utils import make_grid
# 创建文件夹
if not os.path.exists('./img_CVAE-GAN_MNIST'):
    os.mkdir('./img_CVAE-GAN_MNIST')
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_epoch = 50
z_dimension=100
img_transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist = datasets.MNIST(
    root='./data', train=True, transform=img_transform, download=True
)
# data loader 数据载入
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True
)
# 初始化变分自编码器，分类器与判别器
vae=VAE_MNIST(z_dimension,device).to(device)
classfier=Discriminator_MNIST(10).to(device)
discriminator=Discriminator_MNIST(1).to(device)
batchSize = 128
if not os.path.exists('./img_CVAE-GAN_MNIST'):
        os.mkdir('./img_CVAE-GAN_MNIST')
cudnn.benchmark=True

# 初始化优化器
optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)
optimizerC = optim.Adam(classfier.parameters(), lr=0.0001)
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0001)
# 初始化损失函数
criterion_BCE = nn.BCELoss().to(device)
criterion_CE =nn.CrossEntropyLoss().to(device)

# 训练cVAE-GAN
for epoch in range(num_epoch):
      for i,(data,label) in enumerate(dataloader,0):
            data=data.to(device)
            label_onehot = torch.zeros((data.shape[0], 10)).to(device)
            label_onehot[torch.arange(data.shape[0]), label] = 1
            batch_size=data.shape[0]
            # 训练分类器
            real_output =classfier(data)
            real_label=label_onehot.to(device)
            c_loss=criterion_CE(real_output,real_label)
            classfier.zero_grad()
            c_loss.backward()
            optimizerC.step()
            # 训练判别器
            real_output=discriminator(data)
            real_label=torch.ones(batch_size).to(device)
            fake_label=torch.zeros(batch_size).to(device)
            d_loss_real=criterion_BCE(real_output,real_label)
            # 生成器生成图片
            z=torch.randn(batch_size,z_dimension+10).to(device)
            fake_data=vae.decoder(z)
            fake_output=discriminator(fake_data)
            d_loss_fake=criterion_BCE(fake_output,fake_label)
            d_loss=d_loss_real+d_loss_fake

            discriminator.zero_grad()
            d_loss.backward()
            optimizerD.step()

            z,mean,logstd = vae.encoder(data)
            # 拼接编码器提取的潜在向量z与独热标签
            z=torch.cat([z,label_onehot],1)
            recon_data =vae.decoder(z)
            # 计算重构图片与潜在向量之间的损失 
            vae_loss_recon=loss_function(recon_x=recon_data,
                                         x=data,
                                         mean=mean,
                                         logstd=logstd,
                                         device=device
                                         )
            # 计算判别器损失
            output=discriminator(recon_data)
            real_label = torch.ones(batch_size).to(device)
            vae_loss_d = criterion_BCE(output,real_label)

            # 计算分类器损失
            real_label=label_onehot
            output=classfier(recon_data)
            vae_loss_c=criterion_BCE(output,real_label)

            # 反向传播更新VAE参数
            vae.zero_grad()
            # vae_loss = vae_loss_recon + vae_loss_d
            vae_loss = vae_loss_recon + vae_loss_c + vae_loss_d
            vae_loss.backward()
            optimizerVAE.step()

            #训练分类器
            if i%100==0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f Loss_G: %.4f'
                      % (epoch, num_epoch, i, len(dataloader),
                         d_loss.item(),c_loss.item(),vae_loss.item()))

            # #不训练分类器
            # if i%100==0:
            #     print('[%d/%d][%d/%d] Loss_D: %.4f  Loss_G: %.4f'
            #           % (epoch, num_epoch, i, len(dataloader),
            #              d_loss.item(),vae_loss.item()))

            if epoch==0:
                real_images = make_grid(data.cpu(), nrow=8, normalize=True).detach()
                save_image(real_images, './img_CVAE-GAN_MNIST/real_images.png') 
            if i == len(dataloader)-1:
                sample = torch.randn(data.shape[0],z_dimension ).to(device)
                print(label)
                sample = torch.cat([sample,real_label],1)
                output = vae.decoder(sample)
                fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                save_image(fake_images, './img_CVAE-GAN_MNIST/fake_images-{}.png'.format(epoch + 1))
torch.save(vae.state_dict(), 'E:/Project/ModelAndDataset/model/CVAE-GAN-VAE.pth')
torch.save(discriminator.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Discriminator.pth')
torch.save(classfier.state_dict(),'E:/Project/ModelAndDataset/model/CVAE-GAN-Classifier.pth')










