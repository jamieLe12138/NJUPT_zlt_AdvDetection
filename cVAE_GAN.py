import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
class VAE_MNIST(nn.Module):
    def __init__(self,z_dimension,device):
        super(VAE_MNIST, self).__init__()
        # 定义编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.encoder_fc1=nn.Linear(32*7*7,z_dimension)
        self.encoder_fc2=nn.Linear(32*7*7,z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(z_dimension+10,32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.device=device

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(self.device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    def encoder(self,x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        return z,mean,logstd
    def decoder(self,z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(-1, 32, 7, 7)
        out3 = self.decoder_deconv(out3)
        return out3
class Discriminator_MNIST(nn.Module):
    def __init__(self,outputn=1):
        super(Discriminator_MNIST, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, outputn),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)
    
class VAE_cifar10(nn.Module):
     def __init__(self,z_dimension,device):
        super(VAE_cifar10, self).__init__()


def loss_function(recon_x,x,mean,logstd,device):
    MSECriterion = nn.MSELoss().to(device)
    MSE = MSECriterion(recon_x,x)
    logvar = 2 * logstd
    KLD = KLD = -0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - torch.exp(logvar))
    return MSE+KLD

