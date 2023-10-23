from Model_Options import *
import torch
import torch.nn as nn
import torch.nn.functional as F

#编码块
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlock, self).__init__()
        self.block =nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
        ) 
    def forward(self, x):
        return self.block(x)
#编码器
class Encoder_cifar10(nn.Module):
    def __init__(self,z_dimension=80,device="cpu"):
        super(Encoder_cifar10, self).__init__()
        self.device=device
        # 定义编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.ReLU(),
            EncoderBlock(64,128,4,2,1),
            EncoderBlock(128,256,4,2,1),
            nn.Conv2d(256,80,4,2,0),
            nn.Flatten()
        ).to(self.device)
        self.encoder_fc1=nn.Linear(80,z_dimension).to(self.device)
        self.encoder_fc2=nn.Linear(80,z_dimension).to(self.device)
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(self.device)
        z = mean + eps * torch.exp(logvar)
        return z
    def forward(self,x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        return z,mean,logstd
    
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, z_dims_after_concat):
        super(GenBlock, self).__init__()
        self.conditional_bn = conditional_bn
        if self.conditional_bn:
            self.bn1 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=in_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
            self.bn2 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=out_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
        else:
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if g_spectral_norm:
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, label):
        x0 = x
        if self.conditional_bn:
            x = self.bn1(x, label)
        else:
            x = self.bn1(x)

        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.conv2d1(x)
        if self.conditional_bn:
            x = self.bn2(x, label)
        else:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class Generator_cifar10(nn.Module):
    """Generator."""
    def __init__(self, z_dim=80, shared_dim=128, g_conv_dim=96, attention_after_nth_gen_block=2,activation_fn="ReLU",
                num_classes=10,device="cpu"):
        super(Generator_cifar10, self).__init__()
        self.z_dim = z_dim #80
        self.shared_dim = shared_dim #128
        self.num_classes = num_classes
        conditional_bn = True 
        self.in_dim =  g_conv_dim*4
        self.out_dim = g_conv_dim*4
        self.bottom = 4
        self.n_blocks = 3 
        self.chunk_size = z_dim//(self.n_blocks+1)# 80//3+1
        self.z_dims_after_concat = self.chunk_size + self.shared_dim #20+128
        # assert self.z_dim % (self.n_blocks+1) == 0, "z_dim should be divided by the number of blocks "
        self.device=device
        self.linear0 = snlinear(in_features=self.chunk_size, out_features=self.in_dim*self.bottom*self.bottom).to(self.device)
        self.shared = embedding(self.num_classes, self.shared_dim).to(self.device)
        

        self.blocks =nn.ModuleList().to(self.device)
        for index in range(self.n_blocks):
            self.blocks.append(GenBlock(in_channels=self.in_dim,
                                      out_channels=self.out_dim,
                                      g_spectral_norm=True,
                                      activation_fn=activation_fn,
                                      conditional_bn=conditional_bn,
                                      z_dims_after_concat=self.z_dims_after_concat)).to(self.device) 
            if index+1 == attention_after_nth_gen_block :
                self.blocks.append(Self_Attn(self.out_dim, True)).to(self.device)

        self.bn4 = batchnorm_2d(in_features=self.out_dim).to(self.device)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True).to(self.device)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True).to(self.device)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True).to(self.device)
        elif activation_fn == "GELU":
            self.activation = nn.GELU().to(self.device)
        else:
            raise NotImplementedError

        self.conv2d5 = snconv2d(in_channels=self.out_dim, out_channels=3, kernel_size=3, stride=1, padding=1).to(self.device)
        self.tanh = nn.Tanh().to(self.device)
        # Weight init
       
        init_weights(self.modules, 'ortho')


    def forward(self, z, label):
        zs = torch.split(z, self.chunk_size, 1)
        z = zs[0].to(self.device)
        label=label.to(self.device)
        shared_label = self.shared(label)
       
        labels = [torch.cat([shared_label, item.to(self.device)], 1) for item in zs[1:]]

        act = self.linear0(z)
        act = act.view(-1, self.in_dim, self.bottom, self.bottom)
        counter = 0
        for index, block in enumerate(self.blocks):
            if isinstance(block, Self_Attn):
                act = block(act)
            else:
                act = block(act, labels[counter])
                counter +=1

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn):
        super(DiscOptBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm

        if d_spectral_norm:
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            self.bn0 = batchnorm_2d(in_features=in_channels)
            self.bn1 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.average_pooling = nn.AvgPool2d(2)


    def forward(self, x):
        x0 = x
        x = self.conv2d1(x)
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        if self.d_spectral_norm is False:
            x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn, downsample=True):
        super(DiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm
        self.downsample = downsample

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if d_spectral_norm:
            if self.ch_mismatch or downsample:
                self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            if self.ch_mismatch or downsample:
                self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            if self.ch_mismatch or downsample:
                self.bn0 = batchnorm_2d(in_features=in_channels)
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)


    def forward(self, x):
        x0 = x

        if self.d_spectral_norm is False:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)
        if self.d_spectral_norm is False:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            if self.d_spectral_norm is False:
                x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)

        out = x + x0
        return out



class Discriminator_cifar10(nn.Module):
    """Discriminator."""
    def __init__(self, d_conv_dim=96, d_spectral_norm=True, attention=True, attention_after_nth_dis_block=1, activation_fn="ReLU",
                  num_classes=10,device="cpu"):
        super(Discriminator_cifar10, self).__init__()

    

        self.in_dims  =  [3] + [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2]
        self.out_dims = [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2, d_conv_dim*2]
        self.down = [True, True, False, False]
        self.device=device
        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[DiscOptBlock(in_channels=self.in_dims[index],
                                              out_channels=self.out_dims[index],
                                              d_spectral_norm=d_spectral_norm,
                                              activation_fn=activation_fn).to(self.device)]]
            else:
                self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
                                           out_channels=self.out_dims[index],
                                           d_spectral_norm=d_spectral_norm,
                                           activation_fn=activation_fn,
                                           downsample=self.down[index]).to(self.device)]]

            if index+1 == attention_after_nth_dis_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], d_spectral_norm).to(self.device)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks]).to(self.device)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True).to(self.device)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True).to(self.device)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True).to(self.device)
        elif activation_fn == "GELU":
            self.activation = nn.GELU().to(self.device)
        else:
            raise NotImplementedError

        self.linear1 = snlinear(in_features=self.out_dims[-1], out_features=1).to(self.device)
        self.embedding = sn_embedding(num_classes, self.out_dims[-1]).to(self.device)
            

        
        init_weights(self.modules, 'ortho')


    def forward(self, x, label):
            h = x.to(self.device)
            label=label.to(self.device)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.activation(h)
            h = torch.sum(h, dim=[2,3])
        
            authen_output = torch.squeeze(self.linear1(h))
            proj = torch.sum(torch.mul(self.embedding(label), h), 1)
            return authen_output + proj

    