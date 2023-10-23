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

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features,eps=1e-4, momentum=0.1, affine=True )  
        self.gamma_embed = nn.Linear(num_conditions, num_features)
        self.beta_embed = nn.Linear(num_conditions, num_features)

    def forward(self, x, condition):
        # Calculate batch statistics for normalization
        batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
        batch_var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        
        # Normalize the input using batch statistics
        x = (x - batch_mean) / torch.sqrt(batch_var + 1e-5)
        
        # Calculate scale and bias based on condition
        gamma = self.gamma_embed(condition).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(condition).view(-1, self.num_features, 1, 1)
        
        # Apply scale and bias
        x = gamma * x + beta
        
        return x
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
        self.encoder_fc=nn.Linear(80,z_dimension).to(self.device)
    def forward(self,x):
        z=self.encoder_conv(x)
        z=self.encoder_fc(z)
        return z

    
    #     self.encoder_fc1=nn.Linear(80,z_dimension).to(self.device)
    #     self.encoder_fc2=nn.Linear(80,z_dimension).to(self.device)
    # def noise_reparameterize(self,mean,logvar):
    #     eps = torch.randn(mean.shape).to(self.device)
    #     z = mean + eps * torch.exp(logvar)
    #     return z
    # def forward(self,x):
    #     out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
    #     mean = self.encoder_fc1(out1).to(self.device)
    #     logstd = self.encoder_fc2(out2).to(self.device)
    #     z = self.noise_reparameterize(mean, logstd)
    #     return z,mean,logstd

class Decoder_Conv_CBN(nn.Module):
    def __init__(self,gen_size,num_features,embed_size,chunksize):
        super(Decoder_Conv_CBN,self).__init__()

        self.gen_size=gen_size
        self.num_features=num_features
        self.embed_size=embed_size
        self.chunksize=chunksize

        self.conv=nn.Conv2d(num_features,num_features,3,1,'same')
        self.cbn=ConditionalBatchNorm2d(self.num_features,10)
        self.linear=nn.Linear(embed_size+chunksize,10)
        self.relu=nn.ReLU()
    
    def forward(self,x,y):
        x=self.conv(x)
        y=self.linear(y)
        x=self.cbn(x,y)
        x=self.relu(x)
        return x


#生成块
class DecoderBlock(nn.Module):
    def __init__(self,gen_size,num_features,embed_size,chunksize,ys_length):
        super(DecoderBlock,self).__init__()

        self.gen_size=gen_size
        self.num_features=num_features
        self.embed_size=embed_size
        self.chunksize=chunksize
        self.ys_length=ys_length

        self.conv=nn.Conv2d(num_features,num_features,1,1,'same')
        self.linear=nn.Linear(embed_size+chunksize,10)
        self.cbn=ConditionalBatchNorm2d(self.num_features,10)
        self.upsample1=nn.Upsample(size=(self.gen_size*2, self.gen_size*2), mode='bilinear', align_corners=True)
        self.upsample2=nn.Upsample(size=(self.gen_size*2, self.gen_size*2), mode='bilinear', align_corners=True)
        self.relu=nn.ReLU()

        #生成len(ys)个卷积CBN块
        self.conv_cbns=nn.ModuleList()
        for i in range(self.ys_length-1):  
            conv_cbn=Decoder_Conv_CBN(gen_size,
                                      num_features,
                                      embed_size,
                                      chunksize)
            self.conv_cbns.append(conv_cbn)
       

    def forward(self,z,ys):
        z1=self.upsample1(z)
        z1=self.conv(z1)

        y1=self.linear(ys[0])
        z2=self.cbn(z,y1)
        z2=self.relu(z2)
        z2=self.upsample2(z2)
        for i,conv_cbn in enumerate(self.conv_cbns):
            z2=conv_cbn(z2,ys[i+1])
        h=z1+z2
        return h            

class Decoder_cifar10(nn.Module):
    def __init__(self,z_dimension=80,chunksize=20,
                 num_features=384,gen_size=4,
                 embed_size=128,device="cpu"):
        super(Decoder_cifar10,self).__init__()

        self.z_dimension=z_dimension 
        self.chunksize=chunksize  
        self.num_features=num_features  
        self.gen_size=gen_size
        self.embed_size=embed_size
        self.device=device

        self.linear=nn.Linear(chunksize,384*gen_size*gen_size).to(self.device)
        self.embed=nn.Embedding(10,embed_size).to(self.device)
        
        #定义生成块
        self.gen_blocks=nn.ModuleList()
        for i in range((z_dimension//chunksize)-1):  
            gen_block = DecoderBlock(self.gen_size *2**(i),
                                     self.num_features,
                                     self.embed_size,
                                     self.chunksize,
                                     (z_dimension//chunksize)-1
                                     ).to(self.device)  # 使用不同的 gen_size
            self.gen_blocks.append(gen_block)
        self.BnReLu=nn.Sequential(
                    nn.BatchNorm2d(384),
                    nn.ReLU()).to(self.device)
        self.conv=nn.Conv2d(384,3,kernel_size=3,stride=1,padding='same').to(self.device)
        self.tanh=nn.Tanh().to(device)
    def forward(self,z,y):
        #将z切片
        zs = torch.split(z, self.chunksize, dim=1)
        z=zs[0].to(self.device)
        zs=zs[1:]
        y=self.embed(y)
        y=y.to(self.device)
        #拼接
        ys=[]
        for i in range(len(zs)):
            ys.append(torch.cat((y, zs[i].to(self.device)), dim=1))
        z=self.linear(z)
        z=z.view(-1, 384, 4, 4)
        for i,genblock in enumerate(self.gen_blocks):
            z=genblock(z,ys)
        z=self.BnReLu(z)
        z=self.conv(z)
        z=self.tanh(z)
        return z
    
class Dis_BN_Relu_Conv(nn.Module):
    def __init__(self,in_channels,num_features):
        super(Dis_BN_Relu_Conv,self).__init__()
        self.in_channels=in_channels
        self.num_features=num_features

        self.bn=nn.BatchNorm2d(num_features=self.in_channels)
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels,num_features,3,1,'same')                 
        
    def forward(self,x):
        x=self.bn(x)
        x=self.relu(x)
        x=self.conv(x)
        return x


class DisBlock(nn.Module):
    def __init__(self,in_channels,num_features,dis_size,downsample):
        super(DisBlock,self).__init__()
        self.in_channels=in_channels
        self.num_features=num_features
        self.dis_size=dis_size

        self.conv=nn.Conv2d(self.in_channels,self.num_features,1,1,'same')
        if downsample==True:
            self.downsample1=nn.AvgPool2d(kernel_size=2, stride=2)
            self.downsample2=nn.AvgPool2d(kernel_size=2,stride=2)
        else:
            self.downsample1=None
            self.downsample2=None
        self.bn_relu_conv=nn.Sequential(
            Dis_BN_Relu_Conv(self.in_channels,self.num_features),
            Dis_BN_Relu_Conv(self.num_features,self.num_features)
        )
    def forward(self,x):
        x1=self.conv(x)
        if self.downsample1!=None:
            x1=self.downsample1(x1)
        x2=self.bn_relu_conv(x)
        if self.downsample2!=None:
            x2=self.downsample2(x2)
        x3=x1+x2
        return x3




class Discriminator_cifar10(nn.Module):
    def __init__(self,num_features=192,dis_size=32,
                 embed_size=192,device="cpu"):
        super(Discriminator_cifar10,self).__init__()
        self.num_features=num_features
        self.dis_size=dis_size
        self.embed_size=embed_size
        self.device=device
        
        self.embed=nn.Embedding(10,self.embed_size).to(self.device)
        self.dis_block0=DisBlock(3,self.num_features,self.dis_size,True).to(self.device)
        self.dis_block1=DisBlock(self.num_features,self.num_features,self.dis_size//2,True).to(self.device)
        self.dis_block2=DisBlock(self.num_features,self.num_features,self.dis_size//4,False).to(self.device)
        self.dis_block3=DisBlock(self.num_features,self.num_features,self.dis_size//4,False).to(self.device)                              
                                    
        self.relu=nn.ReLU().to(self.device)
        self.linear1=nn.Sequential(nn.Linear(self.num_features,self.num_features),nn.ReLU()).to(self.device)
        self.linear2=nn.Linear(self.num_features,1).to(self.device)
        
    def forward(self,x,y):
        x=x.to(self.device)
        y=y.to(self.device)
        y=self.embed(y)
        x=self.dis_block0(x)
        x=self.dis_block1(x)
        x=self.dis_block2(x)
        x=self.dis_block3(x)
        x=self.relu(x)
        x=x.sum(dim=(2, 3))
        x=self.linear1(x)
        x=x+x*y
        x=self.linear2(x)
        return x


def loss_function(recon_x,x,mean,logstd,device):
    MSECriterion = nn.MSELoss().to(device)
    MSE = MSECriterion(recon_x,x)
    logvar = 2 * logstd
    KLD = -0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - torch.exp(logvar))
    return MSE+KLD

def myloss_function(recon_x,x,device):
    # MSECriterion = nn.MSELoss().to(device)
    # MSE = MSECriterion(recon_x,x)
    l2=F.pairwise_distance(recon_x,x)
    return l2

        




    























# class Decoder_cifar10(nn.Module):
#     def __init__(self,z_dimension=80,chunksize=20,
#                  num_features=384,gen_size=4,
#                  embed_size=128):
#         super(Decoder_cifar10,self).__init__()

#         self.z_dimension=z_dimension 
#         self.chunksize=chunksize  
#         self.num_features=num_features  
#         self.gen_size=gen_size
#         self.embed_size=embed_size
        

#         self.linear=nn.Linear(chunksize,384*gen_size*gen_size)
#         self.embed=nn.Embedding(10,embed_size)
        
#         #定义生成块
#         self.gen_blocks=nn.ModuleList()
#         for i in range((z_dimension//chunksize)-1):  
#             gen_block = DecoderBlock(self.gen_size *2**(i),
#                                      self.num_features,
#                                      self.embed_size,
#                                      self.chunksize,
#                                      (z_dimension//chunksize)-1
#                                      )  # 使用不同的 gen_size
#             self.gen_blocks.append(gen_block)
#         self.BnReLu=nn.Sequential(
#                     nn.BatchNorm2d(384),
#                     nn.ReLU())
#         self.conv=nn.Conv2d(384,3,kernel_size=3,stride=1,padding='same')
#         self.tanh=nn.Tanh()
#     def forward(self,z,y):
#         #将z切片
#         zs = torch.split(z, self.chunksize, dim=1)
#         z=zs[0]
#         zs=zs[1:]
#         y=self.embed(y)
#         y=y
#         #拼接
#         ys=[]
#         for i in range(len(zs)):
#             ys.append(torch.cat((y, zs[i]), dim=1))
#         z=self.linear(z)
#         z=z.view(-1, 384, 4, 4)
#         for i,genblock in enumerate(self.gen_blocks):
#             z=genblock(z,ys)
#         z=self.BnReLu(z)
#         z=self.conv(z)
#         z=self.tanh(z)
#         return z

        




# linear_layer = nn.Linear(20, 384*4*4)
# mapped_condition = linear_layer(condition_vector)
# mapped_condition = mapped_condition.view(-1, 384, 4, 4)

# # 2. 上采样为384x8x8
# upsample_layer = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=True)
# upsampled_condition = upsample_layer(mapped_condition)

# # 3. 使用1x1卷积进行特征处理
# conv1x1 = nn.Conv2d(384, 384, kernel_size=1)
# processed_condition = conv1x1(upsampled_condition)
