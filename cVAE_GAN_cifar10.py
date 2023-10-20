import torch
import torch.nn as nn
import torch.nn.functional as F



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlock, self).__init__()
        self.block =nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
        ) 
    def forward(self, x):
        return self.block(x)

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)  # Disable affine transformation
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

class Encoder_cifar10(nn.Module):
    def __init__(self,z_dimension=80,device="cpu"):
        super(Encoder_cifar10, self).__init__()
        # 定义编码器
        self.encoder_conv = nn.Sequential(
            EncoderBlock(3,64,3,1,'same'),
            EncoderBlock(64,64,3,1,'same'),
            EncoderBlock(64,128,3,1,'same'),
            EncoderBlock(128,128,3,1,'same'),
            EncoderBlock(128,256,3,1,'same'),
            EncoderBlock(256,256,3,1,'same'),
            EncoderBlock(256,512,3,1,'same'),
            EncoderBlock(512,512,3,1,'same'),
            nn.Flatten(),
            nn.Linear(512*32*32,128),
            nn.LeakyReLU(0.2)
        )
        self.encoder_fc1=nn.Linear(128,z_dimension)
        self.encoder_fc2=nn.Linear(128,z_dimension)
        self.device=device
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(self.device)
        z = mean + eps * torch.exp(logvar)
        return z
    def forward(self,x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1).to(self.device)
        logstd = self.encoder_fc2(out2).to(self.device)
        z = self.noise_reparameterize(mean, logstd)
        return z,mean,logstd

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
    
    def forward(self,x,y):
        x=self.conv(x)
        x=self.linear(y)
        x=nn.ReLU(self.cbn(x,y))
        return x



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
        self.upsample=nn.Upsample(size=(self.gen_size*2, self.gen_size*2), mode='bilinear', align_corners=True)
        

        #生成len(ys)个卷积CBN块
        self.conv_cbns=nn.ModuleList()
        for i in range(self.ys_length):  
            conv_cbn=Decoder_Conv_CBN(gen_size,
                                      num_features,
                                      embed_size,
                                      chunksize)
            self.conv_cbns.append(conv_cbn)
       

    def forward(self,z,ys):
        z1=self.upsample1(z)
        z1=self.conv(z1)

        y1=self.linear(ys[0])
        z2=nn.ReLU(self.cbn1(z,y1))
        z2=self.upsample(z2)
        for i,conv_cbn in enumerate(self.conv_cbns):
            z2=conv_cbn(z2,ys[i+1])
        h=z1+z2
        return h
        

class Decoder_cifar10(nn.Module):
    def __init__(self,z_dimension=80,chunksize=20,num_features=384,gen_size=4,embed_size=128):
        super(Decoder_cifar10,self).__init__()

        self.z_dimension=z_dimension 
        self.chunksize=chunksize  
        self.num_features=num_features  
        self.gen_size=gen_size
        self.embed_size=embed_size
        self.linear=nn.Linear(chunksize,384*gen_size*gen_size)
        self.embed=nn.Embedding(10,embed_size)
        #定义生成块
        self.gen_blocks=nn.ModuleList()
        for i in range((z_dimension//chunksize)-1):  
            gen_block = DecoderBlock(self.gen_size * 2**(i+1),
                                     self.num_features,
                                     self.embed_size,
                                     self.chunksize,
                                     (z_dimension//chunksize)-1
                                     )  # 使用不同的 gen_size
            self.gen_blocks.append(gen_block)
        self.BnReLu=nn.ReLU(nn.BatchNorm2d(384))
        self.conv=nn.Conv2d(384,3,kernel_size=3,stride=1,padding='same')
        self.tanh=nn.Tanh()
    def forward(self,z,y):
        #将z切片
        zs = torch.split(z, self.chunksize, dim=0)
        z=zs[0]
        ys=zs[1:]
        y=self.embed(y)
        #拼接
        for i in range(len(ys)):
            zs[i] = torch.cat((y, ys[i]), dim=0)
        z=self.linear(z)
        z=z.view(-1, 384, 4, 4)
        for i,genblock in enumerate(self.genblocks):
            z=genblock(z,ys)
        z=self.BnReLu(z)
        z=self.conv(z)
        z=self.tanh(z)
        return z
            


        




# linear_layer = nn.Linear(20, 384*4*4)
# mapped_condition = linear_layer(condition_vector)
# mapped_condition = mapped_condition.view(-1, 384, 4, 4)

# # 2. 上采样为384x8x8
# upsample_layer = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=True)
# upsampled_condition = upsample_layer(mapped_condition)

# # 3. 使用1x1卷积进行特征处理
# conv1x1 = nn.Conv2d(384, 384, kernel_size=1)
# processed_condition = conv1x1(upsampled_condition)
