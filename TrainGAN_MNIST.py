import os
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from Discriminator_MNIST import discriminator_mnist
from Generator_MNIST import generator_mnist
from torchvision.utils import save_image
# 创建文件夹
if not os.path.exists('./img_CGAN_MNIST'):
    os.mkdir('./img_CGAN_MNIST')
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_epoch = 60
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
#定义生成器与判别器
D=discriminator_mnist()
G=generator_mnist(z_dimension=100)
D=D.to(device)
G=G.to(device)
#判别器训练train
criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

z_dimension=100
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, label) in enumerate(dataloader):
        num_img = img.size(0)
        #生成独热编码真实标签
        label_onehot = torch.zeros((num_img,10)).to(device)
        label_onehot[torch.arange(num_img),label]=1
        #将图片展开为28*28=784
        img = img.view(num_img,  -1)
        real_img = img.to(device)
        real_label = label_onehot
    
        fake_label = torch.zeros((num_img,10)).to(device)
        # 计算真实图片的损失
        real_out = D(real_img)  # 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        # 计算假的图片的损失
        z = torch.randn(num_img, z_dimension+10).to(device)  # 随机生成一些噪声
        fake_img = G(z)  # 随机噪声放入生成网络中，生成一张假的图片
        fake_out = D(fake_img)  # 判别器判断假的图片
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss

        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

        # 训练生成器
        z = torch.randn(num_img, z_dimension).to(device) # 得到随机噪声
        z = torch.cat([z, real_label],1)
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)
        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
        # 打印中间的损失
        # try:
        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '.format(
                epoch, num_epoch, d_loss.item(), g_loss.item(),
            ))
        # except BaseException as e:
        #     pass
        if epoch == 0:
            real_images = real_img.cpu().clamp(0,1).view(-1,1,28,28).data
            save_image(real_images, './img_CGAN_MNIST/real_images.png')
        if i == len(dataloader)-1:
            fake_images = fake_img.cpu().clamp(0,1).view(-1,1,28,28).data
            save_image(fake_images, './img_CGAN_MNIST/fake_images-{}.png'.format(epoch + 1))
# 保存模型
torch.save(G.state_dict(), 'E:/Project/ModelAndDataset/generator_CGAN_mnist.pth')
torch.save(D.state_dict(), 'E:/Project/ModelAndDataset/model/discriminator_CGAN_mnist.pth')