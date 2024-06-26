import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")
from Utils.VaeModelTrainer import train_cVAE_GAN
root="F:\ModelAndDataset\data"

# selected_classes= [2,7,9,11,14,15,16,32,35,41]
selected_classes= [7,13,17,32,35]
save_model_dir='F:\ModelAndDataset\model\GTSRB\cVAE_GAN'# 模型参数存放目录
result_dir = 'E:/Project/ZLTProgram/Images/cvae_gan_gtsrb'# 实验结果存放目录

train_cVAE_GAN(root=root,
                selected_classes=selected_classes,
                save_model_dir=save_model_dir,
                nz=100,
                fsize=64,
                Epochs=75,
                result_dir=result_dir
               )