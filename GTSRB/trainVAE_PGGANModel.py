import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")
from Utils.VaePG_GANModelTrainer import train_cVAE_GAN
root="F:\ModelAndDataset\data"

# selected_classes= [2,7,9,11,14,15,16,32,35,41]
selected_classes= [7,13,17,32,35]
save_model_dir='F:\ModelAndDataset\model\GTSRB\cVAE_PG_GAN'# 模型参数存放目录
result_dir = 'E:/Project/ZLTProgram/Images/cvae_pg_gan_gtsrb'# 实验结果存放目录

train_cVAE_GAN(selected_classes=selected_classes,
               root="F:\ModelAndDataset\data",
               train_batch_size=64,
               test_batch_size=64,
               nz=100,
               fsize=32,
               stages=4,
               epochs_per_stage=40,
               save_model_dir=save_model_dir,
               result_dir=result_dir
                ) 