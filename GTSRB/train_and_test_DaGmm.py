from Utils.DaGmmUtil import train_DaGmm,test_DaGmm
from Model.CVAE_GAN import CVAE
import torch
from art.attacks.evasion import *
from Model.DAGMM import DAGMM
from os.path import join
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_names=["resnet18","vgg19","densenet169","mobilenet"]
attackers=[FastGradientMethod,BasicIterativeMethod,ProjectedGradientDescent]
cvae_dir="F:\ModelAndDataset\model\GTSRB\cVAE_GAN"
dagmm_dir="F:\ModelAndDataset\model\GTSRB\DAGMM"
selected_classes= [2,7,9,11,14,15,16,32,35,41]
train=True
#===============Train====================
if train:
   cvae = CVAE(nz=100,
			   imSize=64,
			   enc_self_attn=True,
			   dec_self_attn=True,
			   g_spectral_norm=False,
			   CBN=True,
			   fSize=64,
            numLabels=10,
			   device=device)
   cvae.load_params(cvae_dir,10)
   cvae.to(device)
   train_DaGmm(selected_classes=selected_classes,
               gen_model=cvae,
               root="F:\ModelAndDataset\data",
               dagmm_model_path=dagmm_dir, 
               num_epochs=2,              
               device=device
               )
#===============Test==================== 

cvae = CVAE(nz=100,
			   imSize=64,
			   enc_self_attn=True,
			   dec_self_attn=True,
			   g_spectral_norm=False,
			   CBN=True,
			   fSize=64,
            numLabels=10,
			   device=device)
cvae.load_params(cvae_dir,10)
cvae.to(device)
dagmm=DAGMM(3,64,64)
dagmm.load_params(dagmm_dir)
dagmm.to(device)
for attacker in attackers:
   for model_name in model_names:
      test_DaGmm(selected_classes=selected_classes,
                  gen_model=cvae,
                  dagmm_model=dagmm,
                  Attck_method=attacker,
                  root="F:\ModelAndDataset\data",
                  target_model_dir="F:\ModelAndDataset\model\GTSRB",
                  model_name=model_name,
                  test_result_path="E:\Project\ZLTProgram\Images\detection_result",
                  )