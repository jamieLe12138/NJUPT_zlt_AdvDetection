from Utils.DaGmmUtil import train_DaGmm,test_DaGmm
from Model.CVAE_GAN_AdvancedV2 import CVAE
import torch
from art.attacks.evasion import *
from Model.DAGMM import DAGMM
from os.path import join
device = 'cuda' if torch.cuda.is_available() else 'cpu'
attr_names = [ 'Male', 'Smiling','Young']
#attr_names = [ '','Smiling', "Eyeglasses","Young"]
model_names=["resnet18","vgg19","densenet169","mobilenet"]
epsilons=[0.025,0.05,0.075]
attackers=[FastGradientMethod,BasicIterativeMethod,ProjectedGradientDescent]
cvae_dir="F:\ModelAndDataset\model\CelebA\cVAE_GAN"
dagmm_dir="F:\ModelAndDataset\model\CelebA\DAGMM"
train=False
#===============Train====================
if train:
   for attr_name in attr_names:
      cvae = CVAE(nz=100,
			   imSize=64,
			   enc_self_attn=True,
			   dec_self_attn=True,
			   g_spectral_norm=False,
			   CBN=True,
			   fSize=64,
			   device=device)
      cvae.load_params(join(cvae_dir,attr_name))
      cvae.to(device)
      train_DaGmm(root="F:\ModelAndDataset\data",
                  dagmm_model_path=dagmm_dir,
                  attr_name=attr_name,
                  gen_model=cvae,
                  device=device
                  )
#===============Test==================== 
for attr_name in attr_names:
   cvae = CVAE(nz=100,
			   imSize=64,
			   enc_self_attn=True,
			   dec_self_attn=True,
			   g_spectral_norm=False,
			   CBN=True,
			   fSize=64,
			   device=device)
   cvae.load_params(join(cvae_dir,attr_name))
   cvae.to(device)
   dagmm=DAGMM(3,64,64)
   dagmm.load_params(join(dagmm_dir,attr_name))
   dagmm.to(device)
   for attacker in attackers:
      for model_name in model_names:
         for eps in epsilons:
            test_DaGmm(attr_name=attr_name,
                        gen_model=cvae,
                        dagmm_model=dagmm,
                        Attck_method=attacker,
                        eps=eps,
                        root="F:\ModelAndDataset\data",
                        target_model_dir="F:\ModelAndDataset\model\CelebA",
                        model_name=model_name,
                        test_result_path="E:\Project\ZLTProgram\Images\detection_result",
                        )