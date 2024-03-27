from Utils.DaGmmUtil import train_DaGmm,test_DaGmm
from Model.CVAE_GAN import CVAE
import torch
from art.attacks.evasion import *
from Model.DAGMM import DAGMM
from os.path import join
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
attr_names = [ 'Male', 'Smiling','Young']
#attr_names = [ '','Smiling', "Eyeglasses","Young"]
model_names=["resnet18","vgg19","densenet169","mobilenet"]
epsilons=[0.025,0.05,0.075]
attackers={"FastGradientMethod":FastGradientMethod,
           "BasicIterativeMethod":BasicIterativeMethod,
           "ProjectedGradientDescent":ProjectedGradientDescent}
cvae_dir="F:\ModelAndDataset\model\CelebA\original_CVAE_GAN"
dagmm_dir="F:\ModelAndDataset\model\CelebA\original_DAGMM"
train=False
#===============Train====================
if train:
   for attr_name in attr_names:
      cvae = CVAE(nz=100,
			   imSize=64,
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
			   fSize=64,
			   device=device)
   cvae.load_params(join(cvae_dir,attr_name))
   cvae.to(device)
   dagmm=DAGMM(3,64,64)
   dagmm.load_params(join(dagmm_dir,attr_name))
   dagmm.to(device)
   test_result_path="E:\Project\ZLTProgram\Images\detection_result\CelebA_cVAE_GAN_Original"
   
   for attacker_name,attacker in attackers.items():
      for model_name in model_names:
         for eps in epsilons:
            format_epsilon= "{:e}".format(eps).replace(".","")
            pic_name='{}_CelebA_{}_{}_{}.png'.format(model_name,attr_name,attacker_name,format_epsilon)
            pic_savepath=join(test_result_path,pic_name)
            print(pic_savepath)
            if os.path.exists(pic_savepath):              
               print("Dagmm test finished!")
               continue
            test_DaGmm(attr_name=attr_name,
                        gen_model=cvae,
                        dagmm_model=dagmm,
                        Attck_method=attacker,
                        eps=eps,
                        root="F:\ModelAndDataset\data",
                        target_model_dir="F:\ModelAndDataset\model\CelebA",
                        model_name=model_name,
                        test_result_path=test_result_path,
                        )