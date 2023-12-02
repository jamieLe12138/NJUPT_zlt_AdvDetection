from Utils.DaGmmUtil import train_DaGmm
from Model.CVAE_GAN_AdvancedV2 import CVAE
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cvae = CVAE(nz=100,
			 imSize=64,
			 enc_self_attn=True,
			 dec_self_attn=True,
			 g_spectral_norm=False,
			 CBN=True,
			 fSize=64,
			 device=device)
cvae.load_params("E:/Project/ModelAndDataset/model/CelebA/cVAE_GAN/Smiling")
cvae.to(device)
train_DaGmm(attr_name="Smiling",
            gen_model=cvae,
            device=device
            )