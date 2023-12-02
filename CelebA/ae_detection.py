from Utils.DaGmmUtil import test_DaGmm
from art.attacks.evasion import *
from Model.CVAE_GAN_AdvancedV2 import *
from Model.DAGMM import DAGMM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

attacker=FastGradientMethod
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
dagmm=DAGMM(3,64,64)
dagmm.load_params("E:\Project\ModelAndDataset\model\CelebA\DAGMM\Smiling")
dagmm.to(device)
test_DaGmm(attr_name="Smiling",
           gen_model=cvae,
           dagmm_model=dagmm,
           Attck_method=attacker,
           model_name="resnet18",
           test_result_path="E:\Project\ZLTProgram\Images\detection_result",
        )