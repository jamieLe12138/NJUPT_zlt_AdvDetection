import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")
from Utils.AT_TrainingUtil import train_GTSRB_at_model
from art.attacks.evasion import *
import torch.nn.functional as F
root='F:\ModelAndDataset\data'
save_dir="F:\ModelAndDataset\model\GTSRB"


model_names=['resnet18','vgg19','densenet169','mobilenet']
#selected_classes=[2,7,9,11,14,15,27,29,30,35]
# selected_classes= [2,7,9,11,14,15,16,32,35,41]
selected_classes= [7,13,17,32,35]

for model_name in model_names:
    train_GTSRB_at_model(root=root,
                             selected_classes=selected_classes,
                             num_epochs=10,
                             batch_size=128,
                             at_model_name=model_name,
                             at_model_dir=save_dir,
                             target_model_name=model_name,
                             target_model_dir=save_dir,
                             test_result_path="E:\Project\ZLTProgram\Images\detection_result",
                             criterion=F.cross_entropy,
                             Attack_method=ProjectedGradientDescent,
                             train_eps=0.1,
                             test_eps=0.05,
                             device="cuda",
                             save=True
                             )
    