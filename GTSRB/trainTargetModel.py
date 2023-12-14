import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")
from Utils.TargetModelTrainer import train_GTSRB_target_model

root='F:\ModelAndDataset\data'
save_dir="F:\ModelAndDataset\model\GTSRB"


model_names=['resnet18','vgg19','densenet169','mobilenet']
#selected_classes=[2,7,9,11,14,15,27,29,30,35]
# selected_classes= [2,7,9,11,14,15,16,32,35,41]
selected_classes= [7,13,17,32,35]


for model_name in model_names:
    if model_name=="vgg19":
        train_GTSRB_target_model(root=root,
                            save_dir=save_dir,
                            selected_classes=selected_classes,
                            save=True,
                            num_epochs=30,
                            optimizer="SGD",
                            model=model_name,
                            device='cuda',
                         )
    else:
        train_GTSRB_target_model(root=root,
                            save_dir=save_dir,
                            selected_classes=selected_classes,
                            save=True,
                            num_epochs=30,
                            optimizer="Adam",
                            model=model_name,
                            device='cuda',
                         )