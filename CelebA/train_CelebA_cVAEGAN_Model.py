from Utils.VaeModelTrainer import train_cVAE_GAN

train_cVAE_GAN(root="F:\ModelAndDataset\data",
               attr_name="Eyeglasses",
               save_model_dir="F:\ModelAndDataset\model\CelebA\cVAE_GAN"
               )
train_cVAE_GAN(root="F:\ModelAndDataset\data",
               attr_name="Male",
               save_model_dir="F:\ModelAndDataset\model\CelebA\cVAE_GAN"
               )
train_cVAE_GAN(root="F:\ModelAndDataset\data",
               attr_name="Young",
               save_model_dir="F:\ModelAndDataset\model\CelebA\cVAE_GAN"
               )


