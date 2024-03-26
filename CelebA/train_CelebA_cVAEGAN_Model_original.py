from Utils.VaeModelTrainer_original import train_cVAE_GAN

train_cVAE_GAN(root="F:\ModelAndDataset\data",
               attr_name="Smiling",
               save_model_dir="F:\ModelAndDataset\model\CelebA\original_CVAE_GAN"
               )
train_cVAE_GAN(root="F:\ModelAndDataset\data",
               attr_name="Male",
               save_model_dir="F:\ModelAndDataset\model\CelebA\original_CVAE_GAN"
               )
train_cVAE_GAN(root="F:\ModelAndDataset\data",
               attr_name="Young",
               save_model_dir="F:\ModelAndDataset\model\CelebA\original_CVAE_GAN"
               )


