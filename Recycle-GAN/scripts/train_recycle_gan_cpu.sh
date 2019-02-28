#!./scripts/train_recycle_gan_cpu.sh
python train.py --dataroot ./datasets/faces/$1/ --name $1  --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 --dataset_mode unaligned_triplet --no_dropout --gpu -1 --identity 0  --pool_size 0 
