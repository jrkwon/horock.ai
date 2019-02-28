#!./scripts/test_recycle_gan.sh
python test.py --dataroot ./datasets/faces/$1 --name $1 --model cycle_gan  --which_model_netG resnet_6blocks   --dataset_mode unaligned  --no_dropout --gpu 0  --how_many 595  --loadSize 256 
