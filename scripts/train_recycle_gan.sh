#!/bin/bash

if test "$#" -ne 3; then
    echo "Use nameA nameB gpu_id"
    exit 1
fi

NAMES=$1-$2
GPU_ID=$3

PROJECT_LOC=../Recycle-GAN
DATA_LOC=datasets/faces/$NAMES

source activate recycle-gan
cd $PROJECT_LOC
python train.py --dataroot $DATA_LOC/ --name $NAMES --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 --dataset_mode unaligned_triplet --no_dropout --gpu_ids $GPU_ID --identity 0  --pool_size 0 

