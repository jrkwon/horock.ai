#!/bin/bash

if test "$#" -ne 5; then
    echo "Use nameA nameB num_test_images gpu_id epoch"
    exit 1
fi

NAME_A=$1
NAME_B=$2
NAMES=$NAME_A-$NAME_B
NUM_TEST_IMAGES=$3
GPU_ID=$4
EPOCH=$5

PROJECT_LOC=Recycle-GAN
DATA_LOC=../datasets/$NAMES
cd $PROJECT_LOC
#source activate recycle-gan
python test.py --which_epoch $EPOCH --dataroot $DATA_LOC --name $NAMES --model cycle_gan  --which_model_netG resnet_6blocks   --dataset_mode unaligned  --no_dropout --gpu $GPU_ID  --how_many $NUM_TEST_IMAGES  --loadSize 256

cd .. 
#source deactivate
