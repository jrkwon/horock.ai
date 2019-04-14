#!/bin/bash

if test "$#" -ne 4; then
    echo "Use nameA nameB gpu_id visdom_port"
    exit 1
fi

NAMES=$1-$2
GPU_ID=$3
VISDOM_PORT=$4

PROJECT_LOC=Recycle-GAN
DATA_LOC=${DATA_LOC:-../datasets}
PIDFILE=$( cd $(dirname $0)/../stamps/ && echo $(pwd)/train.pid )

#source activate recycle-gan
cd $PROJECT_LOC
python train.py --dataroot $DATA_LOC/$NAMES --name $NAMES --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 --dataset_mode unaligned_triplet --no_dropout --gpu_ids $GPU_ID --identity 0  --pool_size 0  --display_port $VISDOM_PORT &
echo $! > $PIDFILE
wait
