#!/bin/bash

PROJECT_LOC=../Recycle-GAN

source activate recycle-gan
cd $PROJECT_LOC

python -m visdom.server
