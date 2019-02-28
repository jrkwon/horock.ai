#!/bin/bash

AB=$1
NAME_A=$2
NAME_B=$3

NAME=$NAME_A
if [ $AB = "B" ]
then 
    NAME=$NAME_B
fi
 
PROJECT_LOC=../faceswap
SRC_DATA_LOC=$PROJECT_LOC/datasets/data$AB/$NAME-faces
DST_DATA_LOC=$PROJECT_LOC/datasets/data$AB/$NAME-faces-triplet

bash sub_make_triplet_append.sh $SRC_DATA_LOC $DST_DATA_LOC

NAMES=$NAME_A-$NAME_B
PROJECT_LOC=../Recycle-GAN
SRC_DATA_LOC=$DST_DATA_LOC
DST_DATA_LOC=$PROJECT_LOC/datasets/faces/$NAMES/train$AB

mkdir -p $DST_DATA_LOC
cp $SRC_DATA_LOC/*.png $DST_DATA_LOC/.
