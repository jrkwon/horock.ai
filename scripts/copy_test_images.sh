#!/bin/bash

if test "$#" -ne 3; then
    echo "Use A|B nameA nameB"
    exit 1
fi

AB=$1
NAME_A=$2
NAME_B=$3

NAME=$NAME_A
if [ $AB = "B" ]
then 
    NAME=$NAME_B
fi
 
PROJECT_LOC=../faceswap
SRC_DATA_LOC=$PROJECT_LOC/datasets/data$AB/$NAME-faces-test

NAMES=$NAME_A-$NAME_B
PROJECT_LOC=../Recycle-GAN
DST_DATA_LOC=$PROJECT_LOC/datasets/faces/$NAMES/test$AB

mkdir -p $DST_DATA_LOC
cp $SRC_DATA_LOC/*.png $DST_DATA_LOC/.
