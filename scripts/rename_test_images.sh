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

NAMES=$NAME_A-$NAME_B
PROJECT_LOC=../Recycle-GAN
DATA_LOC=$PROJECT_LOC/datasets/faces/$NAMES/test$AB

ls $DATA_LOC | cat -n | while read n f; do mv $DATA_LOC/"$f" `printf $DATA_LOC/"%05d.png" $n`; done
