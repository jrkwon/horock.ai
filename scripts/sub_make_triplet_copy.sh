#!/bin/bash

AB=$1
NAME_A=$2
NAME_B=$3
SCRIPTS=scripts

NAME=$NAME_A
if [ $AB = "B" ]
then 
    NAME=$NAME_B
fi
 
SRC_DATA_LOC=datasets/$NAME-faces-train
DST_DATA_LOC=datasets/$NAME-faces-triplet

bash $SCRIPTS/sub_make_triplet_append.sh $SRC_DATA_LOC $DST_DATA_LOC

NAMES=$NAME_A-$NAME_B
SRC_DATA_LOC=$DST_DATA_LOC
DST_DATA_LOC=datasets/$NAMES/train$AB

mkdir -p $DST_DATA_LOC
cp $SRC_DATA_LOC/*.png $DST_DATA_LOC/.
