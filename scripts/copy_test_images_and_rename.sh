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
 
SRC_DATA_LOC=datasets/$NAME-faces-test

NAMES=$NAME_A-$NAME_B
DST_DATA_LOC=datasets/$NAMES/test$AB

mkdir -p $DST_DATA_LOC
cp $SRC_DATA_LOC/*.png $DST_DATA_LOC/.

ls $DST_DATA_LOC | cat -n | while read n f; do mv $DST_DATA_LOC/"$f" `printf $DST_DATA_LOC/"%05d.png" $n`; done
