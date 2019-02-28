#!/bin/bash

if test "$#" -ne 2; then
    echo "Use NameA NameB"
    exit 1
fi

NAME_A=$1
NAME_B=$2
NAMES=$NAME_A-$NAME_B

bash sub_make_triplet_copy.sh A $NAME_A $NAME_B
bash sub_make_triplet_copy.sh B $NAME_A $NAME_B 

