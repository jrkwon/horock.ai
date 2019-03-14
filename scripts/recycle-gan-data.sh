#!/bin/bash

A=$1
B=$2
DATA_LOC=$3

if test -z "$DATA_LOC" -o -z "$A" -o -z "$B"; then
	echo "Usage: $0 <name A> <name B> <datasets dir>"
	exit 1
fi

NAMES="$A-$B"

DSTDIRS=( "$DATA_LOC/$NAMES/trainA" "$DATA_LOC/$NAMES/trainB" "$DATA_LOC/$NAMES/testA" "$DATA_LOC/$NAMES/testB" )
SRCDIRS=( "$DATA_LOC/$A-faces-triplets" "$DATA_LOC/$B-faces-triplets" "$DATA_LOC/$A-faces-tests" "$DATA_LOC/$B-faces-tests")

for dir in "${DSTDIRS[@]}"
do
	echo rm -rf $dir
	mkdir -p $dir || exit 1
done

IDX=0
while test $IDX -lt ${#DSTDIRS[@]}
do
	echo "Linking files ${SRCDIRS[$IDX]} to ${DSTDIRS[$IDX]}/"
	find ${SRCDIRS[$IDX]} -name '*.png' | xargs -n 100 echo | 
		while read files
		do
			ln -f $files ${DSTDIRS[$IDX]}/ || exit 1
		done || exit 1
	let 'IDX++'
done

