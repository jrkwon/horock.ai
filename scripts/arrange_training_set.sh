#!/bin/bash

NAME="$1"
DATA_LOC="$2"
TRAININGCOUNT="$3"
TESTCOUNT="$4"

if test -z "$NAME" -o -z "$DATA_LOC" -o -z "$TRAININGCOUNT" -o -z "$TESTCOUNT"; then
	echo "Usage: $0 <name> <datasets-dir> <training images number> <test images number>"
	exit 1
fi

if test -x $(dirname $0)/common.sh; then
	. $(dirname $0)/common.sh
fi

CPUS=$(cat /proc/cpuinfo 2>/dev/null | grep processor | wc -l)
CPUS=${CPUS:-4}

echo "I found $CPUS cpus, making triplet image process will be launched at most $CPUS instances"
trap 'killall convert; exit' SIGINT

SRC_DATA_LOC=$DATA_LOC/$NAME-faces
TRAIN_DATA_LOC=$DATA_LOC/$NAME-faces-triplets
TEST_DATA_LOC=$DATA_LOC/$NAME-faces-tests

rm -rf $TRAIN_DATA_LOC $TEST_DATA_LOC
mkdir -p $TRAIN_DATA_LOC $TEST_DATA_LOC

FILE1= FILE2= FILE3= COUNT=0

echo "Searching $SRC_DATA_LOC for png files ..."
find $SRC_DATA_LOC/ -name '*.png' | sort > $SRC_DATA_LOC.txt
echo "Found $(wc -l $SRC_DATA_LOC.txt) files"
COUNT=0
echo "Now linking test images: $TESTCOUNT"
tail -n $TESTCOUNT $SRC_DATA_LOC.txt | while read file
do
	let 'COUNT++'
	outfile=$(printf %05d $COUNT)
	echo -e -n Linking $file to "$TEST_DATA_LOC/$outfile.png\r"
	ln -f $file "$TEST_DATA_LOC/$outfile.png"
done
echo -e "\nNow making triplets images: $TRAININGCOUNT"
head -n $TRAININGCOUNT $SRC_DATA_LOC.txt | while read file
do
	#Strip leading '0's upto six times, and strip trailing .png suffix
	FILENUM=${file##*/}
	FILENUM=${FILENUM##0} FILENUM=${FILENUM##0} FILENUM=${FILENUM##0}
	FILENUM=${FILENUM##0} FILENUM=${FILENUM##0} FILENUM=${FILENUM##0}
	FILENUM=${FILENUM%.png}
	let 'NUMDIFF = FILENUM - PREVFILENUM'
	if test "$NUMDIFF" -eq 0; then
		FILE1= FILE2= FILE3=
		echo -e "\nJump detected at $file"
	fi
        FILE1=$FILE2
        FILE2=$FILE3
        FILE3=$file
	PREVFILENUM=$FILENUM
        if test -n "$FILE1" -a -n "$FILE2" -a -n "$FILE3"; then
                let 'COUNT++'
                outfile="000000$COUNT"
                outfile=${outfile:0-5}
                echo -e -n "convert $FILE1 $FILE2 $FILE3 +append $TRAIN_DATA_LOC/$outfile.png\r"
                convert "$FILE1" "$FILE2" "$FILE3" +append $TRAIN_DATA_LOC/$outfile.png &

		wait_max $CPUS
        fi
done

