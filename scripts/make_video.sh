#!/bin/bash

if test "$#" -ne 4; then
    echo "Use AB|BA nameA nameB frame_rate"
    exit 1
fi

DIRECTION=$1
NAMES=$2-$3
FRAME_RATE=$4

SRC=A
DST=B
if [ $DIRECTION = "AB" ]; then
    echo "A->B"
elif [ $DIRECTION = "BA" ]; then
    echo "B->A"
    SRC=B
    DST=A
fi
PROJECT_LOC=Recycle-GAN
DATA_LOC=datasets
REAL_DATA=$PROJECT_LOC/results/$NAMES/test_latest/images/%05d_real_$SRC.png
REAL_OUTPUT=$DATA_LOC/$NAMES/real_$SRC.mp4
FAKE_DATA=$PROJECT_LOC/results/$NAMES/test_latest/images/%05d_fake_$DST.png
FAKE_OUTPUT=$DATA_LOC/$NAMES/fake_$DST.mp4
REAL_FAKE_OUTPUT=$DATA_LOC/$NAMES/real_${SRC}_fake_$DST.mp4

echo Making $REAL_OUTPUT ...
ffmpeg -i $REAL_DATA -framerate $FRAME_RATE $REAL_OUTPUT
echo Making $FAKE_OUTPUT ...
ffmpeg -i $FAKE_DATA -framerate $FRAME_RATE $FAKE_OUTPUT

echo Making $REAL_FAKE_OUTPUT ...
ffmpeg -i $REAL_OUTPUT -i $FAKE_OUTPUT -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map [vid] $REAL_FAKE_OUTPUT

