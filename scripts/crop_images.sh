if test "$#" -ne 6; then
    echo "Use A|B name xsize ysize xoffset yoffset"
    exit 1
fi

AB=$1
NAME=$2
XSIZE=$3
YSIZE=$4
XOFFSET=$5
YOFFSET=$6

PROJECT_LOC=../faceswap
SRC_DATA_LOC=$PROJECT_LOC/datasets/data$AB/$NAME
DST_DATA_LOC=$PROJECT_LOC/datasets/data$AB/$NAME-crop-resize

mkdir -p $DST_DATA_LOC

mogrify -path $DST_DATA_LOC/ \
        -crop ${XSIZE}x${YSIZE}+${XOFFSET}+${YOFFSET} \
        -resize 256x256^ \
        $SRC_DATA_LOC/*.png
