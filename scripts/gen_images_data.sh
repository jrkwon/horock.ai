if test "$#" -ne 2; then
    echo "Use A|B name"
    exit 1
fi

AB=$1
NAME=$2
PROJECT_LOC=../faceswap
DATA_LOC=$PROJECT_LOC/datasets/data$AB/$NAME

cp ../workspace/$NAME.mp4 $DATA_LOC.mp4
mkdir -p  $DATA_LOC/data$AB/$NAME
ffmpeg -i $DATA_LOC/data$AB/$NAME.mp4 -vf fps=30 $DATA_LOC/data$AB/$NAME/%06d.png
