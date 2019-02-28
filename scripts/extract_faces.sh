if test "$#" -ne 2; then
    echo "Use A|B name"
    exit 1
fi

PROJECT_LOC=../faceswap
DATA_LOC=$PROJECT_LOC/datasets
AB=$1
NAME=$2

mkdir -p $DATA_LOC/data$AB/$NAME-faces
python faceswap.py extract -i $DATA_LOC/data$AB/$NAME -o $DATA_LOC/data$AB/$NAME-faces -D mtcnn -r 45 -ae
