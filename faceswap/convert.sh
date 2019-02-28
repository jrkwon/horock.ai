rm datasets/dataA/idol1/*
python faceswap.py convert -i datasets/dataA/actress1 -o datasets/dataA/idol1 -a datasets/dataA/actress1/extracted -m model/model1 -t OriginalHighRes -c Adjust -sm
