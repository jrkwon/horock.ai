# Facial Landmarks Detection

## Folder

Create `dlib-models` folder

## Pretrained model

Download this file and unzip it from `https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2`. Copy the `shape_predictor_68_face_landmarks.dat` file to `dlib-models`

## How to use

The video file folder is `datasets`.
```
$ python facial-landmarks/main.py video.mp4
```

If you want scaled output
```
$ python facial-landmarks/main.py --scale=.25 video.mp4
```

You can also speficify extracted frame files with wildcard.
```
$ python facial-landmarks/main.py jaein-faces/*/*/*.png
```



