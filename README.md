# HOROCK.AI

This repository describes the image processing pipeline to use Recycle-GAN to generate faces. When we want to make a video images B based on video images A. We call the video A an _actor_ and call the video B a _target_.

# Usage

## FILES

prepare.py : make train, test videos
train.py: train the NN model
test.py: generate images using the trained NN model.
mkvideo.py: make video

## Environment
- If 
  - no X window is available, see USAGE-X-Window.md.
  - you use MacOS, see USAGE-Iterm2.md 를 참고하세요.
- Activate the Anaconda env,`conda-horock`
```
    . conda-horock
```

## Prepare Videos
- Install `youtube-dl`.
```
  sudo apt install youtube-dl
```
- Update `youtube-dl` if necessary.
```
  sudo youtube-dl --update
```
- Download videos using `youtube-dl`
- Use `-F` for 360p format.
- Use `-f 43`. 
```
 youtube-dl -F 'https://www.youtube.com/watch?v=4zg1oLQ3gOI'
 youtube-dl -f 133 -o muhyun.webm 'https://www.youtube.com/watch?v=4zg1oLQ3gOI'
 youtube-dl -F 'https://www.youtube.com/watch?v=GPKgbcO5ppw'
 youtube-dl -f 43 -o jaein.webm 'https://www.youtube.com/watch?v=GPKgbcO5ppw'
```
- Move the videos to `datasets/`.

## Choose Reference Image

```
    ./prepare.py pic foo
```
This command searches images where the full shot of a subject is shown and saves them to `datasets/<name>/pic`.
Then it selects an image in random and creates a sympboic link `datasets/<name>.png`.

You can select specif frames. 

```
    ./prepare.py pic foo --begin=2000 --end=3000
```

   The number of samples can be specified. 

```
    ./prepare.py pic foo --samples=500
```

    To specify an image as the reference image,
```
    ./prepare.py pic foo --samples=0
```

ex)
```
    ./prepare.py pic muhyun --begin=2000 --end=3000
    ./prepare.py pic jaein
```

## Extract Images for Training
```
    ./prepare.py train foo
```

    The images are saved to `./datasets/<name>/train`. The sub-directories are created based on the image size.

```
    ./datasets/foo/train/724x648
    ./datasets/foo/train/502x500
```
    
After extracing images, the command finds a directory where the number of images is the biggest. Then it creates a symbolic link to ` ./datasets/foo/images/`

```
    ./prepare.py train muhyun --begin=2000 --end=3000
    ./prepare.py train jaein
```

## Create Test Image

```
    ./prepare test1 muhyun --begin=4000
    ./prepare test1 jaein --begin=3000
```

## Train the NN model

```
    ./train.py foo1 foo2
```

## Generate Target Image

```
    ./test.py foo1 foo2
```

## Make Video

```
    ./mkvideo.py foo1 foo2 AB

    or

    ./mkvideo.py foo1 foo2 AB

