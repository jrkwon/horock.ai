# HOROCK.AI

This repository describes the image processing pipeline to use Recycle-GAN to generate faces. When we want to make a video images B based on video images A. We call the video A an _actor_ and call the video B a _target_.
The total length of a video that has a face is at least five minutes. The interviews or public addresses of the _actor_ and _target_ could be a good choice. To get a five-minute fake face video, we need to have a pair of videos whose length is between 20 to 30 minutes.

 - 9,000 images of a face for training. This is about 5 minutes in 30 fps.
 - 1,000 images of a face to test. This is the about 0.5-minute length.

## Video preparation 
Find a pair of videos and save as the `.mp4` format to the `workspace` folder.

## Folder structure

- workspace: The location where a pair of video files reside.
- scripts: The shell scripts that can be used to process the videos and images.
- Recycle-GAN: Recycle-GAN: Unsupervised Video Retargeting. I forked this from [here](https://github.com/aayushbansal/Recycle-GAN).
- faceswap: I forked this from [here](https://github.com/deepfakes/faceswap)

## Use scripts
Go to the `scripts` folder to start. This folder must be your workspace folder.

### Convert mp4 to images

- gen_images_data.sh: This will convert a video file to `%05d.png` images. The output images will be at `faceswap/datasets/dataA/actor_name/` or `faceswap/datasets/dataB/target_name`.

```
$bash gen_images_data A actor
$bash gen_images_data B target
```

### Extract faces from the images
Two choices: an automatic face finding or setting a cropping region.

#### Use faceswap
- extract_faces.sh: Extract faces from the images. This will generate face images that are aligned to the eyes.
```
$ bash extract_faces.sh A actor
$ bash extract_faces.sh B target
```

#### Use simple cropping and resizing.

When faces show in a particular position in the video, you may crop a certain area.

- crop_images.sh: Crop images with a specified area. (width, height, x_offset, y_offset)
```
$ bash crop_images.sh A actor 400 400 427 24
$ bash crop_images.sh B actor 270 270 480 70
```

### Remove non-facials images from the extraction/cropping

Do this manually. You may use `face_recogntion` to classify faces.
After the recognition, you may still need to do the manual job.

### Making training and test datasets

Set aside around 1,000 face images from each of the actor and target.

#### Make training datasets

After getting face images, the training triplet images must be prepared.

- make_triplet_images_and_copy.sh: This makes triplet images and copy them to the right place inside `Recycle-GAN`.

```
$ bash make_triplet_images_and_copy.sh actor target
```
`sub_make_triplet_copy.sh` and `sub_make_triplet_append.sh` are used from `make_triplet_images_and_copy.sh`.

#### Make test datasets

```
$ bash copy_test_images.sh A actor target
$ bash copy_test_images.sh B actor target
```
### Start visdom

To see the training progress, use `start_visdom.sh`
```
$ bash start_visdom.sh
```
It will start a web server showing the progress of the training. Use the port number 8097 to access the page.

### Start training

Start `train_recycle_gan.sh` with the actor name, target name, and the GPU ID (starting from 0).
```
$bash train_recycle_gan.sh actor target 0
```

The training will take hours. It takes 16 to 40 epochs to stabilize the generator's output.

### Rename test images

The test image names must be sequential to create a video.
```
$ bash rename_test_images.sh A actor target
$ bash rename_test_images.sh B actor target
```

### Test the trained Recycle-GAN

- test_recycle_gan.sh: This uses the latest trained network model to generate the _target_ faces from the _actor_ faces. (Actually, this process will generate the _target_ faces from the _actor_ faces at the same time).

### Make side-by-side videos
- `AB` means that real _actor_ faces will generate fake _target_ faces.
- `BA` means that real _target_ faces will genereate fake _actor_ faces.
- The last parameter is the frame rate of the generated video.
```
$ bash make_video.sh AB actor target 5
$ bash make_video.sh BA actor target 5
```
