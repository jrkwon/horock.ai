# This directory is the holder for training dataset

- Add `actor.mp4` and `target.mp4`. 
- Make `actor-known-faces` and `target-known-faces` folder and place `actor.png` and `target.png` to the respective folders. These .png files must have the _actor_'s face and _target_'s face. They will be used later in face recognition to automatically detect _actor_'s face and _taget_'s face.
- Run make
```
$ make data
```
## Folder structure

### Actor

There will be several folders in the datasets. Ignore others but the followings.

- `actor/`: full images
- `actor-faces/`: detected faces

Check faces in `actor-faces`. Faces are automatically recognized but there might be some errors. Remove the error faces.

Select face images and move to the following folders.
 
- `actor-faces-train/`: selected faces to create triplet images
- `actor-faces-test/`: selected faces to use face generation via recycle-gan

Make triplet images
```
$ bash scripts/make_triplet_images_and_copy.sh actor target
```
Then, the followin folder will be created.

- `actor-faces-triplet/`: generated triplet images for recycle-gan training

### Target
The same sets of folders for _target_.

### Actor-Target

- `actor-target/`: trainA, trainB, testA, testB folders for recycle-gan
 
