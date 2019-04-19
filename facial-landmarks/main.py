import cv2, dlib, sys
import numpy as np
import argparse

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("video_file")
args = parser.parse_args()
#print(args.video_file)
dataset_dir = 'datasets/'

# use a number to identify a camera for a live video streaming
# default camera is 0
if '.mp4' in args.video_file:
  # video source
  video_file = dataset_dir + args.video_file ## '../datasets/jaein.mp4'
else:
  video_file = int(args.video_file)

shape_predictor = 'facial-landmarks/dlib-models/shape_predictor_68_face_landmarks.dat'
img_show_time = 1 ## 25 # ms 

class FacialLandmarks:
  def __init__(self):
    self.scaler = 1
    # initialize face detector and shape predictor
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(shape_predictor)
    self.face_roi = []
    #self.face_sizes = []
    self.original_image = None
    self.landmarks_image = None

  def load_video(self, video_file):
    # load video
    self.cap = cv2.VideoCapture(video_file)

  def find_landmarks(self):
    # read frame buffer from video
    ret, self.landmarks_image = self.cap.read()
    if not ret:
      print('Done.')
      return False # no more images from the video

    # resize frame
    if self.scaler != 1:
      img = cv2.resize(img, (int(self.landmarks_image.shape[1] * self.scaler), 
                            (int(self.landmarks_image.shape[0] * self.scaler))))
    #org = img.copy()

    # find faces
    roi_img = None
    if len(self.face_roi) == 0:
      faces = self.detector(self.landmarks_image, 1)
    else:
      roi_img = self.landmarks_image[self.face_roi[0]:self.face_roi[1], self.face_roi[2]:self.face_roi[3]]
      cv2.imshow('roi', roi_img)
      faces = self.detector(roi_img)

    # no faces
    if len(faces) == 0:
      #print('no faces!')
      # reset roi
      #self.face_roi.clear()
      return True
    #else:
    #  face = faces[0]

    # find facial landmarks
    for face in faces: # process the 1st face only
    # ------>
      if len(self.face_roi) == 0:
        dlib_shape = self.predictor(self.landmarks_image, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
      else:
        dlib_shape = self.predictor(roi_img, face)
        shape_2d = np.array([[p.x + self.face_roi[2], p.y + self.face_roi[0]] for p in dlib_shape.parts()])

      for s in shape_2d:
        cv2.circle(self.landmarks_image, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

      # compute face center
      center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

      # draw face center
      cv2.circle(self.landmarks_image, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

      # compute face boundaries
      min_coords = np.min(shape_2d, axis=0)
      max_coords = np.max(shape_2d, axis=0)

      # draw min, max coords
      cv2.circle(self.landmarks_image, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
      cv2.circle(self.landmarks_image, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

      # compute face size
      face_size = max(max_coords - min_coords)
      """
      self.face_sizes.append(face_size)
      if len(self.face_sizes) > 10:
        del self.face_sizes[0]
      mean_face_size = int(np.mean(self.face_sizes))
      """

      """
      # compute face roi
      if min_coords[1] > face_size/2:
        left = int(min_coords[1] - face_size/2)
      else:
        left = 0
      if min_coords[0] > face_size/2:
        top = int(min_coords[0] - face_size/2)
      else:
        top = 0
      right = int(max_coords[1] + face_size/2)
      bottom = int(max_coords[0] + face_size/2)
      """

      #self.face_roi = [top, bottom, left, right]
      #self.face_roi = np.clip(face_roi, 0, 10000)
    # <----

    #self.original_image = org
    #self.landmarks_image = img 
    return True

  def visualize(self, img_show_time):
    # visualize
    # cv2.imshow('original', self.original_image)
    cv2.imshow('facial landmarks', self.landmarks_image)
    if cv2.waitKey(img_show_time) == ord('q'):
      return False
    return True


def main():
  facial_landmarks = FacialLandmarks()
  facial_landmarks.load_video(video_file)
  while True:
    if facial_landmarks.find_landmarks() == False: # no more images
      break 
    if facial_landmarks.visualize(img_show_time) == False:
      break
  
if __name__== "__main__":
  main()
