import cv2, dlib, sys
import numpy as np
import argparse

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("video_file")
args = parser.parse_args()
#print(args.video_file)
dataset_dir = '../datasets/'

# video source
video_file = dataset_dir + args.video_file ## '../datasets/jaein.mp4'
shape_predictor = './dlib-models/shape_predictor_68_face_landmarks.dat'
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
    ret, img = self.cap.read()
    if not ret:
      print('Done.')
      return

    # resize frame
    if self.scaler != 1:
      img = cv2.resize(img, (int(img.shape[1] * self.scaler), 
                            (int(img.shape[0] * self.scaler))))
    #org = img.copy()

    # find faces
    if len(self.face_roi) == 0:
      faces = self.detector(img, 1)
    else:
      roi_img = img[self.face_roi[0]:self.face_roi[1], self.face_roi[2]:self.face_roi[3]]
      # cv2.imshow('roi', roi_img)
      faces = self.detector(roi_img)

    # no faces
    #if len(faces) == 0:
    #  print('no faces!')

    # find facial landmarks
    for face in faces:
      if len(self.face_roi) == 0:
        dlib_shape = self.predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
      else:
        dlib_shape = self.predictor(roi_img, face)
        shape_2d = np.array([[p.x + self.face_roi[2], p.y + self.face_roi[0]] for p in dlib_shape.parts()])

      for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

      # compute face center
      center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

      # draw face center
      cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

      # compute face boundaries
      min_coords = np.min(shape_2d, axis=0)
      max_coords = np.max(shape_2d, axis=0)

      # draw min, max coords
      cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
      cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

      # compute face size
      face_size = max(max_coords - min_coords)
      """
      self.face_sizes.append(face_size)
      if len(self.face_sizes) > 10:
        del self.face_sizes[0]
      mean_face_size = int(np.mean(self.face_sizes))
      """

      # compute face roi
      face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
      face_roi = np.clip(face_roi, 0, 10000)

    #self.original_image = org
    self.landmarks_image = img 

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
    facial_landmarks.find_landmarks()
    if facial_landmarks.visualize(img_show_time) == False:
      break
  
if __name__== "__main__":
  main()