#!/usr/bin/env python3

import cv2, dlib, sys, os
import numpy as np
import argparse
import glob
import face_recognition.api as face_recognition

# Advanced Scene Detection Parameters
INTENSITY_THRESHOLD = 16    # Pixel intensity threshold (0-255), default 16
MINIMUM_PERCENT     = 95    # Min. amount of pixels to be below threshold.
BLOCK_SIZE          = 32    # Num. of rows to sum per iteration.

def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

class FaceTracer:
    shape_predictor = 'facial-landmarks/dlib-models/shape_predictor_68_face_landmarks.dat'

    def __init__(self):
        self.person_name = None
        self.person_picture = None

        self.video = None
        self.image = None
        self.sample = None
        self.rfps = 1
        self.scale = 1.0
        self.reco_tolerance = 0.5
        self.roi = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(FaceTracer.shape_predictor)
        self.scene_change_log = ''
        self.scene_keep = False
        self.count = 0
        self.count_in_scene = 0
        self.prev_hsv = None

    def load_person_picture(self, file_path):
        basename = os.path.splitext(os.path.basename(file_path))[0]
        img = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 1:
            print("Error: The picture file has  more than one faces", file_path)
            return False
        elif len(encodings) == 0:
            print("Error: No face is found", file_path)
            return False

        self.person_name    = basename
        self.person_picture = encodings[0]
        return True

    def read_frame(self):
        ret, self.image = self.video.read()
        if self.scale != 1.0:
            w = int(self.image.shape[1] * self.scale)
            h = int(self.image.shape[0] * self.scale)
            self.image = cv2.resize(self.image, (w, h) )
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return ret

    def scene_changed(self):
        #Idea borrowed: https://github.com/Breakthrough/PySceneDetect ... scenedetect/detectors/content_detector.py
        scale = 256. / self.image.shape[0]
        scene = cv2.resize(self.image, (0,0), 0, scale, scale)
        curr_hsv = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV)
        delta_hsv_avg = 0
        self.scene_change_log = ''
        if self.prev_hsv is not None:
            delta_hsv = [0, 0, 0]
            for i in range(3):
                num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
                curr_hsv[i] = curr_hsv[i].astype(np.int32)
                self.prev_hsv[i] = self.prev_hsv[i].astype(np.int32)
                delta_hsv[i] = np.sum(np.abs(curr_hsv[i] - self.prev_hsv[i])) / float(num_pixels)
                self.scene_change_log += '%6.2f ' % (delta_hsv[i])
            delta_hsv_avg = sum(delta_hsv) / 3.0
            self.scene_change_log += 'delta hsv avg: %6.2f' % (delta_hsv_avg)
        self.prev_hsv = curr_hsv
        if delta_hsv_avg > self.scene_threashold:
            return True
        return False

    def recognize(self, faces):
        faces = [_trim_css_to_bounds(_rect_to_css(face), self.sample.shape) for face in faces]
        unknown_encodings = face_recognition.face_encodings(self.sample, faces)
        distances = face_recognition.face_distance(unknown_encodings, self.person_picture)
        for i, distance in enumerate(distances):
            self.reco_distance = distance
            if self.reco_distance <= self.reco_tolerance:
                return i
        return -1

    def inspect(self):
        sensitive_frames = 3
        if self.roi:
            self.sample = self.image[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
            (left, top) = (self.roi[0][0], self.roi[0][1])
        else:
            self.sample = self.image
            (left, top) = (0, 0)

        if self.count_in_scene > sensitive_frames and not self.scene_keep:
            print("{}: Skip this scene is irrelevant".format(self.count))
            if self.count % 5 == 0:
                cv2.imshow('sample', self.sample)
            return

        cv2.imshow('sample', self.sample)

        interested_idx = -1
        faces = self.detector(self.sample, 1)
        interested_idx = self.recognize(faces)
        if interested_idx == -1:
            print("{}: The face is not found".format(self.count))
            return

        if self.count_in_scene <= sensitive_frames:
            self.scene_keep = True

        face = faces[interested_idx]

        shape = self.predictor(self.sample, face)
        shape_2d = np.array([[left+p.x, top+p.y] for p in shape.parts()])
        for s in shape_2d:
            cv2.circle(self.image, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)
        size = max_coords - min_coords
        size[0] /= 1.5
        size[1] /= 1.5
        adj_min = tuple(min_coords - size)
        adj_max = tuple(max_coords + size)
        cv2.rectangle(self.image, adj_min, adj_max, (255, 255, 0))

        (face_left, face_top, face_right, face_bottom) = \
            ( left + face.left(), top + face.top(), left + face.right(), top + face.bottom() )
        cv2.rectangle(self.image, (face_left, face_top), (face_right, face_bottom), (255,0,0) )
        print("{}: Area:({},{})-({},{}) Scene diff: {}, Reco dist: {:6.2f}".format(self.count, face_left, face_top, face_right, face_bottom, self.scene_change_log, self.reco_distance))

        if not self.roi and interested_idx != -1:
            self.roi = (adj_min, adj_max)

    def run(self, args):
        print("Picture: ", args.picture_file)
        print("Video  : ", args.video_file)

        if not self.load_person_picture(args.picture_file):
            return

        self.video = cv2.VideoCapture(args.video_file)
        self.scale = args.scale
        self.scene_threashold = args.scene_threashold
        self.reco_tolerance = args.reco

        self.count = 0
        while True:
            if not self.read_frame():
                break

            self.count += 1
            if self.count < args.start:
                continue

            if self.scene_changed():
                print("{}: Scene changed.....................................................................................".format(self.count))
                self.roi = None
                self.scene_keep = False
                self.count_in_scene = 0

            self.count_in_scene += 1
            self.inspect()

            image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            cv2.imshow('facial landmarks', image)
            if cv2.waitKey(self.rfps) == ord('q'):
              break


parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, default=1.0, help="Scale factor before processing (default: 1.0)")
parser.add_argument("--log_file", type=str, default=None, help="Face detect log")
parser.add_argument("--scene_threashold", type=float, default=30.0, help="Scene change detection diff-hsv threashold (default: 30.0)")
parser.add_argument("--reco", type=float, default=0.5, help="Face diff tolerence (default: 0.5)")
parser.add_argument("--start", type=int, default=0, help="Start frame")
parser.add_argument("picture_file", default=None, help="Reference face image file")
parser.add_argument("video_file", default=None, help="Source video file")
args = parser.parse_args()

trace = FaceTracer()
trace.run(args)
