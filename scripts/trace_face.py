#!/usr/bin/env python3

import cv2, dlib, sys, os
import numpy as np
import argparse
import glob
import face_recognition.api as face_recognition

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

    def __init__(self):
        self.person_name = None
        self.person_picture = None

        self.video = None
        self.time_pos = 0.
        self.frames = 0
        self.history = []
        self.image = None
        self.sample = None
        self.rfps = 1
        self.scale = 1.0
        self.detect_width = 320
        self.detect_scale = None
        self.reco_tolerance = 0.5
        self.roi = None
        self.detector = face_recognition.face_detector
        self.predictor = face_recognition.pose_predictor_68_point

        self.scene_change_log = ''
        self.scene_keep = False
        self.scene_prev = None
        self.scene_image = None

        self.count = 0
        self.count_in_scene = 0
        self.full_shot = False
        self.full_shot_threshold = .15

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

    def get_time_pos(self):
        return "{} {:05d} {:02d}:{:02d}.{:03d})".format(self.count, self.frames, (self.time_pos // 1000) // 60, (self.time_pos // 1000) % 60, self.time_pos % 1000)

    def fast_forward(self, msec):
        print("{}: Fast forward {} msecs".format(self.get_time_pos(), msec))
        self.time_pos = self.video.set(cv2.CAP_PROP_POS_MSEC, float(self.time_pos + msec))

    def read_frame(self):
        self.time_pos = int(self.video.get(cv2.CAP_PROP_POS_MSEC))
        self.frames = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        ret, self.image = self.video.read()
        if len(self.history) > 5:
            self.history.pop(0)
        self.history.append(self.image)

        if not self.detect_scale:
            self.detect_scale = self.detect_width / self.image.shape[1]

        self.display_image = cv2.resize(self.image, (0,0), 0, self.scale, self.scale )
        self.image = cv2.resize(self.image, (0,0), 0, self.detect_scale, self.detect_scale )
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.scene_prev = self.scene_image
        scale = 64. / self.image.shape[1]
        scene = cv2.resize(self.image, (0,0), 0, scale, scale)
        self.scene_image = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV, cv2.INTER_LANCZOS4)
        return ret

    def scene_changed(self):
        #Idea borrowed: https://github.com/Breakthrough/PySceneDetect ... scenedetect/detectors/content_detector.py
        self.scene_change_log = ''
        if self.scene_prev is None:
            return True

        delta_hsv = [0, 0, 0]
        for i in range(3):
            num_pixels = self.scene_image[i].shape[0] * self.scene_image[i].shape[1]
            self.scene_image[i] = self.scene_image[i].astype(np.int32)
            self.scene_prev[i] = self.scene_prev[i].astype(np.int32)
            delta_hsv[i] = np.sum(np.abs(self.scene_image[i] - self.scene_prev[i])) / float(num_pixels)
            #self.scene_change_log += '%6.2f ' % (delta_hsv[i])
        delta_hsv_avg = sum(delta_hsv) / 3.0
        self.scene_change_log += 'scene-diff: %6.2f' % (delta_hsv_avg)
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
            print("{}: Skip this scene is irrelevant: {}".format(self.get_time_pos(), self.scene_change_log))
            if self.count % 5 == 0:
                sample = cv2.cvtColor(self.sample, cv2.COLOR_RGB2BGR)
                cv2.imshow('roi', sample)
            self.fast_forward(1000)
            return

        sample = cv2.cvtColor(self.sample, cv2.COLOR_RGB2BGR)
        cv2.imshow('roi', sample)

        interested_idx = -1
        faces = self.detector(self.sample, 1)
        interested_idx = self.recognize(faces)
        if interested_idx == -1:
            print("{}: Skip the relavant face not found: {}".format(self.get_time_pos(), self.scene_change_log))
            return

        if self.count_in_scene <= sensitive_frames:
            self.scene_keep = True

        face = faces[interested_idx]

        shape = self.predictor(self.sample, face)
        shape_2d = np.array([[left+p.x, top+p.y] for p in shape.parts()])
        display_scale = (np.array(self.display_image.shape) / np.array(self.image.shape))[0:2]
        for s in shape_2d:
            s = np.multiply(s, display_scale).astype(np.int32)
            cv2.circle(self.display_image, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)

        size = max_coords - min_coords
        size[0] /= 1.2
        size[1] /= 1.2
        pos  = np.array([0, 0])

        adj_min = min_coords - size - pos
        adj_max = max_coords + size - pos

        if not self.roi and interested_idx != -1:
            self.roi = (adj_min, adj_max)

        adj_min = tuple(np.multiply(adj_min, display_scale).astype(np.int32))
        adj_max = tuple(np.multiply(adj_max, display_scale).astype(np.int32))
        cv2.rectangle(self.display_image, adj_min, adj_max, (255, 255, 0))
        crop_area = (adj_max[0] - adj_min[0]) * (adj_max[1] - adj_min[1])

        face_pos = (face_left, face_top, face_right, face_bottom) = \
            ( left + face.left(), top + face.top(), left + face.right(), top + face.bottom() )

        display_pos = [ int(p * display_scale[i%2]) for (i, p) in enumerate(face_pos) ]
        cv2.rectangle(self.display_image, tuple(display_pos[0:2]), tuple(display_pos[2:4]), (255,0,0) )

        original_pos = [ int(p * display_scale[i%2]) for (i, p) in enumerate(face_pos) ]
        cv2.rectangle(self.display_image, tuple(display_pos[0:2]), tuple(display_pos[2:4]), (255,0,0) )

        picture_ratio = crop_area / (self.image.shape[0] * self.image.shape[1])
        if picture_ratio > self.full_shot_threshold:
            self.full_shot = True

        print("{}: Area:({},{})+({}x{}) {}, Reco dist: {:6.2f} {} {:6.2f}".format(
            self.get_time_pos(), face_left, face_top, (face_right-face_left), (face_bottom-face_top), 
            self.scene_change_log, self.reco_distance, 
            'FULLSHOT' if self.full_shot else '', picture_ratio )
        )


    def run(self, args):
        print("Picture: ", args.picture_file)
        print("Video  : ", args.video_file)

        if not self.load_person_picture(args.picture_file):
            return

        self.video = cv2.VideoCapture(args.video_file)
        self.scale = args.scale
        self.scene_threashold = args.scene_threashold
        self.reco_tolerance = args.reco
        self.full_shot_threshold = args.fullshot
        self.detect_width = args.detect_width

        self.count = 0
        while True:
            if not self.read_frame():
                break

            self.count += 1
            if self.frames < args.start:
                continue

            if self.scene_changed():
                print("{}: Scene changed.....................................................................................".format(self.count))
                self.roi = None
                self.full_shot = False
                self.scene_keep = False
                self.count_in_scene = 0

            self.count_in_scene += 1
            self.inspect()

            cv2.imshow('Monitor', self.display_image)
            if cv2.waitKey(self.rfps) == ord('q'):
              break


parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, default=1.0, help="Scale factor before processing (default: 1.0)")
parser.add_argument("--log_file", type=str, default=None, help="Face detect log")
parser.add_argument("--scene_threashold", type=float, default=50.0, help="Scene change detection diff-hsv threashold (default: 30.0)")
parser.add_argument("--reco", type=float, default=0.5, help="Face diff tolerence (default: 0.5)")
parser.add_argument("--fullshot", type=float, default=0.15, help="Threshold of fullshot detect,  crop-area / whole-area ratio")
parser.add_argument("--detect_width", type=int, default=320, help="Width to resize for internal processing to detect faces")
parser.add_argument("--start", type=int, default=0, help="Start frame")
parser.add_argument("picture_file", default=None, help="Reference face image file")
parser.add_argument("video_file", default=None, help="Source video file")
args = parser.parse_args()

try:
    trace = FaceTracer()
    trace.run(args)
except KeyboardInterrupt:
    print("\nStop!")
