#!/usr/bin/env python3

import cv2, dlib, sys, os
import numpy as np
import argparse
import glob
import face_recognition.api as face_recognition
import traceback

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

def _same_rect(rect1, rect2):
    return rect1[0] == rect2[0] and rect1[1] == rect2[1]

class FaceTracer:

    def __init__(self):
        self.person_name = None
        self.person_picture = None

        self.video = None

        # Video time stamp
        self.time_pos = 0.
        self.frame_num = 0

        # Video frame history (recent 5 frames?)
        self.history = []

        self.rfps = 1

        # Monitoring image scale
        self.scale = 1.0

        # Face detection parameter
        self.image = None
        self.detect_width = 320
        self.detect_scale = None
        self.detect2display_scale = None

        # Face detection hotspot
        self.roi = None
        self.sample = None
        self.sample_pos = (0, 0) # Sample position on detection image (self.image)
        self.face_pos = None

        # Output cropping area
        self.detect_output = None
        self.output = None
        self.output_frame = None

        # Face detector
        self.detector = face_recognition.face_detector

        # Face pose predictor and comparing threshold
        self.predictor = face_recognition.pose_predictor_68_point
        self.reco_tolerance = 0.5

        # Scene change comparison images
        self.scene_prev = None
        self.scene_image = None

        # Scene change detection log
        self.scene_change_log = ''

        # Cloth change detection
        self.cloth = None

        # So, this scene is relavant?
        self.scene_keep = False

        # Read count
        self.count = 0

        # Read count in a scene
        self.count_in_scene = 0

        # Big face?
        self.full_shot = False
        self.full_shot_threshold = .15
        self.overlap_threshold = 0.
        self.picture_ratio = 0.
        self.overlap_ratio = 0.

        self.output_triplets = []
        self.output_last_frame = -1
        self.output_last_size = ''
        self.output_count = 0
        self.output_dir = './output'
        self.output_width  = 256
        self.output_height = 256

    def log(self, fmt, *args):
        time_pos = "{} {:05d} {:02d}:{:02d}.{:03d}".format(self.count, self.frame_num, (self.time_pos // 1000) // 60, (self.time_pos // 1000) % 60, self.time_pos % 1000)
        fmt = '{}: {}'.format(time_pos, fmt)
        print(fmt.format(*args))

    def load_person_picture(self, file_path):
        basename = os.path.splitext(os.path.basename(file_path))[0]
        img = face_recognition.load_image_file(file_path)
        faces = self.detector(img, 1)
        encodings = face_recognition.face_encodings(img, [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in faces])
        if len(encodings) > 1:
            print("Error: The picture file has  more than one faces: {}", file_path)
            return False
        elif len(encodings) == 0:
            print("Error: No face is found: {}", file_path)
            return False

        self.person_name    = basename
        self.person_picture = encodings[0]

        return True

    def crop_cloth(self, image, sample_pos, face):
        (left, top) = sample_pos
        (face_left, face_top, face_right, face_bottom) = \
            ( left + face.left(), top + face.top(), left + face.right(), top + face.bottom() )
        height = face_bottom - face_top
        return image[face_bottom+height//2:face_bottom+height, face_left:face_right]

    def read_frame(self):
        self.time_pos = int(self.video.get(cv2.CAP_PROP_POS_MSEC))
        self.frame_num = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))

        #Original image: self.original_image (BGR)
        ret, self.original_image = self.video.read()
        if not ret:
            return ret

        #self.log('Origin Size: {}x{}px', self.video.get( cv2.CAP_PROP_FRAME_WIDTH ), self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) )
        original_width = self.original_image.shape[1]

        if len(self.history) > 5:
            self.history.pop(0)
        self.history.append( { 'image': self.original_image, 'output_area': None, 'full_shot': False })

        #Monitoring image: self.display_image (BGR)
        # self.scale = display_width / original_width
        # detect2display scale = display_width / detect_width = self.detect_scale / self.scale
        self.display_image = cv2.resize(self.original_image, (0,0), 0, self.scale, self.scale )
        display_width = self.display_image.shape[1]

        # self.detect_scale = detect_width / original_width
        if not self.detect_scale:
            self.detect_scale = self.detect_width / original_width

        #Face detection internal image: self.image (RGB)
        self.image = cv2.resize(self.original_image, (0,0), 0, self.detect_scale, self.detect_scale )
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        if not self.detect2display_scale:
            self.detect2display_scale = display_width / self.detect_width

        #Secene change detection image: self.scene_image (HSV)
        self.scene_prev = self.scene_image
        scale = 64. / self.image.shape[1]
        self.scene_image = cv2.resize(self.image, (0,0), 0, scale, scale)
        self.scene_image = cv2.cvtColor(self.scene_image, cv2.COLOR_RGB2HSV, cv2.INTER_LANCZOS4)
        return ret

    def fast_forward(self, msec):
        self.log("Fast forward {} msecs", msec)
        self.video.set(cv2.CAP_PROP_POS_MSEC, float(self.time_pos + msec))

    def init_scene(self):
        self.roi = None
        self.full_shot = False
        self.scene_keep = False
        self.count_in_scene = 0
        self.detect_output = None
        self.face_pos = None
        self.output = None

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
        self.scene_change_log += 'Scene-diff: %.2f' % (delta_hsv_avg)
        if delta_hsv_avg > self.scene_threshold:
            self.scene_change_log += '(scene_threshold: >%.2f)' % ( self.scene_threshold )
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

    def pick_roi_sample(self):
        if self.roi:
            #self.log("ROI {} {}", self.roi, self.image.shape)
            self.sample = self.image[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
            self.sample_pos = (self.roi[0][0], self.roi[0][1])
        else:
            #self.log("IMAGE: {}", self.image.shape)
            self.sample = self.image
            self.sample_pos = (0, 0)

    def calculate_roi(self, shape_2d):
        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)

        size = max_coords - min_coords
        size[0] /= 1.2
        size[1] /= 1.2
        pos  = np.array([0, 0])
        adj_min = min_coords - size - pos
        adj_max = max_coords + size - pos

        adj_min = np.amax( [adj_min, (0,0)], 0 )
        adj_max = np.amin( [adj_max, np.flip(np.array(self.image.shape[0:2]))], 0 )

        self.roi = (adj_min, adj_max)

    def calculate_full_shot(self):
        (adj_min, adj_max) = self.roi
        crop_size = (adj_max[0] - adj_min[0]) * (adj_max[1] - adj_min[1])

        # Picture portion ratio = current-crop / whole-picture
        self.picture_ratio = crop_size / (self.image.shape[0] * self.image.shape[1])

        if self.picture_ratio > self.full_shot_threshold:
            self.full_shot = True

        self.history[-1]['full_shot'] = self.full_shot

    def calculate_overlap_ratio(self, area1, area2):
        (area1_min, area1_max) = area1
        area1_size = (area1_max[0] - area1_min[0]) * (area1_max[1] - area1_min[1])

        (area2_min, area2_max) = area2
        overlap_min = ( max(area2_min[0], area1_min[0]), max(area2_min[1], area1_min[1]) )
        overlap_max = ( min(area2_max[0], area1_max[0]), min(area2_max[1], area1_max[1]) )

        # Calculation ratio of prev-current-overlapped-area / current-crop
        overlap_ratio  = (overlap_max[0] - overlap_min[0]) * (overlap_max[1] - overlap_min[1])
        overlap_ratio /= area1_size
        return overlap_ratio

    def update_output_area(self):
        overlap_ratio = 0.

        (adj_min, adj_max) = self.roi
        crop_size = (adj_max[0] - adj_min[0]) * (adj_max[1] - adj_min[1])

        # Recalculate output image cropping area
        if self.detect_output:
            overlap_ratio = self.calculate_overlap_ratio(self.roi, self.detect_output)

        if overlap_ratio < self.overlap_threshold:
            self.detect_output = [ e.copy() for e in self.roi ]
            self.output = np.divide( self.detect_output, self.detect_scale ).astype(np.int32)
            self.output_frame = self.frame_num
            size = self.output[1] - self.output[0]
            if size[0] < size[1]: # width < height
                # Horizontal: Enlarge the width to the same size of height keeping center align
                self.output[0][0] += int((size[0] - size[1])/2)
                self.output[1][0] = self.output[0][0] + size[1]
            else:
                # Vertical: Enlarge the height to the same size of width keeping bottom align
                self.output[0][1] -= size[0] - size[1]

        self.history[-1]['output_area'] = self.output
        self.overlap_ratio = overlap_ratio

    def inspect(self):
        LEADING_FRAMES = 3

        self.pick_roi_sample()

        if self.count_in_scene > LEADING_FRAMES and not self.scene_keep:
            self.log("Skip, this scene is irrelevant: {}", self.scene_change_log)
            self.fast_forward(1000)
            return False

        faces = self.detector(self.sample, 1)
        interested_idx = self.recognize(faces)
        if interested_idx == -1:
            self.log("Skip, the relavant face not found: {}", self.scene_change_log)
            return False

        if self.count_in_scene <= LEADING_FRAMES:
            self.scene_keep = True

        face = faces[interested_idx]

        (left, top) = self.sample_pos
        self.face_pos = (face_left, face_top, face_right, face_bottom) = \
            ( left + face.left(), top + face.top(), left + face.right(), top + face.bottom() )

        shape = self.predictor(self.sample, face)
        shape_2d = np.array([[left+p.x, top+p.y] for p in shape.parts()])

        self.calculate_roi(shape_2d)
        self.calculate_full_shot()
        self.update_output_area()

        ## For display ........
        for s in shape_2d:
            s = np.multiply(s, self.detect2display_scale).astype(np.int32)
            cv2.circle(self.display_image, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        adj_min = tuple(np.multiply(self.roi[0], self.detect2display_scale).astype(np.int32))
        adj_max = tuple(np.multiply(self.roi[1], self.detect2display_scale).astype(np.int32))
        cv2.rectangle(self.display_image, adj_min, adj_max, (255, 255, 0))

        display_pos = [ int(p * self.detect2display_scale) for (i, p) in enumerate(self.face_pos) ]
        cv2.rectangle(self.display_image, tuple(display_pos[0:2]), tuple(display_pos[2:4]), (255,0,0) )

        if self.output is not None:
            output = np.multiply(self.output, self.scale ).astype(np.int32)
            cv2.rectangle(self.display_image, tuple(output[0]), tuple(output[1]), (0,0,255) )

        return True

    def write_frame(self):
        self.display_output = None
        if self.output is None:
            return

        if not self.full_shot:
            return

        output = self.original_image[self.output[0][1]:self.output[1][1], self.output[0][0]:self.output[1][0]]
        output_size = '%dx%d' % ( output.shape[1], output.shape[0] )

        if self.frame_num != self.output_last_frame + 1:
            self.output_triplets = []

        if self.output_last_size != output_size:
            self.output_triplets = []

        if len(self.output_triplets) == 3:
            self.output_triplets.pop(0)

        self.display_output = cv2.resize(output, (self.output_width,self.output_height))
        self.output_triplets.append(self.display_output)
        self.output_last_frame = self.frame_num
        self.output_last_size = output_size

        if len(self.output_triplets) != 3:
            return

        filename = "%s/%06d.png" % (self.output_dir, self.output_count)
        self.output_count += 1
        output_img = np.concatenate(self.output_triplets, axis=1)
        cv2.imwrite(filename, output_img)
        cv2.imshow('Triplet', output_img)

    def run(self, args):
        print("Picture: ", args.picture_file)
        print("Video  : ", args.video_file)

        if not self.load_person_picture(args.picture_file):
            return

        self.video = cv2.VideoCapture(args.video_file)
        if not self.video.isOpened():
            print("Can't open video source")
            return

        self.scale = args.scale
        self.scene_threshold = args.scene_threshold
        self.overlap_threshold = args.overlap
        self.reco_tolerance = args.reco
        self.full_shot_threshold = args.fullshot
        self.detect_width = args.detect_width
        self.output_dir = args.output_dir
        if 'x' not in args.output_size:
            print("Invalid size ('x' should be exists in size)")
            return
        (self.output_width, self.output_height) = [ int(v) for v in args.output_size.split('x') ]
        print('Scale:', self.scale)
        print('Scene threshold:', self.scene_threshold)
        print('Overlap threshold:', self.overlap_threshold)
        print('Reco tolerance:', self.reco_tolerance)
        print('Full-shot threshold:', self.full_shot_threshold)
        print('Internal detect width:', self.detect_width)
        print('Output dir:', self.output_dir)

        self.video.set( cv2.CAP_PROP_POS_FRAMES, args.begin )

        while True:
            if not self.read_frame():
                break

            if 0 <= args.end < self.frame_num:
                break

            self.count += 1

            if self.scene_changed():
                self.log("{0}: Scene changed {1}", self.count, str('.') * 80)
                self.init_scene()

            self.count_in_scene += 1

            do_inspect = False
            throttling = False
            if args.throttling > 0 and self.scene_keep and self.count_in_scene > 50:
                if self.count_in_scene % args.throttling == 0:
                    do_inspect = True
                else:
                    throttling = True
            else:
                do_inspect = True

            if do_inspect:
                self.inspect()

            if self.count % 5 == 0:
                sample = cv2.cvtColor(self.sample, cv2.COLOR_RGB2BGR)
                cv2.imshow('roi', sample)

            cv2.imshow('Monitor', self.display_image)
            if self.output is not None:
                self.write_frame()
                cv2.imshow('Output', self.display_output)

            if self.face_pos is not None:
                (face_left, face_top, face_right, face_bottom) = self.face_pos
                self.log("Area:({},{})+({}x{}) (org:{}x{}) {}, Reco-dist: {:.2f} Fullshot: {:.1f}%({})  Overlap: {:.1f}% {}",
                    face_left, face_top, (face_right-face_left), (face_bottom-face_top), 
                    self.output[1][0] - self.output[0][0], self.output[1][1] - self.output[0][1],
                    self.scene_change_log, self.reco_distance, 
                    (100. * self.picture_ratio),
                    'Y' if self.full_shot else 'N',
                    (100. * self.overlap_ratio),
                    '(throttling)' if throttling else ''
                )

            if cv2.waitKey(self.rfps) == ord('q'):
              break


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scale", type=float, default=1.0, help="Scale factor before processing (default: 1.0)")
parser.add_argument("-l", "--log_file", type=str, default=None, help="Face detect log")
parser.add_argument("-c", "--scene_threshold", type=float, default=80.0, help="Scene change detection diff-hsv threshold (default: 80.0)")
parser.add_argument("-r", "--reco", type=float, default=0.5, help="Face diff tolerence (default: 0.5)")
parser.add_argument("-f", "--fullshot", type=float, default=0.15, help="Threshold of fullshot detect,  crop-area / whole-area ratio")
parser.add_argument("-w", "--detect_width", type=int, default=320, help="Width to resize for internal processing to detect faces")
parser.add_argument("-L", "--overlap", type=float, default=.8, help="Threshold of changing new cropping area, (overlap of prev & curr)/curr")
parser.add_argument("-t", "--throttling", type=int, default=0, help="In same scene, skip inspection next to THROTTLING frames")
parser.add_argument("-b", "--begin", "--start", dest="begin", type=int, default=0, help="Start frame")
parser.add_argument("-e", "--end", type=int, default=-1, help="End frame")
parser.add_argument("-o", "--output", dest="output_dir", type=str, default="./output", help="Output directory")
parser.add_argument("-x", "--output_size", dest="output_size", type=str, default="256x256", help="Output W x H size")
parser.add_argument("picture_file", default=None, help="Reference face image file")
parser.add_argument("video_file", default=None, help="Source video file")
args = parser.parse_args()

try:
    trace = FaceTracer()
    trace.run(args)
except KeyboardInterrupt:
    print("\nStop!")
