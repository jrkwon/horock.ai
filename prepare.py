#!/usr/bin/env python3

import cv2, dlib, sys, os
import numpy as np
import math
import argparse
import glob
import face_recognition.api as face_recognition
import traceback
import re

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

        self.rfps = 1

        # Monitoring image scale
        self.scale = 1.0

        # Face detection parameter
        self.image = None
        self.display_image = None
        self.original_image = None
        self.detect_width = 320
        self.detect_scale = None
        self.detect2display_scale = None
        self.heat_map = np.zeros((256,256), np.uint8)

        # Face detection hotspot
        self.roi = None
        self.sample = None
        self.sample_pos = (0, 0) # Sample position on detection image (self.image)
        self.face_pos = None

        # Output cropping area
        self.detect_output = None
        self.output = None
        self.output_frame = None
        self.output_image = None

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
        self.skip_in_scene = 0

        # Big face?
        self.full_shot = False
        self.full_shot_threshold = .15
        self.overlap_threshold = 0.
        self.picture_ratio = 0.
        self.overlap_ratio = 0.

        self.roi_history = []
        self.output_triplets = []
        self.output_last_frame = -1
        self.output_last_size = ''
        self.output_count = {}
        self.output_dir = './output'
        self.output_width  = 256
        self.output_height = 256

        self.bgformula = None
        self.chroma = (255,255,255)

    def log(self, fmt, *args):
        time_pos = "{:05d}/{:05d} {:02d}:{:02d}.{:03d}".format(self.frame_num, self.total_frames, (self.time_pos // 1000) // 60, (self.time_pos // 1000) % 60, self.time_pos % 1000)
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

        self.resize_images()
        return ret

    def resize_images(self):

        #self.log('Origin Size: {}x{}px', self.video.get( cv2.CAP_PROP_FRAME_WIDTH ), self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) )
        original_width = self.original_image.shape[1]

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

    def fast_forward(self, msec):
        self.log("Fast forward {} msecs", msec)
        self.video.set(cv2.CAP_PROP_POS_MSEC, float(self.time_pos + msec))

    def init_scene(self):
        self.roi = None
        self.full_shot = False
        self.scene_keep = False
        self.count_in_scene = 0
        self.skip_in_scene = 0
        self.detect_output = None
        self.face_pos = None
        self.output = None
        self.output_image = None

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
        image_height = self.image.shape[0]
        roi_height = size[1]
        factor = min( 0.8, (image_height - roi_height)/(2 * roi_height) )
        size = np.multiply(size, factor).astype(np.int32)
        pos  = np.array([0, size[1] * .25]).astype(np.int32)
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

    def calculate_overlap_ratio(self, area1, area2):
        (area1_min, area1_max) = area1
        area1_size = (area1_max[0] - area1_min[0]) * (area1_max[1] - area1_min[1])

        (area2_min, area2_max) = area2
        area2_size = (area2_max[0] - area2_min[0]) * (area2_max[1] - area2_min[1])

        overlap_min = ( max(area2_min[0], area1_min[0]), max(area2_min[1], area1_min[1]) )
        overlap_max = ( min(area2_max[0], area1_max[0]), min(area2_max[1], area1_max[1]) )

        # Calculation ratio of prev-current-overlapped-area / current-crop and prev-crop
        overlap_size  = (overlap_max[0] - overlap_min[0]) * (overlap_max[1] - overlap_min[1])
        overlap_ratio = min( overlap_size / area1_size, overlap_size / area2_size )
        return overlap_ratio

    def update_output_area(self):
        overlap_ratio = 0.

        (adj_min, adj_max) = self.roi
        crop_size = (adj_max[0] - adj_min[0]) * (adj_max[1] - adj_min[1])

        # Recalculate output image cropping area
        if self.detect_output:
            overlap_ratio = self.calculate_overlap_ratio(self.roi, self.detect_output)

        if overlap_ratio < self.overlap_threshold:
            self.log('ROI prev-current overlap < threhold: {:.2f} < {:.2f}', overlap_ratio, self.overlap_threshold)
            self.detect_output = None
            for roi in self.roi_history:
                olap = self.calculate_overlap_ratio(self.roi, roi)
                self.log('ROI History: roi:{},{}+{}x{} : {},{}+{}x{} ({:.2f})',
                    self.roi[0][0], self.roi[0][1], self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1],
                    roi[0][0], roi[0][1], roi[1][0] - roi[0][0], roi[1][1] - roi[0][1],
                    olap)
                if olap < self.overlap_threshold:
                    continue
                if overlap_ratio < olap:
                    self.detect_output = roi
                    overlap_ratio = olap
            if self.detect_output is None:
                self.detect_output = [ e.copy() for e in self.roi ]
                self.roi_history.append(self.detect_output)
            self.output = np.divide( self.detect_output, self.detect_scale ).astype(np.int32)
            self.output_frame = self.frame_num
            size = self.output[1] - self.output[0]
            if size[0] < size[1]: # width < height
                # Horizontal: Enlarge the width to the same size of height keeping center align
                self.output[0][0] += int((size[0] - size[1])/2)
                self.output[1][0] = self.output[0][0] + size[1]
            else:
                # Vertical: Enlarge the height to the same size of width keeping bottom align
                diff = min(size[0] - size[1], self.output[0][1])
                self.output[0][1] -= min(size[0] - size[1], self.output[0][1])
                self.output[1][1] += (size[0] - size[1]) - diff
            self.log('ROI Select: roi:{},{}+{}x{} -> org:{},{}+{}x{}',
                self.detect_output[0][0], self.detect_output[0][1],
                self.detect_output[1][0] - self.detect_output[0][0], self.detect_output[1][1] - self.detect_output[0][1],
                self.output[0][0], self.output[0][1],
                self.output[1][0] - self.output[0][0], self.output[1][1] - self.output[0][1] )

        self.overlap_ratio = overlap_ratio
        self.output_image = self.original_image[self.output[0][1]:self.output[1][1], self.output[0][0]:self.output[1][0]]
        self.display_output = cv2.resize(self.output_image, (self.output_width,self.output_height))

    def inspect(self):
        LEADING_FRAMES = 3

        self.pick_roi_sample()

        if self.count_in_scene > LEADING_FRAMES and not self.scene_keep:
            self.skip_in_scene += 1
            self.log("Skip, this scene is irrelevant: {}", self.scene_change_log)
            self.fast_forward(1000)
            return False

        faces = self.detector(self.sample, 1)
        interested_idx = self.recognize(faces)
        if interested_idx == -1:
            self.skip_in_scene += 1
            self.log("Skip, the relavant face not found: {}", self.scene_change_log)
            if self.skip_in_scene > 30:
                self.fast_forward(1000)
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
        landmarks = []
        for s in shape_2d:
            s = np.multiply(s, self.detect2display_scale).astype(np.int32)
            landmarks.append(s)
            cv2.circle(self.display_image, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        (angle, ratio, leye, reye, mouth) = self.calculate_rotation(landmarks, True)
        byte_angle = 128 + int(127 * angle / 90)
        byte_ratio = 128 + int(127 * ratio / 100)
        cv2.circle(self.heat_map, center=tuple([byte_angle, byte_ratio]), color=(255,255,255), radius=1, thickness=2, lineType=cv2.LINE_AA)
        self.log('Angle:{:.1f}({}) Ratio:{:.1f}({})', angle, byte_angle, ratio, byte_ratio)

        adj_min = tuple(np.multiply(self.roi[0], self.detect2display_scale).astype(np.int32))
        adj_max = tuple(np.multiply(self.roi[1], self.detect2display_scale).astype(np.int32))
        cv2.rectangle(self.display_image, adj_min, adj_max, (255, 255, 0))

        display_pos = [ int(p * self.detect2display_scale) for (i, p) in enumerate(self.face_pos) ]
        cv2.rectangle(self.display_image, tuple(display_pos[0:2]), tuple(display_pos[2:4]), (255,0,0) )

        if self.output is not None:
            output = np.multiply(self.output, self.scale ).astype(np.int32)
            cv2.rectangle(self.display_image, tuple(output[0]), tuple(output[1]), (0,0,255) )

        return True

    def erase_background(self):
        if not self.full_shot:
            return False
        if not self.display_output is not None:
            return False

        if not self.bgformula:
            return False

        hsv = None
        rgb = None
        channels = {}
        for f in self.bgformula:
            (ch, compare, val) = f
            if ch in ('R', 'G', 'B'):
                if rgb is None:
                    rgb = cv2.cvtColor(self.display_output, cv2.COLOR_BGR2RGB)
                    rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
                    red, green, blue = cv2.split(rgb)
                    channels['R'] = red
                    channels['G'] = green
                    channels['B'] = blue
                    rgb = np.concatenate([red, green, blue], axis=1)
                    if not self.hide_display:
                        cv2.imshow('RGB', rgb)
            elif ch in ('H', 'S', 'V'):
                if hsv is None:
                    hsv = cv2.cvtColor(self.display_output, cv2.COLOR_BGR2HSV)
                    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
                    hue, sat, gray = cv2.split(hsv)
                    channels['H'] = hue
                    channels['S'] = sat
                    channels['V'] = gray
                    hue = cv2.cvtColor(hue.copy(), cv2.COLOR_GRAY2BGR)
                    sat = cv2.cvtColor(sat.copy(), cv2.COLOR_GRAY2BGR)
                    gray = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
                    hsv = np.concatenate([hue, sat, gray], axis=1)
                    if not self.hide_display:
                        cv2.imshow('HSV', hsv)
            else:
                raise Exception('Invalid channel', ch)

        mask = None
        for f in self.bgformula:
            (ch, compare, val) = f
            thresh = cv2.threshold(channels[ch], val, 255, cv2.THRESH_BINARY if compare == '>' else cv2.THRESH_BINARY_INV)
            thresh = thresh[1]
            if self.erode_dilate:
                thresh = cv2.erode(thresh, None, iterations=self.erode_dilate)
                thresh = cv2.dilate(thresh, None, iterations=self.erode_dilate)
            mask = cv2.bitwise_and(thresh, thresh, mask = mask) if mask is None else thresh
        if not self.hide_display:
            cv2.imshow('Mask', mask)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0 if len(cnts) == 2 else 1]
        if len(cnts) == 0:
            return
        c = max(cnts, key=cv2.contourArea)
        cv2.fillPoly(self.display_output,pts=[c], color=self.chroma)

    def write_frame(self, mode, frame_max):
        if not self.full_shot:
            return False

        output_size = '%dx%d' % ( self.output_image.shape[1], self.output_image.shape[0] )

        if self.frame_num != self.output_last_frame + 1:
            self.output_triplets = []

        if self.output_last_size != output_size:
            self.output_triplets = []

        if len(self.output_triplets) == 3:
            self.output_triplets.pop(0)

        self.output_triplets.append(self.display_output)
        self.output_last_frame = self.frame_num
        self.output_last_size = output_size

        if len(self.output_triplets) != 3:
            return False

        if output_size not in self.output_count:
            try:
                os.mkdir("%s/%s" % (self.output_dir, output_size))
            except FileExistsError:
                pass
            self.output_count[output_size] = 0

        if 0 < frame_max <= self.output_count[output_size]:
            self.log('Maximum count reached: {}', frame_max)
            return True

        filename = "%s/%s/%06d.png" % (self.output_dir, output_size, self.output_count[output_size])
        self.output_count[output_size] += 1
        if mode == 'train':
            output_img = np.concatenate(self.output_triplets, axis=1)
        elif mode == 'test':
            output_img = self.display_output
        else:
            return False
        cv2.imwrite(filename, output_img)
        output_img = cv2.resize(output_img, (0,0), 0, self.output_scale, self.output_scale)
        cv2.imshow('Write', output_img)
        return False

    def init_variables(self, args):

        def fix_dataset_path(p, *exts):
            if os.path.exists(p):
                return p
            for ext in exts:
                newpath = os.path.join(args.dataset_dir, p + ext)
                print(newpath)
                if os.path.exists(newpath):
                    return newpath
            return None

        args.picture_file = fix_dataset_path(args.picture_file or args.name, '.png', '.jpg')
        args.video_file   = fix_dataset_path(args.video_file or args.name, '.mp4', '.mkv', '.avi', '.mov', 'webm')

        #Output related
        if args.output_dir is None:
            subdir = args.mode
            if args.mode in ('heat', ):
                subdir = 'train'
            self.output_dir = os.path.join(args.dataset_dir, args.name, subdir)

            if args.mode.startswith('test'):
                test_dir = os.path.join(args.dataset_dir, args.name, 'test')
                try:
                    os.unlink(test_dir)
                except:
                    pass
                if os.path.exists(test_dir):
                    raise Exception("Remove test dir first: %s" % test_dir)
                else:
                    output_dir = os.path.relpath(self.output_dir, start=os.path.dirname(self.output_dir))
                    os.symlink(output_dir, test_dir)
        else:
            self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if 'x' not in args.output_size:
            print("Invalid size ('x' should be exists in size)")
            return False

        self.scale = args.scale
        self.scene_threshold = args.scene_threshold

        #Face Detecting
        self.overlap_threshold = args.overlap
        self.reco_tolerance = args.reco
        self.full_shot_threshold = args.fullshot
        self.detect_width = args.detect_width

        #Erase background
        if args.bg:
            self.bgformula = []
            for f in args.bg.split(','):
                self.bgformula.append( [ f[0], f[1], int(f[2:]) ] )

        if re.match( '[0-9a-fA-F]{6}', args.chroma):
            self.chroma = ( int(args.chroma[4:6], 16), int(args.chroma[2:4], 16), int(args.chroma[0:2], 16) )
        else:
            print("Invalid chroma color format")

        self.erode_dilate = args.erode_dilate

        (self.output_width, self.output_height) = [ int(v) for v in args.output_size.split('x') ]
        self.output_scale = args.output_scale
        self.hide_display = args.hide_display

        self.video = cv2.VideoCapture(args.video_file)
        if not self.video.isOpened():
            print("Can't open video source")
            return False

        #Read first frame
        ret, firstframe = self.video.read()

        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video.set( cv2.CAP_PROP_POS_FRAMES, args.begin )

        print("Name: ", args.name)
        print("Mode: ", args.mode)
        print("Picture: ", args.picture_file)
        print("Video  : ", args.video_file)
        print('Scale:', self.scale)
        print('Scene threshold:', self.scene_threshold)
        print('Overlap threshold:', self.overlap_threshold)
        print('Reco tolerance:', self.reco_tolerance)
        print('Full-shot threshold:', self.full_shot_threshold)
        print('Internal detect width:', self.detect_width)
        print('Output dir:', self.output_dir)
        print('Background Detection:', args.bg)
        print('Background chroma color:', args.chroma)

        original_width  = firstframe.shape[1]
        original_height = firstframe.shape[0]
        detect_scale = self.detect_width / original_width
        print('Video size: %sx%s' % (original_width, original_height))
        print('Internal detect size: %sx%s (scale:%s)' % (self.detect_width, int(original_height * detect_scale), detect_scale))

        # Init heat map
        cv2.line(self.heat_map, tuple([0,128]), tuple([255,128]), (0,0,255), 1)
        cv2.line(self.heat_map, tuple([128,0]), tuple([128,255]), (0,0,255), 1)
        if not args.picture_file or not self.load_person_picture(args.picture_file):
            return False

        return True

    def make_images_symlink(self, args):
        maxlength = 0
        maxlength_dir = ''
        for root, _, fnames in os.walk(self.output_dir):
            dname = os.path.basename(root)
            if dname == 'images':
                continue
            files = len(fnames)
            print("Images count:", dname, files)
            if maxlength < files:
                maxlength = files
                maxlength_dir = root

        if maxlength == 0:
            return
        source = os.path.relpath(maxlength_dir, start=self.output_dir)
        target = os.path.join(self.output_dir, 'images')
        try:
            os.unlink(target)
        except:
            pass
        os.symlink(source, target)
        print('Symlink created', maxlength_dir, '->', target)

    def run(self, args):
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
                self.erase_background()
                if self.write_frame(args.mode, args.max):
                    break
                #cv2.imshow('Output', self.display_output)


            if self.face_pos is not None:
                (face_left, face_top, face_right, face_bottom) = self.face_pos
                self.log("face:{},{}+{}x{} roi:{},{}+{}x{} org:{},{}+{}x{} {}, Reco-dist: {:.2f} Fullshot: {:.1f}%({})  Overlap: {:.2f} {}",
                    face_left, face_top, (face_right-face_left), (face_bottom-face_top), 
                    self.roi[0][0], self.roi[0][1], self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1],
                    self.output[0][0], self.output[0][1], self.output[1][0] - self.output[0][0], self.output[1][1] - self.output[0][1],
                    self.scene_change_log, self.reco_distance, 
                    (100. * self.picture_ratio),
                    'Y' if self.full_shot else 'N',
                    self.overlap_ratio,
                    '(throttling)' if throttling else ''
                )

            cv2.imshow('Heat Map', self.heat_map)
            if cv2.waitKey(self.rfps) == ord('q'):
              break

    def calculate_rotation(self, shape, showlines=False):

        def distance(dot1, dot2):
            return int( math.sqrt( (dot1[0] - dot2[0]) * (dot1[0] - dot2[0]) + (dot1[1] - dot2[1]) * (dot1[1] - dot2[1]) ) )

        def atan(dot1, dot2):
            return int(10. * math.atan( (dot2[1] - dot1[1]) / (dot2[0] - dot1[0]) ) * 180. / math.pi) / 10.

        L = shape[36]
        M = shape[27]
        R = shape[45]
        angle1 = atan(L, R)
        face_l = distance(L, M)
        face_r = distance(R, M)

        face_h = (face_r + face_l)
        ratio1 = (face_r - face_l) * 100. / face_h
        ratio1 = int(ratio1 * 10) / 10

        leye  = atan(shape[40]-shape[41], shape[37]-shape[41])
        reye  = atan(shape[46]-shape[47], shape[43]-shape[47])
        mouth = atan(shape[63]-shape[65], shape[65]-shape[67])

        if showlines:
            cv2.line(self.display_image, tuple(L), tuple(R), (0,255,0),1)
            cv2.line(self.display_image, tuple(L), tuple(M), (0,255,255),2)
            cv2.line(self.display_image, tuple(M), tuple(R), (0,255,255),2)
            cv2.line(self.display_image, tuple(shape[40]), tuple(shape[37]), (0,255,255),2)
            cv2.line(self.display_image, tuple(shape[43]), tuple(shape[46]), (0,255,255),2)
            cv2.line(self.display_image, tuple(shape[63]), tuple(shape[67]), (0,255,255),2)
        return (angle1, ratio1, leye, reye, mouth)


    def get_landmarks(self, image):
        faces = self.detector(image, 1)
        if len(faces) != 1:
            return (None, faces)
        shape = self.predictor(image, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return (landmarks, faces)

    def sample_pic(self, args):
        count = 0
        while count < args.samples:
            if cv2.waitKey(self.rfps) == ord('q'):
                break

            ret = self.read_frame()
            if not ret:
                print("End of file")
                break

            filename = "%s/%06d.png" % (self.output_dir, self.frame_num)

            (landmarks, faces) = self.get_landmarks(self.display_image)

            if landmarks is None:
                print( count, "frame", self.frame_num, len(faces))
                cv2.imshow('Sample', self.display_image)
                self.video.set(cv2.CAP_PROP_POS_MSEC, float(self.time_pos + 500))
                continue

            front = False
            leyeopen = False
            reyeopen = False
            mouthopen = False
            (angle, ratio, leye, reye, mouth) = self.calculate_rotation(landmarks, True)

            openness = []
            if abs(angle) < 10 and abs(ratio) < 5:
                front = True
                openness.append('FRONT')
            if abs(leye) > 30:
                leyeopen = True
                openness.append('L-EYE')
            if abs(leye) > 30:
                reyeopen = True
                openness.append('R-EYE')
            if abs(mouth) > 10:
                mouthopen = True
                openness.append('MOUTH')

            print( count, "frame", self.frame_num, len(faces), angle, ratio, ' '.join(openness))

            if front and leyeopen and reyeopen:
                color = (255, 0, 0)
                cv2.imwrite(filename, self.original_image)
            else:
                color = (0, 0, 255)

            cv2.putText(self.display_image, filename, (20,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            description = "LR Nodding(%s), LR Rotate(%s), L-Eye(%s) R-Eye(%s) Mouth(%s)" % (angle, ratio, leye, reye, mouth) 
            cv2.putText(self.display_image, description, (20,40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            description = ' '.join(openness)
            cv2.putText(self.display_image, description, (20,60), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
            for s in landmarks:
                #s = np.multiply(s, self.detect2display_scale).astype(np.int32)
                cv2.circle(self.display_image, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2.imshow('Sample', self.display_image)

            count += 1
            if not front:
                self.video.set(cv2.CAP_PROP_POS_MSEC, float(self.time_pos + 300))

        self.sample_pic_link(args)

    def sample_pic_link(self, args):
        sample = None
        for root, _, fnames in os.walk(self.output_dir):
            sample = os.path.join(root, np.random.choice(fnames))

        if sample is None:
            print("\nPicking random image failed:", self.output_dir)
            return
        source = os.path.relpath(sample, start=args.dataset_dir)
        target = os.path.join(args.dataset_dir, args.name + '.png')
        try:
            os.unlink(target)
        except:
            pass
        os.symlink(source, target)
        print('\nSample picture symlink created', source, '->', target)

    def get_images(self, images_dir):
        images = []
        for root, _, fnames in os.walk(images_dir):
            images += [ os.path.join(root, fname) for fname in sorted(fnames)]
        return images

    def show_heat_map(self, args):
        sample = None
        print(self.output_dir)
        images_dir = os.path.join(self.output_dir, 'images')
        dupcheck = {}
        for image_file in self.get_images(images_dir):
            self.original_image = cv2.imread(image_file)
            height = self.original_image.shape[0]

            #Cut the first rectangle
            self.original_image = self.original_image[:, 0:height, 0:height]
            self.resize_images()

            cv2.imshow('Sample', self.display_image)
            if cv2.waitKey(self.rfps) == ord('q'):
              break
            (landmarks, faces) = self.get_landmarks(self.original_image)
            if landmarks is None:
                continue
            (angle, ratio, leye, reye, mouth) = self.calculate_rotation(landmarks, True)

            byte_angle = 128 + int(127 * angle / 90)
            byte_ratio = 128 + int(127 * ratio / 100)
            cv2.circle(self.heat_map, center=tuple([byte_angle, byte_ratio]), color=(255,255,255), radius=1, thickness=2, lineType=cv2.LINE_AA)
            k = '(%s,%s)' % (byte_angle, byte_ratio)
            if k in dupcheck:
                dupcheck[k] += 1
            else:
                dupcheck[k] = 1
            print(image_file, 'Angle:', angle, 'Ratio:', ratio, byte_angle, byte_ratio, 'Dup:', dupcheck[k])

            cv2.imshow('Heat Map', self.heat_map)
        heatmap_file = os.path.join(self.output_dir, 'heatmap.png')
        cv2.imwrite(heatmap_file, self.heat_map)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scale", type=float, default=1.0, help="Scale factor before processing (default: 1.0)")
parser.add_argument("-l", "--log_file", type=str, default=None, help="Face detect log")
parser.add_argument("-c", "--scene_threshold", type=float, default=80.0, help="Scene change detection diff-hsv threshold (default: 80.0)")
parser.add_argument("-r", "--reco", type=float, default=0.5, help="Face diff tolerence (default: 0.5)")
parser.add_argument("-f", "--fullshot", type=float, default=0.15, help="Threshold of fullshot detect,  crop-area / whole-area ratio")
parser.add_argument("-w", "--detect_width", type=int, default=512, help="Width to resize for internal processing to detect faces")
parser.add_argument("-L", "--overlap", type=float, default=.8, help="Threshold of changing new cropping area, (overlap of prev & curr)/curr")
parser.add_argument("-t", "--throttling", type=int, default=0, help="In same scene, skip inspection next to THROTTLING frames")
parser.add_argument("-b", "--begin", "--start", dest="begin", type=int, default=0, help="Start frame")
parser.add_argument("-e", "--end", type=int, default=-1, help="End frame")
parser.add_argument("-m", "--max", type=int, default=5000, help="Maximum output images")
parser.add_argument("-o", "--output", dest="output_dir", type=str, default=None, help="Output directory (default: <dataset>/<NAME>/<MODE>)")
parser.add_argument("-x", "--output_size", dest="output_size", type=str, default="256x256", help="Output W x H size")
parser.add_argument("-p", "--output_scale", dest="output_scale", type=float, default=1.0, help="Output monitoring scale")
parser.add_argument("-B", "--bg", dest="bg", type=str, default='', help="Background area formula: 'X?n,[X?n[...]]' [X = H,S,V(Grayscale),R,G,B] [? = < >] [n = 0~255]")
parser.add_argument("-E", "--erode", dest="erode_dilate", type=int, default=10, help="Erode/Dilate after finding background area")
parser.add_argument("-D", "--dataset", dest="dataset_dir", type=str, default='datasets', help="Set dataset directory (default: datasets)")
parser.add_argument("-C", "--chroma", dest="chroma", type=str, default='FFFFFF', help="Background filling color (default:FFFFFF; rgb)")
parser.add_argument("-H", "--hide", dest="hide_display", action='store_true', default=False, help="Hide background intermediate images")
parser.add_argument("-a", "--picture", dest="picture_file", default=None, help="Reference face image file: (default: NAME.png)")
parser.add_argument("-v", "--video", dest="video_file", default=None, help="Source video file: (default: NAME.mp4)")
parser.add_argument("-S", "--samples", dest="samples", type=int, default=1000, help="Picture sample count for 'pic' mode (default: 100)")
parser.add_argument("mode", default=None, help="Run mode: train, test, pic (pic for picture sample 1000 images to datasets/NAME/NNNNN.png)")
parser.add_argument("name", default=None, help="Picture label")
args = parser.parse_args()

try:
    face_trace = FaceTracer()
    inited = face_trace.init_variables(args)
    if args.mode == "pic":
        face_trace.sample_pic(args)
    elif args.mode == "heat":
        face_trace.show_heat_map(args)
    else:
        if inited and (args.mode == 'train' or args.mode.startswith('test')):
            face_trace.run(args)
            face_trace.make_images_symlink(args)
        else:
            print("Check your mode (pic, heat, train, test{NNN..}:", args.mode)
except KeyboardInterrupt:
    if args.mode == "train" or args.mode == 'test':
        face_trace.make_images_symlink(args)
    elif args.mode == "pic":
        face_trace.sample_pic_link(args)
    print("\nStop!")
