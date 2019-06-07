#!/usr/bin/env python

import os
import cv2
import sys
import time
import traceback

sys.path.append( os.path.dirname(sys.argv[0]) + '/Recycle-GAN')

from options.base_options import BaseOptions

class HorockMkvideoOptions(BaseOptions):
    def remove_argument(self, opt):
        action = self.parser._option_string_actions.pop(opt, None)
        if action is not None:
            self.parser._remove_action(action)

    def update_argument(self, opt, **kwargs):
        self.remove_argument(opt)
        if kwargs == {}:
            return

        self.parser.add_argument(opt, **kwargs)

    def initialize(self):
        self.update_argument('--epoch', type=str, default='latest', help='chooses which epoch to build video')
        self.update_argument('--results_dir', type=str, default='./results', help='saved results directory.')
        self.update_argument('--datasets', required=False, default='./datasets', help='path to write output mp4 file')
        self.update_argument('--dataroot', required=False, default='./dataroot', help='path to run GAN')
        self.update_argument('--fps', required=False, type=float, default=30., help='output video frames/sec')
        self.update_argument('nameA', help='Actor person as ./datasets/nameA')
        self.update_argument('nameB', help='Target person as ./datasets/nameB')
        self.update_argument('mode', help='Image sequence mode, fake_A, fake_B, real_A, real_B (You can concatename multiple modes with "+", e.g: real_A+fake_B)')

    def make_symlink(self, source, target):
        source = os.path.abspath(source)
        target = os.path.abspath(target)

        if not os.path.exists(source):
            raise Exception('Directory not exists: %s' % source)

        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)

        source = os.path.relpath(source, start=target_dir)
        try:
            os.unlink(target)
        except:
            pass
        os.symlink(source, target)

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        names = '%s-%s' % (self.opt.nameA, self.opt.nameB)
        self.opt.images_dir = os.path.join(self.opt.results_dir, names, 'test_%s' % self.opt.epoch, 'images')
        self.opt.modes = self.opt.mode.split('+')
        return self.opt

def run():
    opt = HorockMkvideoOptions().parse()
    dupcheck = {}
    images = {}
    for mode in opt.modes:
        images[mode] = []

    for root, _, fnames in os.walk(opt.images_dir):
        #fnames = filter( lambda fname: '_'.join(fname.split('.')[0].split('_')[1:]) in opt.modes, fnames )
        count = 0
        for fname in sorted(fnames):
            (seq, mode, ab, ext) = fname.replace('.', '_').split('_')
            mode = '%s_%s' % (mode, ab)
            if mode not in opt.modes:
                continue
            images[mode].append( os.path.join(root, fname) )
            count += 1

    if count == 0:
        print("No images found:", opt.images_dir, opt.modes)
        return

    if count / len(opt.modes) != len(images[opt.modes[0]]):
        print("Not enough images:", count, len(opt.modes) * len(images[opt.modes[0]]) )
        return

    width = 0
    height = 0
    for mode in opt.modes:
        width += cv2.imread(images[mode][0]).shape[1]
        height += cv2.imread(images[mode][0]).shape[0]

    print("Output size = %dx%d" % (width, height))
    fourcc = 'MP4V'
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    filename = os.path.join(opt.datasets, '%s-%s.%s.mp4' % (opt.nameA, opt.nameB, '_'.join(opt.modes)))
    out = cv2.VideoWriter( filename, fourcc, opt.fps, (width,height))

    if not out.isOpened():
        print("Can't open output file")
        return

    count = count / len(opt.modes)
    i = 0
    while i < count:
        for mode in opt.modes:
            f = images[mode][i]
            print(f, end='\r')
            img = cv2.imread(f)
            out.write(img)
        i += 1
    print('')
    out.release()
    print("Output:", filename)


try:
    run()
except KeyboardInterrupt:
    print("Bye bye...")
except:
    print(traceback.format_exc())

