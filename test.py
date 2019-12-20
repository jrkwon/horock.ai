#!/usr/bin/env python

import os
import sys
import time

sys.path.append( os.path.dirname(sys.argv[0]) + '/Recycle-GAN')

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

class HorockTestOptions(TestOptions):
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
        TestOptions.initialize(self)
        self.update_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. cycle_gan, pix2pix, recycle_gan, test')
        self.update_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.update_argument('--which_model_netG', type=str, default='resnet_6blocks', help='selects model to use for netG')
        self.update_argument('--which_model_netP', type=str, default='unet_256', help='selects model to use for netP')
        self.update_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | unaligned_triplet]')
        self.update_argument('--no_dropout', action='store_true', default=True, help='no dropout for the generator')
        self.update_argument('--dropout', action='store_false', dest='no_dropout')
        self.update_argument('--how_many', type=int, default=sys.maxsize, help='how many test images to run')
        self.update_argument('--results_dir', type=str, default='./results', help='saves results here.')
        self.update_argument('--datasets', required=False, default='./datasets', help='path to hold training images directories')
        self.update_argument('--dataroot', required=False, default='./dataroot', help='path to run GAN')
        self.update_argument('--display_id', required=False, type=int, default='1', help='Display ID (default:1, disabled:0)')
        self.update_argument('nameA', help='Actor person as ./datasets/nameA')
        self.update_argument('nameB', help='Target person as ./datasets/nameB')

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
        self.opt.isTrain = self.isTrain   # train or test

        names = '%s-%s' % (self.opt.nameA, self.opt.nameB)
        self.opt.dataroot = os.path.join(self.opt.dataroot, names)
        self.opt.name = names

        src = os.path.join(self.opt.datasets, self.opt.nameA, 'test', 'images')
        dst = os.path.join(self.opt.dataroot, 'testA')
        self.make_symlink(src, dst)

        src = os.path.join(self.opt.datasets, self.opt.nameB, 'test', 'images')
        dst = os.path.join(self.opt.dataroot, 'testB')
        self.make_symlink(src, dst)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        return self.opt

def run():
    opt = HorockTestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    print(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()

try:
    run()
except KeyboardInterrupt:
    print("Bye bye...")

