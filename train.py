#!/usr/bin/env python

import os
import sys
import time
import torch
import argparse
import threading

sys.path.append( os.path.dirname(sys.argv[0]) + '/Recycle-GAN')

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import util

class HorockTrainOptions(TrainOptions):
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
        TrainOptions.initialize(self)
        self.update_argument('--model', type=str, default='recycle_gan', help='chooses which model to use. cycle_gan, pix2pix, recycle_gan, test')
        self.update_argument('--which_model_netG', type=str, default='resnet_6blocks', help='selects model to use for netG')
        self.update_argument('--which_model_netP', type=str, default='unet_256', help='selects model to use for netP')
        self.update_argument('--dataset_mode', type=str, default='unaligned_triplet', help='chooses how datasets are loaded. [unaligned | aligned | single | unaligned_triplet]')
        self.update_argument('--no_dropout', action='store_true', default=True, help='no dropout for the generator')
        self.update_argument('--identity', type=float, default=0., help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.update_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.update_argument('--datasets', required=False, default='./datasets', help='path to hold training images directories')
        self.update_argument('--dataroot', required=False, default='./dataroot', help='path to run GAN')
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
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        names = '%s-%s' % (self.opt.nameA, self.opt.nameB)
        self.opt.dataroot = os.path.join(self.opt.dataroot, names)
        self.opt.name = names

        src = os.path.join(self.opt.datasets, self.opt.nameA, 'train', 'images')
        dst = os.path.join(self.opt.dataroot, 'trainA')
        self.make_symlink(src, dst)

        src = os.path.join(self.opt.datasets, self.opt.nameB, 'train', 'images')
        dst = os.path.join(self.opt.dataroot, 'trainB')
        self.make_symlink(src, dst)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.expr_dir = expr_dir
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

def run():
    options = HorockTrainOptions()
    opt = options.parse()

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    print("Save directory", model.save_dir)
    print("Display id", opt.display_id)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

try:
    run()
except KeyboardInterrupt:
    print("Bye bye...")

