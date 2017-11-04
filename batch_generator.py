# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import scipy.misc as misc
import utils

color_dict = {'bg':[255, 0, 0, 0], 'road':[255, 0, 255, 1], 'other':[0, 0, 0, 0]}

class Dataset:

    def __init__(self, records_list, logger, batch_size=5, data_dir='data', split_val=True, rate_val=0.3):
        self.logger = logger
        self.logger.info('Dataset Initializing ...')
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.batch_index = 0
        self.records_list = records_list
        self.names = [utils.extract_name(i['img']) for i in records_list]
        self.load_images()

        self.num_train = len(records_list)
        self.names_train = self.names
        self.images_train = self.images
        self.gts_train = self.gts

        self.num_val = None
        self.names_val = None
        self.images_val = None
        self.gts_val = None

        self.shuffled_index = np.random.permutation(self.num_train)

        if split_val:
            self.num_val = math.ceil(len(records_list)*0.3)
            self.names_val = self.names[-self.num_val:]
            self.images_val = self.images[-self.num_val:]
            self.gts_val = self.gts[-self.num_val:]

            self.num_train -= self.num_val
            self.names_train = self.names[:-self.num_val]
            self.images_train = self.images[:-self.num_val]
            self.gts_train = self.gts[:-self.num_val]
            self.shuffled_index = np.random.permutation(self.num_train)


    def load_images(self):
        if os.path.exists(os.path.join(self.data_dir,'images.npy')):
            self.images = np.load(os.path.join(self.data_dir,'images.npy'))
        else:
            self.images = np.array([self.image_enhance(rec['img']) for rec in self.records_list])
            np.save(os.path.join(self.data_dir,'images.npy'), self.images)
        if os.path.exists(os.path.join(self.data_dir,'gts.npy')):
            self.gts = np.load(os.path.join(self.data_dir,'gts.npy'))
        else:
            self.gts = np.array([self.make_exclusive_lables(rec['gt']) for rec in self.records_list])
            np.save(os.path.join(self.data_dir,'gts.npy'), self.gts)
        self.logger.info('Dataset load SUCCESS!')
        self.logger.info('Shape of images: {}.'.format(self.images.shape))
        self.logger.info('Shape of gts: {}.'.format(self.gts.shape))

    def image_enhance(self, image_path):
        img = misc.imread(os.path.join(self.data_dir, image_path))
        img = misc.imresize(img,(384, 1248, 3), interp='nearest')
        return img

    def make_exclusive_lables(self, image_path):
        img = self.image_enhance(image_path)
        img_p = np.zeros([img.shape[0],img.shape[1]])
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                if img[r][c].tolist() == color_dict['road'][:-1]:
                    img_p[r][c] = 1
        return img_p

    def next_batch(self):
        index_start = self.batch_index * self.batch_size
        index_end = index_start + self.batch_size
        if index_end > self.num_train:
            self.shuffled_index = np.random.permutation(self.num_train)
            index_start = 0
            index_end = index_start + self.batch_size
        idxes = self.shuffled_index[index_start:index_end]
        self.batch_index += 1
        return self.images_train[idxes], self.gts_train[idxes], [self.names_train[i] for i in idxes]

    def next_val_batch(self):
        assert self.num_val is not None
        if self.num_val > self.batch_size:
            val_idx = np.random.choice(self.num_val, self.batch_size)
            return self.images_val[val_idx], self.gts_val[val_idx], [self.names_val[i] for i in val_idx]
        else:
            return self.images_val, self.gts_val, self.names_val

    def all_val_images(self):
        return self.images_val, self.gts_val, self.names_val
