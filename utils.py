# -*- coding: utf-8 -*-
# import h5py
import os
import numpy as np
import scipy.misc as misc
import logging

class TO_SAVE:
    def __init__(self, best_val_loss_init=10, val_threshold=0.05):
        self.best_val_loss = best_val_loss_init
        self.val_threshold = val_threshold

    def maybe_save(self, this_val_loss):
        if this_val_loss < self.best_val_loss:
            self.best_val_loss = this_val_loss
            return True
        elif this_val_loss < self.val_threshold:
            return True
        return False

class LOGGER:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    def get_logger(self):
        return self.logger

    def add_console(self, default_format=True):
        self.console_H = logging.StreamHandler()
        self.console_H.setLevel(logging.INFO)
        self.console_H.setFormatter(self.format)
        self.logger.addHandler(self.console_H)

    def add_file(self, file_name, file_dir='trace', default_format=True):
        file_path = os.path.join(file_dir, file_name)
        self.file_H = logging.FileHandler(file_path)
        self.file_H.setLevel(logging.DEBUG)
        self.file_H.setFormatter(self.format)
        self.logger.addHandler(self.file_H)

def extract_name(fullpath):
    return os.path.splitext(os.path.split(fullpath)[-1])[0]

def color_image(image, num_classes=20):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def make_records_list_for_KITTI(dir):
    if os.path.exists(os.path.join(dir, 'rec.npy')):
        return np.load(os.path.join(dir, 'rec.npy'))
    assert os.path.exists(os.path.join(dir, 'training', 'image_2'))
    assert os.path.exists(os.path.join(dir, 'training', 'gt_image_2'))
    file_list = os.listdir(os.path.join(dir, 'training', 'image_2'))
    records_list = []
    for filename in file_list:
        if os.path.splitext(filename)[1] == '.png':
            img_path = os.path.join('training', 'image_2', filename)
            gt_name = filename.replace('_','_road_')
            gt_path = os.path.join('training', 'gt_image_2', gt_name)
            records_list.append({'img':img_path, 'gt':gt_path})
    assert len(records_list) == 289
    np.save(os.path.join(dir,'rec.npy'), records_list)
    return records_list

def get_evaluate_images(evaluate_dir='evaluate'):
    if not os.path.exists(evaluate_dir):
        raise ValueError('Path "{}" not exists!'.format(evaluate_dir))

    file_list = os.listdir(evaluate_dir)
    file_list = [i for i in file_list if os.path.splitext(i)[-1]=='.png']
    names = [i for i in file_list if i.split('-')[-1]!='eval.png']
    if not names:
        raise ValueError('No valid images in Path "{}"!'.format(evaluate_dir))

    def load_one(file_name, file_dir='evaluate', image_shape=(384, 1248, 3)):
        img = misc.imread(os.path.join(file_dir, file_name))
        img = misc.imresize(img, image_shape, interp='nearest')
        return img
    return np.array([load_one(name) for name in names]), [extract_name(i) for i in names]

def load_weights(session, saver, ckpt_name, ckpt_dir=r'trace/pick'):
    checkpoint_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(checkpoint_path):
        raise ValueError('"{}" file not found in path "{}".'.format(ckpt_name, ckpt_dir))
    else:
        saver.restore(session, checkpoint_path)
        return int(ckpt_name.split('-')[1])

def overlay_for_2_classes(basic_image, segmentation, color=[0, 255, 0, 127]):
    color = np.array(color).reshape(1,4)
    shape = basic_image.shape
    segmentation = segmentation.reshape(shape[0], shape[1], 1)

    overlay = np.dot(segmentation, color)
    overlay = misc.toimage(overlay, mode="RGBA")

    background = misc.toimage(basic_image)
    background.paste(overlay, box=None, mask=overlay)

    return np.array(background)

def confidence_map():
    pass

def save_images(image, name, save_dir='evaluate', prefix=None, postfix=None, ext='.png', logger=None):
    assert os.path.exists(save_dir)
    file_name = name
    if prefix:
        file_name = prefix + file_name
    if postfix:
        file_name = file_name + postfix
    file_name += ext
    file_path = os.path.join(save_dir, file_name)
    misc.imsave(file_path, image)
    if logger:
        logger.debug('Image %s save to %s success!' % (file_name, file_path))

# 在全局区定义句柄，多文件创建logger时，底层自动判断是否需要新建
# 解释来自：http://bbs.csdn.net/topics/392054240
# file_H = logging.FileHandler(r'trace/train.log')
# console_H = logging.StreamHandler()
# def logging_config():
#     logger = logging.getLogger('mylogger')
#     logger.setLevel(logging.DEBUG)
#     file_H.setLevel(logging.DEBUG)
#     console_H.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#     file_H.setFormatter(formatter)
#     console_H.setFormatter(formatter)
#     logger.addHandler(file_H)
#     logger.addHandler(console_H)
#     return logger

if __name__ == '__main__':
    data_dir = r'D:\GitHub\Road-Segmentation\data'
    records_list = make_records_list_for_KITTI(data_dir)
#    pass
