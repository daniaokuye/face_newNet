#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import cv2, PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance, Image


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=3, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='wf_train1', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    '''
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    '''
    args = parser.parse_args()
    return args


def combined_roidb1(imdb_names):
    """
    show how to obtain image & annotation
    """
    imdb = get_imdb(imdb_names)
    print 'length is {}, located at :{},'.format(imdb.num_images, imdb.image_path_from_index('000000'))
    print 'anno is {} , and image indexs:{}'.format(imdb._load_face_annotation('000000')['boxes'], imdb.image_index)
    # image = PIL.Image.open(imdb.image_path_from_index('000000'))
    vis = True
    for term in imdb.image_index:
        anno = imdb._load_face_annotation(term)['boxes']
        image_path = imdb.image_path_from_index(term)
        image = cv2.imread(image_path)
        sz = image.shape
        print 'sz:', sz
        ground_truth_heat_map(sz, anno, vis=vis)
    if vis:
        plt.show()
    return


def combined_roidb():
    """
    show how to obtain image & annotation
    """
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    imdb = get_imdb(args.imdb_name)

    return imdb, imdb.image_index


def obtain_data(imdb, item):
    """
    obtain data according to index
    :param item: index for data
    :return:
    """
    term = imdb.image_index[item]
    anno = imdb._load_face_annotation(term)['boxes']
    image = cv2.imread(imdb.image_path_from_index(term))
    sz = image.shape
    print 'sz:', sz
    gt_map = ground_truth_heat_map(sz, anno)
    return image, gt_map.astype(np.float32)


def ground_truth_heat_map(size, box, vis=False):
    """
    draw box as an image
    """
    faces = len(box)
    h, w, _ = size
    canvas = np.zeros((faces, h, w))  # line col;

    for i, anno in enumerate(box):
        x0, y0, x1, y1 = anno
        x, y = (x1 + x0) / 2.0, (y0 + y1) / 2.0
        eclipse_a = x - x0
        eclipse_b = y - y0
        canvas[i, y0 - 1:y1, x0 - 1:x1] = 1
        eclipse = build_eclipse(x, y, eclipse_a, eclipse_b, (w, h))
        canvas[i] *= eclipse
    canvas = np.max(canvas, axis=0)
    if vis:
        fig, ax = plt.subplots()
        ax.imshow(canvas)
        cv2.imwrite('multi-face.png', (canvas * 255))
    return canvas


def build_eclipse(x, y, L_a, C_b, sz):
    """
    draw eclipse which has core(x,y) corresponding to horitical  and vertical
    circle L_a & C_b corresponding to horitical  and vertical
    the image has size equal sz
    """
    w, h = sz
    delta = 1.1
    eclipse_Lx = np.tile(np.arange(w), (h, 1)).astype(float)
    eclipse_Ly = np.tile(np.arange(h), (w, 1)).T.astype(float)
    eclipse_Lx = ((eclipse_Lx - x) / L_a) ** 2
    eclipse_Ly = ((eclipse_Ly - y) / C_b) ** 2
    eclipse = -(eclipse_Lx + eclipse_Ly) / (2 * delta ** 2)
    eclipse = np.e ** eclipse
    return eclipse


def jitter(image):
    # image = Image.fromarray(data)
    random_factor = np.random.randint(0, 31) / 10.
    image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(5, 11) / 10.
    image = ImageEnhance.Brightness(image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    image = ImageEnhance.Contrast(image).enhance(random_factor)

    # image = np.array(image)
    return image


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    combined_roidb1(args.imdb_name)
