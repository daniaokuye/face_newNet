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
from cfg import stride_all
import argparse
import cv2, PIL, time, os
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance, Image
import multiprocessing, threading


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
                        default='wf_train', type=str)
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


# -----------------------------------------------------------
#
# -------------build canvas----------------
#
# -----------------------------------------------------------
def defin_canv_Pro(anno, w, h):
    x0, y0, x1, y1 = anno
    x, y = (x1 + x0) / 2.0, (y0 + y1) / 2.0
    eclipse_a = x - x0
    eclipse_b = y - y0
    canvas = np.zeros((h, w))
    if eclipse_b and eclipse_a:
        canvas[y0 - 1:y1, x0 - 1:x1] = 1
        eclipse = build_eclipse(x, y, eclipse_a, eclipse_b, (w, h))
        canvas *= eclipse
    return canvas


# def defin_canv(i,anno,canvas, w, h):
#     x0, y0, x1, y1 = anno
#     x, y = (x1 + x0) / 2.0, (y0 + y1) / 2.0
#     eclipse_a = x - x0
#     eclipse_b = y - y0
#     if eclipse_b and eclipse_a:
#         canvas[i, y0 - 1:y1, x0 - 1:x1] = 1
#         eclipse = build_eclipse(x, y, eclipse_a, eclipse_b, (w, h))
#         canvas[i] *= eclipse
#     #return canvas

# generate mask & ground truth
def defin_canv(i, anno, canvas, mask, used_layer, w, h):
    x0, y0, x1, y1 = anno
    x, y = (x1 + x0) / 2.0, (y0 + y1) / 2.0
    eclipse_a = x - x0
    eclipse_b = y - y0
    # work = True
    # big_now = get_status()
    # if big_now and not (eclipse_a * 2 / 16 >= 1 and eclipse_b * 2 / 16 >= 1):  # eclipse_a*2 / 16
    #     work = False
    if eclipse_b and eclipse_a:
        mask[i, y0 - 1:y1, x0 - 1:x1] = 1
        if eclipse_a * 2 / 16 >= 1 and eclipse_b * 2 / 16 >= 1:
            used_layer['big'].append(i)
        else:
            used_layer['tiney'].append(i)
        coor = (eclipse_b * 2, eclipse_a * 2)
        used_layer[i] = [[int(np.ceil(c)) for c in coor]]
        eclipse = build_eclipse(x, y, eclipse_a, eclipse_b, (w, h))
        canvas[i] = mask[i] * eclipse


# -----------------------------------------------------------
#
# -------------build Heatmap----------------
#
# -----------------------------------------------------------
def ground_truth_heat_map(size, box, vis=False):
    """
    draw box as an image
    """
    faces = len(box)
    h, w, _ = size
    canvas = np.zeros((faces, h, w))  # line col;
    mask = np.zeros((faces, h, w))
    result = []
    used_layer = {'big': [], 'tiney': []}  # those layers for draw heatmap
    # # multi processing : not quick
    # pool = multiprocessing.Pool()
    # for anno in box:
    #     result.append(pool.apply_async(defin_canv_Pro, args=(anno, w, h)))
    # pool.close()
    # pool.join()
    # for i in range(faces):
    #     canvas[i] = result[i].get()

    if len(box) > 3:
        # multi thread : the leat quick
        for i, anno in enumerate(box):
            t = threading.Thread(target=defin_canv, args=(i, anno, canvas, mask, used_layer, w, h))
            result.append(t)
        for t in result:
            t.setDaemon(True)  # https://www.cnblogs.com/fnng/p/3670789.html
            t.start()
        result[-1].join()
    else:
        # the usual method : quick
        for i, anno in enumerate(box):
            defin_canv(i, anno, canvas, mask, used_layer, w, h)

    canvas = np.max(canvas, axis=0)
    # mask = np.max(mask, axis=0)
    # generate mask for positive & negtive sample with default num_random_boxes =10
    gate_random_mask(np.max(mask, axis=0), used_layer, random_mask_generator)
    # mask_ used for seen random negtive samples;
    mask_ = draw_mask_with_usedLayer(mask, used_layer)
    mask = squeeze_mask(mask, used_layer)  # [big, tiney]
    if vis:
        fig, ax = plt.subplots()
        ax.imshow(canvas)
        cv2.imwrite('multi-face.png', (canvas * 255))
    return canvas, mask, used_layer


# squeeze mask layer to 2:big & tiney
def squeeze_mask(mask, used_layer):
    big = used_layer['big']
    tiney = used_layer['tiney']
    big_masked = np.max(mask[big], axis=0) if big else 0
    tiney_masked = np.max(mask[tiney], axis=0) if tiney else 1
    masked = [big_masked, tiney_masked]
    for i, m in enumerate(masked):
        if isinstance(m, int):
            masked[i] = np.zeros(masked[not i].shape)
    masked = np.vstack((masked[0][np.newaxis], masked[1][np.newaxis]))
    return masked


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
    eclipse = np.e ** eclipse * 2  # make difference more clear
    return eclipse


# -----------------------------------------------------------
#
# ----------------build mask-----------------------
#
# -----------------------------------------------------------
# this function is for visul checking wheather the used_layer is working properly
def draw_mask_with_usedLayer(mask_, used_layer_):
    mask = np.max(mask_, axis=0)
    compose = np.array(((1, 1), (-1, 1), (-1, -1), (1, -1)))
    nums_mask = 3
    for i in used_layer_.keys():
        if not isinstance(i, int) or len(used_layer_[i]) < 2:
            continue
        size = nums_mask if len(used_layer_) > nums_mask else len(used_layer_) - 1
        random_idx = np.random.randint(1, len(used_layer_[i]), size=size)
        # print 'build mask with i: ', random_idx, i
        for idx in random_idx:
            patch = used_layer_[i][idx]
            x, y = patch[1:]
            h_, w_ = np.array(used_layer_[i][0]) * compose[patch[0]]
            a_y, b_y = (y, y + h_) if y < y + h_ else (y + h_, y)
            a_x, b_x = (x, x + w_) if (x < x + w_) else (x + w_, x)
            mask[a_y:b_y, a_x:b_x] = 2
    return mask


# build random mask for + & - samples
# only used_layers in layers of mask were used
# two levels loop:boxes in one image; random shadow for every box
# Thread_fun was used for last level aims:how to deal with every random shadows
def gate_random_mask(mask, used_layer, Thread_fun):
    # masked = np.max(mask, axis=0)
    if isinstance(mask, np.ndarray):
        H, W = mask.shape
    else:
        H, W = mask.size()
    common = True  # use single processing or False for multiprocessing
    if len(used_layer.keys()) > 2 + 5:  # big & tiney & others
        common = False
        results = []
    # print 'keys:', used_layer.keys()
    for i in used_layer.keys():
        if not isinstance(i, int):
            continue
        if common:
            build_mask_withThread(W, H, mask, used_layer[i], Thread_fun)
        else:
            t = threading.Thread(target=build_mask_withThread,
                                 args=(W, H, mask, used_layer[i], Thread_fun))
            results.append(t)
    if not common:
        for t in results:
            t.setDaemon(True)
            t.start()
        results[-1].join()


# function = random_mask_generator
def build_mask_withThread(W, H, masked, used_layer_now, Thread_fun):
    # print 'Run task (%s)...' % (os.getpid()),
    result = []
    # generator 10 default box of mask
    judge_num_random = len(used_layer_now) == 1  # and used_layer_now[0]
    num_random_boxes = 10 if judge_num_random else len(used_layer_now)
    for j in range(num_random_boxes):
        t = threading.Thread(target=Thread_fun, args=(W, H, masked, used_layer_now, j))
        result.append(t)
    for t in result:
        t.setDaemon(True)
        t.start()
    result[-1].join()


# randomly select parts
def random_mask_generator(W, H, mask, box, idx):
    h, w = box[0]
    anchor_x, anchor_y, direction, times = 0, 0, -1, 0
    while direction < 0 and times < 30:
        times += 1
        anchor_x = int(np.ceil(W * np.random.random()))
        anchor_y = int(np.ceil(H * np.random.random()))
        direction = which_quadrant(anchor_x, anchor_y, h, w, mask)
    # print 'time & direction: ', direction, times
    if direction != -1:
        box.append([direction, anchor_x, anchor_y])


# 1234 four quadrant which direction will generate box
def which_quadrant(x, y, h, w, mask):
    buff = 5 + 1 * stride_all  # one pixel buff with stride 8, result >1.5
    h, w = h + buff, w + buff
    H, W = mask.shape
    compose = np.array(((1, 1), (-1, 1), (-1, -1), (1, -1)))
    for i, item in enumerate(compose):
        h_, w_ = np.array((h, w)) * item
        if not (0 < y + h_ < H and 0 < x + w_ < W):
            continue
        a_y, b_y = (y, y + h_) if y < y + h_ else (y + h_, y)
        a_x, b_x = (x, x + w_) if (x < x + w_) else (x + w_, x)
        if np.sum(mask[a_y:b_y + 1, a_x:b_x + 1]) == 0:
            return i  # 0123
    return -1


# -----------------------------------------------------------
#
# ------------ data augment -----------------------------
#
# -----------------------------------------------------------
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


def show_feature(data_, RGB=False):
    data = data_.cpu().data.numpy()
    b, c, h, w = data.shape
    # assert c == 1
    maxD = np.max(data, (2, 3))
    minD = np.min(data, (2, 3))
    maxArr = np.tile(maxD, (h, w, 1, 1)).transpose((2, 3, 0, 1))
    minArr = np.tile(minD, (h, w, 1, 1)).transpose((2, 3, 0, 1))
    print 'max of feature map: ', maxD
    print 'min of feature map: ', minD
    data = (data - minArr) / (maxArr - minArr) * 255
    data = data.astype(np.uint8)
    show_inx = np.random.randint(b)
    if RGB:
        im = Image.fromarray(data[show_inx].transpose(1, 2, 0))
    else:
        im = Image.fromarray(data[show_inx, 0])
    im.show()


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
        t = time.time()
        anno = imdb._load_face_annotation(term)['boxes']
        image_path = imdb.image_path_from_index(term)
        image = cv2.imread(image_path)
        sz = image.shape
        # print 'sz:', sz
        ground_truth_heat_map(sz, anno, vis=vis)
        print term, ' : ', time.time() - t
    if vis:
        plt.show()
    return


def combined_roidb(phase='train'):
    """
    show how to obtain image & annotation
    """
    args = parse_args()
    if phase == 'test':
        args.imdb_name = 'wf_train1'
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
    # print 'sz:', sz
    gt_map, mask, used_layer = ground_truth_heat_map(sz, anno)
    # these data will be return having: h*w, h*w, 2*h*w, keys=['big','tiney',1,2,...]
    return image, gt_map.astype(np.float32), mask.astype(np.float32), used_layer


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    combined_roidb1(args.imdb_name)
