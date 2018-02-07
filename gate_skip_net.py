import os, time
import torch, threading
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
from data_read import show_feature
import matplotlib.pyplot as plt
from cfg import get_status, total_thread
from data_read import gate_random_mask


class gate_skip_net(nn.Module):
    def __init__(self):
        super(gate_skip_net, self).__init__()
        self.trunk_layers = [1, 3]
        self.build_base()

        # self.up = nn.Upsample(scale_factor=2)
        # self.filter_1 = nn.Conv2d(1024, 256, 1)
        # self.filter_2 = nn.Conv2d(256, 1, 1)
        # self.maxp = nn.MaxPool2d(2, 2)

    def forward(self, x, skip_conn=False):
        y = 0
        # z = []
        # layers = [64, 128, 256, 512, 512]  # 1 2 4 8 16
        # out_layers = 20
        # skip_NO = 0  # from 0 to the last skip connected point
        for k in range(len(self.base_net) - 1):  #
            # print 'k :', k, ' '
            # add skip connection before BN
            # if isinstance(self.base_net[k], nn.BatchNorm2d):
            #     # data
            #     x_clone = x.clone()
            #
            #     #  embeded net in big net
            #     input_layer = layers[skip_NO]
            #     up_trunk_1 = nn.Conv2d(64, 20, 3)
            #     up_trunk_2 = nn.Sigmoid()
            #     down_trunk_1 = nn.Conv2d(64, 20, 1)
            #     skip_NO += 1

            x = self.base_net[k](x)
            # z.append(torch.mean(x, 1).cpu().data.numpy())
            if k == 31:
                y = x
        if not isinstance(y, int):
            # x = self.up(x)
            if not skip_conn:
                # zero = Variable(torch.FloatTensor([0]))
                y = y * 0.0
            x = torch.cat((x, y), 1)
            x = self.filter_1(x)
            x = self.filter_2(x)
        # for k in range(len(self.extra)):
        #    x = self.extra[k](x)

        return x

    def add_skip_connection(self, input_layer):
        """
        embeded skip connection net which located before pooling in big net
        """
        out_layer = input_layer / 8
        up_trunk_1 = nn.Conv2d(input_layer, out_layer, 3)
        up_trunk_2 = nn.Sigmoid()
        down_trunk_1 = nn.Conv2d(input_layer, out_layer, 1)

        feature2heatmap1 = nn.Conv2d(out_layer, out_layer * 8, 3)
        feature2heatmap2 = nn.Conv2d(out_layer * 8, out_layer * 4, 3)
        heatmap3 = nn.Conv2d(out_layer * 4, 1, 1)
        return [up_trunk_1, up_trunk_2, down_trunk_1,
                feature2heatmap1, feature2heatmap2, heatmap3]  # 3 & 3

    def build_base(self):
        """
        build base net VGG according to ssd
        """
        base, main_fold, trunks = [], [], []
        idx = 0
        vgg16 = models.vgg16(pretrained=True)
        out_channel = 0  # vaiable to define BN & trunk
        for i, key in enumerate(vgg16.features):
            if isinstance(key, nn.modules.Conv2d):
                out_channel = key.out_channels
            # BN then ReLu
            if isinstance(key, nn.modules.activation.ReLU):
                base.append(nn.BatchNorm2d(out_channel))
            # add extra trunk before pooling
            if isinstance(key, nn.MaxPool2d):
                if idx in self.trunk_layers:
                    trunk = self.add_skip_connection(out_channel)
                    trunks.append(nn.Sequential(*trunk))
                main_fold.append(nn.Sequential(*base))
                base = []
                idx += 1
            base.append(key)
        self.base_net = nn.ModuleList(main_fold)
        self.trunk = nn.ModuleList(trunks)


def load_saved(net, path):
    '''init the net or load existing parameters'''
    # find existing pth
    parents = path.split('face_')[0]
    bigger = []
    for file in os.listdir(parents):
        if '.pth' in file:
            file = file.split('__')
            if len(file) >= 2:
                bigger.append(int(file[1]))
    if bigger:
        saved_data = torch.load(path.format(max(bigger)))
        saved_state = saved_data['net_state']

        # in case of module dataparallel was used
        if 'module' in saved_state.keys()[0]:
            from collections import OrderedDict
            trans_param = OrderedDict()
            for item, value in saved_state.items():
                name = '.'.join(item.split('.')[1:])
                trans_param[name] = value
            net.load_state_dict(trans_param)
        else:
            net.load_state_dict(saved_state)

        now_iter = saved_data['iter']
        return now_iter
    return 0


def get_param(net, lr):
    # get param for optims,bias and weight have two lr
    lr1 = []
    lr2 = []

    for key, value in net.named_parameters():
        if 'bias' in key:
            lr2.append(value)  # bias
        else:
            lr1.append(value)  # weight
    params = [{'params': lr1, 'lr': lr},
              {'params': lr2, 'lr': lr * 2}]
    return params


# prepare prediction with mask aidding by used_layer
# feature [n*1*h*w], mask [n*2*h*w]--[big, tiney]
def prepare_prediction_with_mask(feature, mask, used_layer):
    # mask_ = mask.copy()
    use_skip = get_status()
    for name in used_layer.keys():
        layer = int(name.split('batch_')[-1])
        gate_random_mask(feature[layer, 0], used_layer[name], assign_score_by_given_sacle)
        # sort every items in every content in batch
        multi_gtbox_in_used_layer(mask[layer], used_layer[name], sort_solution_for_used_layer)
        # will draw random boxes on mask by used_layer
        multi_gtbox_in_used_layer(mask[layer], used_layer[name], draw_given_random_box_to_mask)

    # decide wheather both  big & tiney are all used in mask
    # mask have two dimension big & tiney but shape is (1,2,h,w)
    mask = np.max(mask, axis=1) if use_skip else mask[:, 0]
    device_id = feature.get_device()
    mask = Variable(torch.from_numpy(mask)).unsqueeze(1).cuda(device_id)
    return mask


# -----------------------------------------------------------
#
# -------------- server for prediction mask -----------------
#
# -----------------------------------------------------------
# common function for multi thread
def multi_gtbox_in_used_layer(mask, used_layer, aim_function):
    # multi threading
    is_multi_thred = False if len(used_layer) < 2 + 4 else True
    res = []
    big = used_layer['big']
    # tiney = used_layer['tiney']
    for i in used_layer.keys():  # all gt boxes in used_layer
        if not isinstance(i, int):
            continue
        which_mask = 0 if i in big else 1
        if is_multi_thred:
            # self.__target(*self.__args, **self.__kwargs)
            t = threading.Thread(target=aim_function, args=(used_layer[i], mask[which_mask]))
            res.append(t)
        else:
            aim_function(used_layer[i], mask[which_mask])
    for t in res:
        while True:
            # waiting for the num go down
            # http://blog.csdn.net/jianhong1990/article/details/14671689
            if len(threading.enumerate()) < total_thread:
                break
        t.setDaemon(True)  # https://www.cnblogs.com/fnng/p/3670789.html
        t.start()
    if is_multi_thred:
        res[-1].join()


# this function was used for sort every item of random boxes
def sort_solution_for_used_layer(individual, no_use):  # a list
    for i in range(1, len(individual)):
        if len(individual[i]) == 4:
            continue
        else:
            try:
                assert len(individual[i]) == 3
            except Exception, e:
                print e
            individual[i].append(0)
    individual[1:] = sorted(individual[1:], key=lambda box: box[-1], reverse=True)


def assign_score_by_given_sacle(W, H, feature, used_layer_, idx):
    # h, w = used_layer_[0]
    # idx=0 was origin box info h&w
    if idx > 0 and len(used_layer_[idx]) == 3:
        direction, x, y = used_layer_[idx]
        compose = np.array(((1, 1), (-1, 1), (-1, -1), (1, -1)))
        h_, w_ = np.array(used_layer_[0]) * compose[direction]
        a_y, b_y = (y, y + h_) if y < y + h_ else (y + h_, y)
        a_x, b_x = (x, x + w_) if (x < x + w_) else (x + w_, x)
        score = torch.sum(feature[a_y:b_y + 1, a_x:b_x + 1])  # a number in Variable GPU
        used_layer_[idx].append(score.cpu().data.numpy()[0])


def draw_given_random_box_to_mask(individual, mask):
    # big = used_layer['big']
    # tiney = used_layer['tiney']
    compose = np.array(((1, 1), (-1, 1), (-1, -1), (1, -1)))
    nums_mask = 3  # meaning 1:3 for positive vs negetive
    # for i in used_layer.keys():
    #     if not isinstance(i, int) or len(used_layer[i]) < 2:
    #         continue
    #     which_mask = 0 if i in big else 1
    num = nums_mask if len(individual) > nums_mask else len(individual) - 1
    context = 2  # slice is 1, then its context is context-1=1
    for idx in range(num):
        index = idx + 1
        patch = individual[index]
        h_, w_ = np.array(individual[0]) * compose[patch[0]]
        x, y = patch[1:3]
        a_y, b_y = (y, y + h_) if y < y + h_ else (y + h_, y)
        a_x, b_x = (x, x + w_) if (x < x + w_) else (x + w_, x)
        # 1 is this noisy should be; else (gt 1) will penalize those points
        mask[a_y:b_y + context, a_x:b_x + context] = 1.5
    # return mask
    # print 'o'


if __name__ == "__main__":
    net = gate_skip_net()
    print 'o'
