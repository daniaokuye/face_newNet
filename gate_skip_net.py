import os, time, cv2
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
        num_of_trunk = len(self.trunk_layers)
        self.bias_trunk = Variable(torch.FloatTensor(num_of_trunk))
        # self.up = nn.Upsample(scale_factor=2)
        # self.filter_1 = nn.Conv2d(1024, 256, 1)
        # self.filter_2 = nn.Conv2d(256, 1, 1)
        # self.maxp = nn.MaxPool2d(2, 2)

    def forward(self, x, canvas):
        Is_use_skip = get_status()  # False
        trunk_out = []  # result from trunk
        tiny_out = Variable(torch.zeros(1)).cuda(x.get_device())
        for k in range(len(self.base_net)):
            x = self.base_net[k](x)
            if k in self.trunk_layers:
                idx = self.trunk_layers.index(k)
            else:
                continue
            # trunk
            if Is_use_skip:
                trunk_out.append(self.trunk_net[idx](x))
            else:
                # selected features -- at this point, we want bp to main trunk -- .detach()
                sf = self.score_feature_map(canvas, k, self.trunk_net[idx][0].out_channels, x)
                tiny_out += self.tiny_net_for_fliter(x, sf, idx)

                for i, branch in enumerate(self.trunk_net[idx]):
                    if i < self.fliter_length:
                        continue
                    sf = branch(sf)
                trunk_out.append(sf)
        return trunk_out.append(x), tiny_out  # 1.3.4

    def add_skip_connection(self, input_layer):
        """
        embeded skip connection net which located before pooling in big net
        """
        out_layer = input_layer / 8
        fliter_part, feature_part = [], []

        # fliter_part
        fliter_part.append(nn.Conv2d(input_layer, out_layer, 3))  # up_trunk_1
        fliter_part.append(nn.BatchNorm2d(out_layer))  # nn.Linear(out_layer, out_layer)
        fliter_part.append(nn.Sigmoid())  # up_trunk_2
        fliter_part.append(nn.Conv2d(input_layer, out_layer, 1))  # down_trunk_1
        self.fliter_length = len(fliter_part)

        # feature_part
        feature_part.append(nn.Conv2d(out_layer, out_layer * 8, 3))  # feature2heatmap1
        feature_part.append(nn.Conv2d(out_layer * 8, out_layer * 4, 3))  # feature2heatmap2
        feature_part.append(nn.Conv2d(out_layer * 4, 1, 1))  # heatmap3

        return fliter_part + feature_part

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
        self.trunk_net = nn.ModuleList(trunks)

    def score_feature_map(self, gt_, resize_ratio, k, feature):
        """
        get score from feature
        :param gt_: numpy array
        :param resize_ratio: resize to feature
        :param k: top k
        :param feature: used for score sort
        :return:
        """
        batch, _, h, w = gt_.shape
        gt = gt_[:, 1].transpose(1, 2, 0)
        gt = cv2.resize(gt, (w / (2 ** resize_ratio), h / (2 ** resize_ratio)),
                        interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        gt = gt[:, np.newaxis]  # at this moment, gt has same dimensions with gt_
        mask = (Variable(torch.from_numpy(gt)) > 0).float()  # tiney face
        mask = mask.cuda(feature.get_device())
        wanted = torch.mul(mask, feature)  # n*1 & n*512
        for i in range(len(wanted.size()) - 1, 1, -1):  # sum last two dims
            wanted = torch.sum(wanted, i)
        idx = torch.topk(wanted, k, dim=1)[1]  # a tuple
        res = []
        for i in range(batch):
            res.append(feature[i][idx[i]])
        res = torch.stack(res)
        return res  # .detach()

    def tiny_net_for_fliter(self, trunk, selected_feature, trunk_layer):
        idx = trunk_layer
        # todo: tiny net, net in net
        t = trunk.detach()  # trunk; x.clone()
        fliter = self.trunk_net[idx][0](t)  # conv
        fliter = self.trunk_net[idx][1](fliter)  # linear
        fliter = self.trunk_net[idx][2](fliter)  # sigmoid
        t = self.trunk_net[idx][2](t)
        fliter = (fliter > 0).float()
        t = torch.mul(t, fliter)
        # similiar(t,sf)
        return nn.functional.l1_loss(t, selected_feature)


def load_parameter(net, path):
    # load saved parameters or just init by xaiver
    files = have_valid_model_parameter(path)
    iter = 0
    if files:
        files = path.format(max(files))
        iter = load_saved_model(net, files)
    else:
        init_uniform_parameters(net)
    return iter


def load_saved_model(net, files):
    # load parameters by the order of layers
    saved_data = torch.load(files)
    saved_state = saved_data['net_state']
    net.load_state_dict(saved_state)

    # in case of module data parallel was used
    # if 'module' in saved_state.keys()[0]:
    #     from collections import OrderedDict
    #     trans_param = OrderedDict()
    #     for item, value in saved_state.items():
    #         name = '.'.join(item.split('.')[1:])
    #         trans_param[name] = value
    #     net.load_state_dict(trans_param)
    # else:

    now_iter = saved_data['iter']
    return now_iter


def have_valid_model_parameter(path):
    # find existing pth
    assert '/face_' in path
    parents = path.split('face_')[0]
    bigger = []
    for files in os.listdir(parents):
        if '.pth' in files:
            files = files.split('__')
            if len(files) >= 2:
                bigger.append(int(files[1]))
    return bigger


def init_uniform_parameters(net):
    for m in net:
        if isinstance(m, nn.Conv2d):
            init.constant(m.bias, 0.1)
            init.xavier_uniform(m.weight.data)
        elif isinstance(m, nn.Linear):
            init.constant(m.bias, 4.0)
            init.xavier_uniform(m.weight.data)


if __name__ == "__main__":
    net = gate_skip_net()
    print 'o'
