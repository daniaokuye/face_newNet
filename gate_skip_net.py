import os, time, cv2
import torch, threading
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
from data_read import show_feature
import matplotlib.pyplot as plt
from cfg import get_status, total_thread
from data_read import gate_random_mask


def score_fn(*argus):
    pass


class gate_skip_net(nn.Module):
    def __init__(self):
        super(gate_skip_net, self).__init__()
        self.trunk_layers = [1, 3]
        self.gamma = 5  # for sigmoid
        self.build_base()

    def forward(self, x, scores_fn=score_fn()):  # canvas
        Is_use_skip = get_status()  # False
        # big, tiny = canvas_big_tiny(canvas, x.get_device())
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
                sf = self.tiny_net_for_fliter(x, idx)
            else:
                # This part will not be used in test phase
                skip_out = self.tiny_net_for_fliter(x.detach(), idx)  # train tiney net in net
                # selected features -- at this point, we want bp to main trunk -- .detach()
                sf = scores_fn(k, self.trunk_net[idx][2].out_channels, x)
                # todo:ground truth should be follow big & tiny
                tiny_out += f.l1_loss(skip_out, sf)

            sf = self.trunk_net[idx][3](sf)
            trunk_out.append(sf)
        # todo:simplify.big & tiny switch can move to loss, but it increase the calculation
        trunk_out.append(x)
        return trunk_out, tiny_out  # 1.3.4

    def add_skip_connection(self, input_layer):
        """
        embeded skip connection net which located before pooling in big net
        """
        out_layer = input_layer / 8
        fliter_part, feature_part = [], []

        # fliter_part
        fliter_part.append(nn.Conv2d(input_layer, input_layer, 3, padding=1))  # up_trunk_1
        # fliter_part.append(nn.BatchNorm2d(input_layer))
        fliter_part.append(L2Norm(input_layer, self.gamma))
        fliter_part.append(nn.Sigmoid())  # up_trunk_2
        # fliter_part.append(nn.ReLU())
        upper = nn.Sequential(*fliter_part)
        lower_1 = nn.Conv2d(input_layer, input_layer, 3, padding=1)
        lower_2 = nn.Conv2d(input_layer, out_layer, 1)

        # feature_part
        feature_part.append(nn.Conv2d(out_layer, out_layer * 8, 3, padding=1))  # feature2heatmap1
        feature_part.append(nn.Conv2d(out_layer * 8, out_layer * 4, 3, padding=1))  # feature2heatmap2
        feature_part.append(nn.Conv2d(out_layer * 4, 1, 1))  # heatmap3
        feature_part.append(nn.ReLU())  # relu
        supply = nn.Sequential(*feature_part)

        return nn.ModuleList([upper, lower_1, lower_2, supply])

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
        main_fold.append(nn.Sequential(*self.add_extras(out_channel)))
        self.base_net = nn.ModuleList(main_fold)
        self.trunk_net = nn.ModuleList(trunks)

    def add_extras(self, out_channels):
        """
        add extra modules behind the net vgg
        """
        fc6 = nn.Conv2d(out_channels, out_channels / 2, 1)
        fc7 = nn.Conv2d(out_channels / 2, 1, 1)
        return [fc6, fc7]

    def tiny_net_for_fliter(self, trunk, trunk_layer):
        # todo: tiny net, net in net
        # t = trunk.detach()  # trunk; x.clone()
        # trunk.requires_grad = True
        fliter = self.trunk_net[trunk_layer][0](trunk)  # upper
        fliter = nn.ReLU()(fliter - 0.5)
        transfer = self.trunk_net[trunk_layer][1](trunk)  # lower_1
        transfer = torch.mul(transfer, fliter)
        transfer = self.trunk_net[trunk_layer][2](transfer)  # select
        return transfer


def chunk_gate_net(gt_):
    def score_feature_map(resize_ratio, k, feature):
        """gt_,
        get score from feature; This function will be used in gate_net
        :param gt_: numpy array
        :param resize_ratio: resize to feature
        :param k: top k
        :param feature: used for score sort
        :return:
        """
        # todo:if there's no tiney face or big face, this function should be a switch
        batch, _, h, w = gt_.shape
        gt = gt_[:, 1].transpose(1, 2, 0)
        gt = cv2.resize(gt, (w / (2 ** resize_ratio), h / (2 ** resize_ratio)),
                        interpolation=cv2.INTER_NEAREST)
        if batch == 1:
            gt = gt[np.newaxis]
        else:
            gt = gt.transpose(2, 0, 1)
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
        _, tiny = canvas_big_tiny(gt_, feature.get_device())
        return torch.mul(tiny, res)

    return score_feature_map


def canvas_big_tiny(canvas, cuda_idx):
    # to obtain status of canvas which is numpy array
    c = canvas.copy()
    for i in range(3, 1, -1):
        c = np.sum(c, i)
    c = (c > 0).astype(np.float32)
    big, tiny = c[:, 0], c[:, 1]
    big = Variable(torch.from_numpy(big)).cuda(cuda_idx)
    tiny = Variable(torch.from_numpy(tiny)).cuda(cuda_idx)
    for i in range(3):
        big = big.unsqueeze(-1)
        tiny = tiny.unsqueeze(-1)
    return big, tiny


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
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            init.constant(m.bias, 0.1)
            init.xavier_uniform(m.weight.data)
        # elif 'trunk' in name and isinstance(m, nn.BatchNorm2d):
        #     init.constant(m.weight, 4.0)
        #     init.xavier_uniform(m.weight.data)


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.FloatTensor(n_channels))
        self.reset_parameter()

    def reset_parameter(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x
        return out


if __name__ == "__main__":
    net = gate_skip_net()
    print 'o'
