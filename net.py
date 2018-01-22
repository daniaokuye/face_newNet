import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


class wider_net(nn.Module):
    def __init__(self):
        super(wider_net, self).__init__()
        self.build_base()
        self.up = nn.Upsample(scale_factor=2)
        self.filter_1 = nn.Conv2d(1024, 256, 1)
        self.filter_2 = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        # conv 4_3 =25;conv5_3=29
        y = 0
        for k in range(len(self.base_net)):
            x = self.base_net[k](x)
            if k == 22:
                y = x
        if not isinstance(y, int):
            x = self.up(x)
            x = torch.cat((x, y), 1)
            x = self.filter_1(x)
            x = self.filter_2(x)
        # for k in range(len(self.extra)):
        #    x = self.extra[k](x)
        return x

    def add_extras(self):
        """
        add extra modules behind the net vgg
        """
        fc6 = nn.Conv2d(512, 1024, 3, padding=1)
        fc7 = nn.Conv2d(1024, 1024, 1)
        conv8_1 = nn.Conv2d(1024, 256, 1)
        conv8_2 = nn.Conv2d(256, 512, 3, 2, 1)  # s2
        self.extra = nn.ModuleList([fc6, fc7, conv8_1, conv8_2])

    def weight_init(self, m):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias, 0.1)

    def build_base(self):
        """
        build base net VGG according to ssd
        """
        base = []
        vgg16 = models.vgg16(pretrained=True)
        for i, key in enumerate(vgg16.features):
            if i < 30:
                base.append(key)
        self.base_net = nn.ModuleList(base)
        # self.add_extras()
        # for ex in self.extra:
        #    ex.apply(self.weight_init)


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
        from collections import OrderedDict
        trans_param = OrderedDict()
        for item, value in saved_state.items():
            name = '.'.join(item.split('.')[1:])
            trans_param[name] = value
        net.load_state_dict(trans_param)
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


if __name__ == "__main__":
    pass
