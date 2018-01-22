import torch.nn as nn
import cv2
import numpy as np
import torch
from torch.autograd import Variable


class loss_detection(nn.Module):
    def __init__(self):
        super(loss_detection, self).__init__()
        self.MSE = nn.MSELoss()
        self.smL1 = nn.SmoothL1Loss()

    def forward(self, heatmap, predicted):  # , gt
        """

        :param heatmap: heatmap of face
        :param predicted: conv4_3 & conv5_3 concat and filter
        :param gt: points of boxes
        :return:
        """
        # heatmap = Variable(heatmap).cuda()
        l = self.MSE(predicted, heatmap)
        return l

    def map2point(self, heatmap, gt):
        pass


def test_map():
    """
    gradation will be used to obtain the top point
    """
    import matplotlib.pyplot as plt
    path = 'multi-face.png'
    image = cv2.imread(path)
    # h, w, _ = image.shape
    # h_, w_ = (h / 2 + 1), (w / 2 + 1)
    image = image[:, :, 0] / 255.0
    image = Variable(torch.from_numpy(image)).unsqueeze(0).unsqueeze(0).float()

    image_x = nn.ZeroPad2d((1, 0, 0, 0))  # move left by 1 pixel
    image_y = nn.ZeroPad2d((0, 0, 1, 0))  # move up by 1 pixel
    grad_x = image - image_x(image)[:, :, :, :-1] + 1e-15
    grad_y = image - image_y(image)[:, :, :-1, :] + 1e-15
    sign_x = (grad_x / grad_x.abs())  # .char()
    sign_y = (grad_y / grad_y.abs())  # .char()
    sign = sign_x * sign_y  # useful
    valid = (image > 0).float()  # useful
    sign = sign * valid + (1 - valid)  # image_xy will has misplacement
    # plt.imshow(image_xy.long().data.numpy()[0,0])
    # creat four layer
    kernel = Variable(torch.FloatTensor([1, -1, -1, 1]))

    mod_1 = sign.clone()
    mod = nn.ZeroPad2d((0, 1, 0, 0))
    mod_2 = mod(sign)[:, :, :, 1:]
    mod = nn.ZeroPad2d((0, 0, 0, 1))
    mod_3 = mod(sign)[:, :, 1:, :]
    mod = nn.ZeroPad2d((0, 1, 0, 1))
    mod_4 = mod(sign)[:, :, 1:, 1:]
    mod_ = torch.cat((mod_1, mod_2, mod_3, mod_4), 1)
    mod_ = mod_.transpose(1, 2).transpose(2, 3)
    out = torch.matmul(mod_, kernel)
    out_ = (out > 3.5).float()

    print 'o'


if __name__ == "__main__":
    test_map()
