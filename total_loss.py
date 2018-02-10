import torch.nn as nn
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from net import prepare_prediction_with_mask
from data_prepare import trans_used_layer


class loss_detection(nn.Module):
    def __init__(self):
        """
       heatmap: heatmap of face; numpy; 800*800
       predicted: conv4_3 & conv5_3 concat and filter
       gt: points of boxes
       """
        super(loss_detection, self).__init__()
        self.MSE = nn.MSELoss()
        self.smL1 = nn.SmoothL1Loss()
        self.layers = [1, 3, 4]

    def forward(self, predicted, heatmap, mask, used_layer):
        # loss=Variable(torch.zeros(1))
        predict, trunk = predicted
        device_id = trunk.get_device()
        for i, trunk_out in enumerate(predict):
            k = self.layers[i]
            mask_i = self.np_resize(mask, k)
            heatmap_i = self.np_resize(heatmap, k)
            used_layer_i = trans_used_layer(used_layer, [1.0 / (2 ** k), 1.0 / (2 ** k)])

            mask_i = prepare_prediction_with_mask(trunk_out, mask_i, used_layer_i)
            trunk_out = torch.mul(trunk_out, mask_i)
            heatmap_i = Variable(torch.from_numpy(heatmap_i)).cuda(self.device_id)
            trunk += self.MSE(trunk_out, heatmap_i)
        return trunk

    def np_resize(self, img_, ratio):
        batch, _, h, w = img_.shape
        gt = img_.transpose(2, 3, 0, 1)
        gt = cv2.resize(gt, (w / (2 ** ratio), h / (2 ** ratio)),
                        interpolation=cv2.INTER_NEAREST).transpose(2, 3, 0, 1)
        return gt


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
