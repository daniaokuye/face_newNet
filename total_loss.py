import torch.nn as nn
import cv2, copy
import numpy as np
import torch
from torch.autograd import Variable
from net import prepare_prediction_with_mask
from data_prepare import trans_used_layer, trans_mask, trans_gt
from gate_skip_net import canvas_big_tiny


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
        # big, tiny = canvas_big_tiny(heatmap, device_id)
        # anno = [tiny, tiny, big]
        # mask_idx = [1, 1, 0]
        for i, trunk_out in enumerate(predict):
            # k = self.layers[i]
            # mask_i = self.np_resize(mask, k)  #
            # heatmap_i = self.np_resize(heatmap, k)
            # used_layer_i = self.used_layer_resize(used_layer, [1.0 / (2 ** k), 1.0 / (2 ** k)])

            # mask_i = prepare_prediction_with_mask(trunk_out, mask_i, used_layer_i)
            # mask_i = mask_i[:, mask_idx[i]].unsqueeze(1)
            # heatmap_i = heatmap_i[:, mask_idx[i]].unsqueeze(1)
            mask_i, heatmap_i = self.prepare_data(heatmap, mask, used_layer, trunk_out, i)
            heatmap_i = Variable(torch.from_numpy(heatmap_i)).cuda(device_id)

            trunk_out = torch.mul(trunk_out, mask_i)
            # ignore trunk according to big & tiny in case of adding unwanted loss
            # heatmap_i = torch.mul(heatmap_i, anno[i])
            trunk += self.MSE(trunk_out, heatmap_i)
        return trunk

    def prepare_data(self, heatmap, mask, used_layer, trunk_out, i):
        k = 2 ** (self.layers[i])
        # mask_i = self.np_resize(mask, k)  #
        # heatmap_i = self.np_resize(heatmap, k)
        batch, _, h, w = mask.shape
        mask_i, heatmap_i = [], []
        for j in range(batch):
            mask_i.append(trans_mask(mask[j], k, k)[np.newaxis])
            heatmap_i.append(trans_gt(heatmap[j], k, k)[np.newaxis])
        mask_i = np.vstack(mask_i)
        heatmap_i = np.vstack(heatmap_i)
        used_layer_i = self.used_layer_resize(used_layer, [1.0 / (2 ** k), 1.0 / (2 ** k)])
        mask_i = prepare_prediction_with_mask(trunk_out, mask_i, used_layer_i)

        if i < len(self.layers) - 1:  # i is not the last one
            # obtain the tiny one from mask & gt
            mask_i = mask_i[:, 1].unsqueeze(1)
            # heatmap_i = Variable(torch.from_numpy(heatmap_i)).cuda(device_id)
            heatmap_i = heatmap_i[:, 1][:, np.newaxis]  # .unsqueeze(1)
        else:
            mask_i = torch.max(mask_i, 1)[0].unsqueeze(1)
            heatmap_i = np.max(heatmap_i, 1)[:, np.newaxis]
        return mask_i, heatmap_i

    def used_layer_resize(self, used_layer_, hw_factor):
        used_layer = copy.deepcopy(used_layer_)
        for value in used_layer.values():
            trans_used_layer(value, hw_factor)
        return used_layer

    def np_resize(self, img_, ratio):
        batch, _, h, w = img_.shape
        res = []
        for i in range(batch):
            gt = img_[i].transpose(1, 2, 0)
            gt = cv2.resize(gt, (w / (2 ** ratio), h / (2 ** ratio)),
                            interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
            res.append(gt[np.newaxis])
        gt = np.vstack(res)
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
