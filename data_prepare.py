# this fucntion together with newnet.py
# are used for loading data & ground truth
import torch, cv2, copy  # , time, threading
import torch.utils.data as data
import torchvision.transforms as transform
from data_read import combined_roidb, obtain_data, jitter
from cfg import *
from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt


class wider_face(data.Dataset):
    def __init__(self, phase='train'):
        self.imdb, self.image_index = combined_roidb(phase)

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, item):
        # there are 2 dims for gt_map & mask
        image, gt_map, mask, used_layer = obtain_data(self.imdb, item)
        image = self.trans_img(image)
        gt_map = self.trans_gt(gt_map)
        mask = self.trans_mask(mask)
        used_layer = self.trans_layer(gt_map, used_layer)
        return image, gt_map, mask, used_layer

    def trans_img(self, data):
        trans = transform.Compose([
            transform.ToPILImage(),
            transform.Lambda(jitter),
            transform.Resize((HW_h, HW_w)),
            transform.ToTensor(),
            transform.Normalize(mean=Norm_mean, std=Norm_std)
        ])
        data = trans(data)
        # show_data(data)
        return data

    def trans_mask(self, mask_):
        gt = mask_.transpose(1, 2, 0)
        gt = cv2.resize(gt, (HW_w, HW_h),
                        interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        # gt = gt[:, np.newaxis]
        return gt

    def trans_gt(self, gt_):
        gt = gt_.transpose(1, 2, 0)
        gt_LINEAR = cv2.resize(gt, (HW_w, HW_h),
                               interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        gt_NEAR = cv2.resize(gt, (HW_w, HW_h),
                             interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        gt_NEAR = (gt_NEAR > 0).astype(gt_LINEAR.dtype)
        gt = gt_LINEAR * gt_NEAR
        return gt

    def trans_layer(self, gt, used_layer_):
        total_stride = 1  # 8
        C, H, W = gt.shape if len(gt.shape) == 3 else [1] + list(gt.shape)
        H_W_factor = [1.0 * HW_h / (H * total_stride), 1.0 * HW_w / (W * total_stride)]
        used_layer = trans_used_layer(used_layer_, H_W_factor)
        return used_layer


def trans_used_layer(used_layer_, H_W_factor):
    # this factor will used in prepare ground truth & mask
    used_layer = copy.deepcopy(used_layer_)

    # used_layer has four nums now: width,height,left, top
    #
    # origin
    for i in used_layer.keys():
        if not isinstance(i, int):
            continue
        if len(used_layer[i]) == 1:
            # remove this key: no random box, invalid h or w
            used_layer.pop(i)
            continue
        for j in range(len(used_layer[i])):
            if j == 0:
                used_layer[i][0] = [int(round(x * y)) for x, y in
                                    zip(H_W_factor + H_W_factor[::-1], used_layer[i][0])]
            else:
                # direction, x, y
                used_layer[i][j][1:] = \
                    [int(round(x * y)) for x, y in zip(H_W_factor[::-1], used_layer[i][j][1:])]
    return used_layer


#
# -----------------------------------------------------------
#
def show_data(img):
    """
    just for show, there are nothing usage
    """
    # suppose data is tensor
    import numpy as np
    import matplotlib.pyplot as plt

    if isinstance(img, torch.FloatTensor):
        img = img.numpy().transpose(1, 2, 0)

    Dmin = np.min(img)
    Dmax = np.max(img)
    img = (img - Dmin) / (Dmax - Dmin) * 254
    print 'min & max', np.min(img), np.max(img)
    plt.imshow(img.astype(np.uint8))
    plt.show()


def patch_data(batch):
    """
    collate function
    patch several dataset to be a Variable
    """
    img, gt, mask, used_layer = [], [], [], {}
    keys = 'batch_{}'
    for i, patch in enumerate(batch):
        input, target, mk, use = patch
        img.append(input)
        # gt.append(target)  # torch.unsqueeze(target, 0)
        gt.append(target[np.newaxis])
        mask.append(mk[np.newaxis])
        used_layer[keys.format(i)] = use
    img = torch.stack(img, 0)
    # gt = torch.stack(gt, 0)  # .unsqueeze(1)  # batch,1,h,w
    gt = np.vstack(gt)
    mask = np.vstack(mask)  # .unsqueeze(1)
    return img, gt, mask, used_layer


if __name__ == '__main__':
    wf0 = wider_face()
    print 'len of wf:', len(wf0)
    # im, map = wf0[0]  # this is image and ground truth
