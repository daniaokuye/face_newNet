# this fucntion together with newnet.py
# are used for loading data & ground truth
import torch, cv2, time, threading
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
        image, gt_map, mask, used_layer = obtain_data(self.imdb, item)
        # stride=8 resize
        image = self.trans_img(image)  # FloatTensor
        # t = time.time()
        gt_map = self.trans_gt(gt_map, used_layer)  # FloatTensor gt_map:h*w without third dimension
        # print 'with used_layer: ', time.time() - t
        mask = self.trans_gt(mask)
        return image, torch.from_numpy(gt_map), mask, used_layer

    def trans_img(self, data):
        trans = transform.Compose([
            transform.ToPILImage(),
            # transform.RandomHorizontalFlip(),
            transform.Lambda(jitter),
            transform.Resize((HW_h, HW_w)),
            transform.ToTensor(),
            transform.Normalize(mean=Norm_mean, std=Norm_std)
        ])  # be careful for flip
        data = trans(data)
        # show_data(data)
        return data

    def trans_gt(self, gt, used_layer=0):
        C, H, W = gt.shape if len(gt.shape) == 3 else [1] + list(gt.shape)
        if isinstance(used_layer, dict):
            H_W_factor = [1.0 * HW_h / (H * 8), 1.0 * HW_w / (W * 8)]

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
                        h, w = [int(ceil(x * y)) for x, y in zip(H_W_factor, used_layer[i][0])]
                        h = h if h > 0 else 1
                        w = w if w > 0 else 1
                        used_layer[i][0] = [h, w]
                    else:
                        # direction, x, y
                        used_layer[i][j][1:] = \
                            [int(floor(x * y)) for x, y in zip(H_W_factor[::-1], used_layer[i][j][1:])]
                if used_layer[i][0] == 0 or used_layer[i][1] == 0:
                    used_layer.pop(i)

            # multi threading
            # is_multi_thred = False if len(used_layer.keys()) < 2 + 8 else True
            # res = []
            # for i in used_layer.keys():
            #     if not isinstance(i, int):
            #         continue
            #     if is_multi_thred:
            #         t = threading.Thread(target=trans_used_layer,
            #                              args=(H_W_factor, used_layer[i]))
            #         res.append(t)
            #     else:
            #         trans_used_layer(H_W_factor, used_layer[i])
            # for t in res:
            #     t.setDaemon(True)
            #     t.start()
            # if is_multi_thred:
            #     res[-1].join()

        # total stride equals 8
        if C == 1:
            gt = cv2.resize(gt, (HW_h / 8, HW_w / 8), interpolation=cv2.INTER_NEAREST)
        else:
            res = []
            for i in range(C):
                res.append((cv2.resize(gt[i], (HW_h / 8, HW_w / 8),
                                       interpolation=cv2.INTER_NEAREST))[np.newaxis])
            gt = np.vstack(res)
        return gt


def trans_used_layer(H_W_factor, this_used_layer):
    for j in range(len(this_used_layer)):
        if j == 0:
            # h, w
            this_used_layer[0] = [int(ceil(x * y)) for x, y in zip(H_W_factor, this_used_layer[0])]
        else:
            # direction, x, y
            this_used_layer[j][1:] = \
                [int(floor(x * y)) for x, y in zip(H_W_factor[::-1], this_used_layer[j][1:])]


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
        gt.append(target)  # torch.unsqueeze(target, 0)
        mask.append(mk[np.newaxis])
        used_layer[keys.format(i)] = use
    img = torch.stack(img, 0)
    gt = torch.stack(gt, 0).unsqueeze(1)  # batch,1,h,w
    mask = np.vstack(mask)  # .unsqueeze(1)
    return img, gt, mask, used_layer


if __name__ == '__main__':
    wf0 = wider_face()
    print 'len of wf:', len(wf0)
    # im, map = wf0[0]  # this is image and ground truth
