# this fucntion together with newnet.py
# are used for loading data & ground truth
import torch, cv2
import torch.utils.data as data
import torchvision.transforms as transform
from data_read import combined_roidb, obtain_data, jitter
from cfg import *


class wider_face(data.Dataset):
    def __init__(self):
        self.imdb, self.image_index = combined_roidb()

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, item):
        image, gt_map = obtain_data(self.imdb, item)
        image = self.trans_img(image)  # .transpose((2, 0, 1))
        gt_map = self.trans_gt(gt_map)
        return image, gt_map

    def trans_img(self, data):
        trans = transform.Compose([
            transform.ToPILImage(),
            transform.RandomHorizontalFlip(),
            transform.Lambda(jitter),
            transform.Resize((HW_h, HW_w)),
            transform.ToTensor(),
            transform.Normalize(mean=Norm_mean, std=Norm_std)
        ])
        data = trans(data)
        # show_data(data)
        return data

    def trans_gt(self, gt):
        gt = cv2.resize(gt, (HW_h, HW_w), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(gt)


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
    img, gt = [], []
    for input, target in batch:
        img.append(input)
        gt.append(target)  # torch.unsqueeze(target, 0)
    img = torch.stack(img, 0)
    gt = torch.stack(gt, 0)
    return img, gt


if __name__ == '__main__':
    wf0 = wider_face()
    print 'len of wf:', len(wf0)
    im, map = wf0[0]  # this is image and ground truth
