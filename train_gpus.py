import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from data_prepare import *
from net import *
from total_loss import loss_detection
import visdom


class train_parallel_module(nn.Module):
    """
    the net was used for DataParallel(module)
    """

    def __init__(self, net, optimizer, loss):
        super(train_parallel_module, self).__init__()
        self.net = net
        self.optim = optimizer
        self.loss = loss

    def forward(self, img, gt_heatmap):
        img.requires_grad = True
        predicted = self.net(img)
        ll = self.loss(gt_heatmap, predicted)

        # compute gradient and do SGD step
        self.optim.zero_grad()  # .module
        ll.backward()
        self.optim.step()  # ..module
        return ll


def train_with_parallel():
    epoch = 20  # how many times will be reciur for data
    batch_size = 12  # images in every batch
    # snap_shot = 300
    # batch_iteration = 0
    device_ids = [5, 6]
    model_place = '../output/my_gpus/face__{}__model.pth'

    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-3

    # net
    net = wider_net()
    batch_iteration = load_saved(net, model_place)  # load those snap shot model parameters
    cudnn.benchmark = True
    net = net.cuda(device_ids[0])

    # optimizer
    param = get_param(net, lr)
    optimizer = optim.SGD(param, momentum=momentum, weight_decay=weight_decay)

    # loss
    loss = loss_detection()

    # data
    dataset = wider_face()
    epoch_size = len(dataset) / batch_size
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3,
                                  pin_memory=True, drop_last=True, collate_fn=patch_data)

    # module for DataParallel
    module = train_parallel_module(net, optimizer, loss)
    module_parallel = nn.DataParallel(module, device_ids=device_ids)

    while batch_iteration < (epoch * epoch_size):
        t = time.time()
        # learing rate decay
        if batch_iteration != 0 and (batch_iteration / epoch_size) % 5 == 0:
            for param_lr in optimizer.param_groups:  # .module
                param_lr['lr'] /= 10

        for imgs, gt_heatmaps in data_loader:
            imgs = Variable(imgs.cuda(device_ids[0]))  # device_ids[0], requires_grad=True
            gt_heatmaps = Variable(gt_heatmaps.cuda(device_ids[0]))
            l = module_parallel(imgs, gt_heatmaps)
            print 'loss is: {:.3f},and iter is {} with time {}'. \
                format(l.data[0], batch_iteration, time.time() - t)
            t = time.time()
            # todo:vis score & save parameters
            batch_iteration += 1
        print 'epoch {} is done!'.format(batch_iteration / epoch_size)
        # snapshot
        net_dict = {'iter': batch_iteration, 'net_state': net.state_dict()}
        torch.save(net_dict, model_place.format(batch_iteration))


if __name__ == "__main__":
    train_with_parallel()
