import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from data_prepare import *
from net import *
from total_loss import loss_detection
import visdom


def train_test():
    epoch = 20  # how many times will be reciur for data
    batch_size = 9  # images in every batch
    # snap_shot = 300
    # batch_iteration = 0
    device_ids = [4, 5, 6]
    model_place = '../output/my_gpus/face_new__{}__model.pth'
    dataset = wider_face()
    epoch_size = len(dataset) / batch_size
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                  pin_memory=True, drop_last=True, collate_fn=patch_data)

    net = wider_net()
    batch_iteration = load_saved(net, model_place)  # load those snap shot model parameters
    cudnn.benchmark = True
    net = nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda(device_ids[0])

    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-3
    param = get_param(net, lr)
    optimizer = optim.SGD(param, momentum=momentum, weight_decay=weight_decay)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    loss = loss_detection()

    while batch_iteration < (epoch * epoch_size):
        t = time.time()
        # learing rate decay
        if batch_iteration != 0 and (batch_iteration / epoch_size) % 5 == 0:
            for param_lr in optimizer.param_groups:
                param_lr['lr'] /= 2

        for imgs, gt_heatmaps in data_loader:
            img = Variable(imgs, requires_grad=True).cuda(device_ids[0])
            gt_heatmap = Variable(gt_heatmaps).cuda(device_ids[0])

            predicted = net(img)
            l = loss(gt_heatmap, predicted)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
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
    train_test()
