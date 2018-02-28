import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from data_prepare import *
from net import *
from gate_skip_net import gate_skip_net, load_parameter, chunk_gate_net
from total_loss import loss_detection
from data_read import show_feature


def validation():
    epoch = 20  # how many times will be reciur for data
    batch_size = 6  # images in every batch
    # snap_shot = 300
    # batch_iteration = 0
    device_id = 2
    model_place = '../output/my/face_new__{}__model.pth'

    # net = wider_net()
    # batch_iteration = load_saved(net, model_place)  # load those snap shot model parameters
    net = gate_skip_net()
    batch_iteration = load_parameter(net, model_place)

    cudnn.benchmark = True
    net = net.cuda(device_id)
    skip = False  # for part connection of net,stop use tiny face in the beginning

    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-3
    param = get_param(net, lr)
    optimizer = optim.SGD(param, momentum=momentum, weight_decay=weight_decay)

    loss = loss_detection()

    dataset = wider_face('test')
    epoch_size = len(dataset) / batch_size
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                  pin_memory=True, drop_last=True, collate_fn=patch_data)

    # while batch_iteration < (epoch * epoch_size):
    #
    #     epoch_now = batch_iteration / epoch_size
    #     # learing rate decay
    #     if batch_iteration != 0 and (epoch_now) % 5 == 0:
    #         for param_lr in optimizer.param_groups:
    #             param_lr['lr'] /= 10
    iters = 0
    for imgs, gt_heatmaps, mask, used_layer in data_loader:
        t = time.time()
        epoch_now = batch_iteration / epoch_size
        if epoch_now > 5:
            # skip = True
            pass
        if skip:
            set_big(skip)
        img = Variable(imgs, requires_grad=True).cuda(device_id)
        # gt_heatmap = Variable(gt_heatmaps).cuda(device_id)
        # mask = Variable(mask).cuda(device_id)
        # predicted = net(img, skip)  # freeze tiny face
        scores = chunk_gate_net(gt_heatmaps)
        predicted = net(img, scores)
        # & skip
        # if iters % 1000 == 0:  # show the feature map
        for fi in predicted[0]:
            show_feature(fi)
        show_feature(img, True)
        # t00 = time.time()
        # mask = prepare_prediction_with_mask(predicted, mask, used_layer)
        # # print 'with used_layer: ', time.time() - t00
        # predicted = torch.mul(predicted, mask)  # only face will be penase or congrulation
        l = loss(predicted, gt_heatmaps, mask, used_layer)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        # total n
        # tn = reduce(lambda x, y: x * y, predicted[0].size())
        # loss_now = l * tn / (torch.sum(mask) + 1)
        print 'loss is: {:.3f},and iter is {} with time {}'. \
            format(l.data[0], iters, time.time() - t)
        # todo:vis score & save parameters
        iters += 1


if __name__ == "__main__":
    validation()
