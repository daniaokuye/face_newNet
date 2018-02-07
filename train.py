import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from data_prepare import *
from net import *
from total_loss import loss_detection
from data_read import show_feature
from cfg import set_big


def train():
    epoch = 20  # how many times will be reciur for data
    batch_size = 5  # images in every batch
    # snap_shot = 300
    # batch_iteration = 0
    device_id = 6  # 6 or 7
    model_place = '../output/my/face_new__{}__model.pth'

    net = wider_net()
    batch_iteration = load_saved(net, model_place)  # load those snap shot model parameters
    cudnn.benchmark = True
    net = net.cuda(device_id)
    use_skip = True  # for part connection of net,stop use tiny face in the beginning

    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-3
    param = get_param(net, lr)
    optimizer = optim.SGD(param, momentum=momentum, weight_decay=weight_decay)

    loss = loss_detection()

    dataset = wider_face()
    epoch_size = len(dataset) / batch_size
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                  pin_memory=True, drop_last=True, collate_fn=patch_data)

    while batch_iteration < (epoch * epoch_size):
        t = time.time()
        epoch_now = batch_iteration / epoch_size
        # if batch_iteration > 0:
        #     use_skip = True
        # if epoch_now > 5:
        #     use_skip = True
        if use_skip:
            set_big(use_skip)
        # learing rate decay
        if batch_iteration != 0 and (epoch_now) % 3 == 0:
            for param_lr in optimizer.param_groups:
                param_lr['lr'] /= 10

        for imgs, gt_heatmaps, mask, used_layer in data_loader:
            img = Variable(imgs, requires_grad=True).cuda(device_id)
            gt_heatmap = Variable(gt_heatmaps).cuda(device_id)
            # mask = Variable(mask).cuda(device_id)

            predicted = net(img, use_skip)  # freeze tiny face
            # & skip
            # if batch_iteration % 1000 == 0:  # show the feature map
            #     show_feature(predicted)

            mask = prepare_prediction_with_mask(predicted, mask, used_layer)
            predicted = torch.mul(predicted, mask)  # only face will be penase or congrulation

            l = loss(gt_heatmap, predicted)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # total n
            tn = reduce(lambda x, y: x * y, predicted.size())
            loss_now = l * tn / (torch.sum(mask) + 1)
            print 'loss is: {:.4f},and iter is {} with time {}'. \
                format(loss_now.data[0], batch_iteration, time.time() - t)
            t = time.time()
            # todo:vis score & save parameters
            batch_iteration += 1
        print 'epoch {} is done!'.format(batch_iteration / epoch_size)
        # snapshot
        net_dict = {'iter': batch_iteration, 'net_state': net.state_dict()}
        torch.save(net_dict, model_place.format(batch_iteration))


if __name__ == "__main__":
    train()
