import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from data_prepare import *
from net import *
import visdom


def train():
    epoch = 20  # how many times will be reciur for data
    batch_size = 2  # images in every batch
    snap_shot = 300
    batch_iteration = 0
    model_place = '../output/face__{}__model.pth'
    dataset = wider_face()
    epoch_size = len(dataset) / batch_size
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, collate_fn=patch_data)

    net = wider_net()
    now_iter = load_saved(net, model_place)  # load those snap shot model parameters

    cudnn.benchmark = True
    net.cuda()

    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-3
    param = get_param(net, lr)
    optimizer = optim.SGD(param, momentum=momentum, weight_decay=weight_decay)
    loss = nn.MSELoss()
    vis = visdom.Visdom()

    while batch_iteration < xrange(epoch * epoch_size):
        t = time.time()
        # define batch_iteration
        if batch_iteration == 0:
            batch_iteration = now_iter
        # batch_iteration += 1
        # todo:learing rate decay
        for img, gt in data_loader:
            img = Variable(img, requires_grad=True).cuda()
            out = net(img, vis)

            l = loss(out, gt)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # todo:vis score & save parameters
            batch_iteration += 1
        net_dict = {'iter': batch_iteration, 'net_state': net.state_dict()}
        torch.save(net_dict, model_place.format(batch_iteration))


if __name__ == "__main__":
    train()
