from __future__ import print_function, division

import glob

import matplotlib
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, dataset
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# %matplotlib inline
import random

from torchvision.transforms import transforms

from IPython import display
import matplotlib.pyplot as plt
import torch
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import os

def draw_loss_curve(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

class AlexNet_variety(nn.Module):
    def __init__(self):
        super(AlexNet_variety, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.decay1 = nn.Conv2d(16, 16, 2, stride=2) # 32*32
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decay2 = nn.Conv2d(64, 64, 2, stride=2) # 16*16
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decay3 = nn.Conv2d(64, 64, 2, stride=2) # 8*8
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pre_layers = nn.Sequential(
            nn.Linear(8*8*64, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 3)
        )
    def forward(self, x):
        x = self.block2(self.decay1(self.block1(x)))
        x = self.block4(self.decay3(self.block3(self.decay2(x))))
        x = x.view(x.shape[0], -1)
        return self.pre_layers(x)

def evaluate_net(data_iter, net, criterion):

    psnr_sum, ssni_sum, l, n, iter_num = 0., 0., 0., 0, 0
    for sample in data_iter:
        x, y = sample['fog'], sample['gt']
        pre = net(x)
        l += criterion(pre, y).item()
        pre, gt = np.asarray(pre[0]).transpose((1,2,0)), np.asarray(y[0]).transpose((1,2,0))
        psnr_sum += peak_signal_noise_ratio(gt, pre, data_range=1)
        ssni_sum += structural_similarity(gt, pre, data_range=1, multichannel=True)
        n += len(y)
        iter_num += 1
    return psnr_sum / n, ssni_sum / n, l / iter_num


def trainAddPlot(epochs, trainloader, testloader, net, criterion, optimizer, outdir='', mode=0, plot=True):
    t = time.time()
    train_l, test_l = [], []
    train_acc, test_acc = [], []
    test_epoch, test_psnr, test_ssim = [], [], []
    best_psnr = 0.
    if outdir != '':
        if os.path.exists(outdir) == False:
            os.mkdir(outdir)
    for _ in range(epochs):
        # print(_)
        net = net.cuda()
        net.train()
        train_l_sum, iter_num = 0., 0
        for sample in trainloader:
            if mode == 0:
                x, y = sample['img'], sample['y']
            else:
                x, y = sample['fog'], sample['gt']
            y_hat = net(x.cuda())
            l = criterion(y_hat, y.cuda())
            train_l_sum += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            iter_num += 1
        train_l.append(train_l_sum / iter_num)
        # test every 10 epochs
        if _ % 5 == 4:
            test_epoch.append(_)
            with torch.no_grad():
                # print(train_l_sum)
                net.eval()
                if mode == 0:
                    acc, l = evaluate_net(trainloader, net.cpu(), criterion)
                    train_acc.append(acc)
                    acc, l = evaluate_net(testloader, net.cpu(), criterion)
                    test_acc.append(acc)
                    test_l.append(l)
                else:
                    psnr, ssim, l = evaluate_net(testloader, net.cpu(), criterion, mode=1)
                    test_l.append(l)
                    test_psnr.append(psnr)
                    test_ssim.append(ssim)
                    if psnr > best_psnr:
                        save_path = outdir + '/snapshot_%d.pt' % _
                        torch.save(net.state_dict(), save_path)
    if mode == 0:
        ind = test_acc.index(max(test_acc))
        print('best epoch: test loss %.5f, train_acc %.4f, test_acc %.4f, time: %.3f' % \
              (test_l[ind], train_acc[ind], test_acc[ind], time.time()-t))
        if plot == True:
            draw_loss_curve(range(1, epochs + 1), train_l, 'epochs', 'train_loss')
            draw_loss_curve(test_epoch, test_l, 'epochs', 'test_loss')
            draw_loss_curve(test_epoch, train_acc, 'epochs', 'acc', test_epoch, test_acc, ['train', 'test'])
        return train_l, test_l, train_acc, test_acc
    elif mode == 1:
        ind = test_psnr.index(max(test_psnr))
        print('best epoch: test loss', test_l[ind], 'test_psnr', test_psnr[ind], 'test_ssim', test_ssim[ind])
        if plot == True:
            draw_loss_curve(range(1, epochs + 1), train_l, 'epochs', 'train_loss')
            draw_loss_curve(test_epoch, test_psnr, 'epochs', 'psnr')
            draw_loss_curve(test_epoch, test_ssim, 'epochs', 'ssim')

def get_parameter_number(net):
    # print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    # print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
def resize_img_keep_ratio(img_name, target_size):
    img = cv2.imread(img_name)
    img = img[..., ::-1].copy()

    old_size = img.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    # plt.imshow(img_new)
    # plt.show()
    # exit(0)
    return img_new


def getDataset(bs, num_workers=0):

    trainset = DeFog('./DeFogDataset', (256, 256), is_training=True, transform=True)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_workers)
    testset = DeFog('./DeFogDataset', (256, 256), is_training=False, transform=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return trainloader, testloader


class DeFog(Dataset):
    def __init__(self, root_dir, targetsize, is_training=True, transform=True):
        self.root_dir = root_dir
        self.transform = transform
        self.targetsize= targetsize
        self.imgnames = []
        fogs = os.listdir(root_dir + '/fog')
        fogs.sort()
        seg = len(fogs) * 4 // 5
        if is_training == True:
            self.imgnames = fogs[0:seg]
        else:
            self.imgnames = fogs[seg:]

    def __len__(self):
        return len(self.imgnames)

    def _transform(self, sample):
        img, gt = sample['fog'], sample['gt']
        img, gt = img.astype(np.float32) / 255., gt.astype(np.float32) / 255.
        img, gt = img.transpose((2, 0, 1)), gt.transpose((2, 0, 1))
        return {'fog': torch.from_numpy(img), 'gt':torch.from_numpy(gt), 'name':sample['name']}

    def __getitem__(self, item):
        fog = cv2.imread(os.path.join(self.root_dir+'/fog', self.imgnames[item]))[:,:,::-1]
        gt = cv2.imread(os.path.join(self.root_dir+'/gt', self.imgnames[item]))[:,:,::-1]
        # print(label, img_dir)
        fog = cv2.resize(fog, self.targetsize, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, self.targetsize, interpolation=cv2.INTER_LINEAR)
        sample = {'fog': fog, 'gt': gt, 'name': self.imgnames[item]}

        if self.transform == True:
            sample = self._transform(sample)
        return sample

class ResDilationBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResDilationBlock, self).__init__()
        self.left_1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.left_2 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, padding=2, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, padding=5, stride=1, dilation=5, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.decay = nn.Conv2d(outchannel*2, outchannel, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        left = self.decay(torch.cat([self.left_1(x), self.left_2(x)], dim=1))
        return F.relu(left + self.shortcut(x))

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        return F.relu(self.left(x) + self.shortcut(x))

class Ushape_ResNet(nn.Module):
    def __init__(self, is_dilation=False):
        super(Ushape_ResNet, self).__init__()
        if is_dilation == False:
            self.encode1 = nn.Sequential(
                ResBlock(3, 16),
                ResBlock(16, 16),
                ResBlock(16, 16),
            )
            self.encode2 = nn.Sequential(
                ResBlock(16, 32, stride=2),
                ResBlock(32, 32),
                ResBlock(32, 32),
            )
            self.encode3 = nn.Sequential(
                ResBlock(32, 32, stride=2),
                ResBlock(32, 32),
                ResBlock(32, 32)
            )
        else:
            self.encode1 = nn.Sequential(
                ResDilationBlock(3, 16),
                ResDilationBlock(16, 16),
                ResDilationBlock(16, 16),
            )
            self.encode2 = nn.Sequential(
                ResDilationBlock(16, 32, stride=2),
                ResDilationBlock(32, 32),
                ResDilationBlock(32, 32),
            )
            self.encode3 = nn.Sequential(
                ResDilationBlock(32, 32, stride=2),
                ResDilationBlock(32, 32),
                ResDilationBlock(32, 32)
            )
        self.encode4 = nn.Sequential(
            ResBlock(32, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32)
        )
        self.deconv1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.decode1 = nn.Sequential(
            ResBlock(32+16, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.deconv2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.decode2 = nn.Sequential(
            ResBlock(32+16, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.deconv3 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.decode3 = nn.Sequential(
            ResBlock(16+16, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.pre = nn.Conv2d(32, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        E1 = self.encode1(x)
        E2 = self.encode2(E1)
        E3 = self.encode3(E2)
        E4 = self.encode4(E3)
        D1 = self.decode1(torch.cat([E3, self.deconv1(E4)], dim=1))
        D2 = self.decode2(torch.cat([E2, self.deconv2(D1)], dim=1))
        D3 = self.decode3(torch.cat([E1, self.deconv3(D2)], dim=1))
        return self.sigmoid(self.pre(D3))



def train():
    model = Ushape_ResNet(is_dilation=True)

    root_dir = './DeFogDataset'

    # data_transforms = transforms.Compose([transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    # batch_size = 3
    # epoch = 200
    # lr = 0.01

    # train_dataset = MyDeFog(root_dir)
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    batch_size = 3
    train_loader, testloader = getDataset(batch_size)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # writer = SummaryWriter()
    # criterion = nn.L1Loss()
    # cnt = 0
    # for epoch_ in range(epoch):
    #     for (id, item) in enumerate(train_loader):
    #         images = item['fog']
    #         target = item['gt']
    #         output = model(images)
    #
    #         output_image = output.detach()
    #         output_image = torch.clamp(output_image, 0.0, 1.0)
    #
    #         loss = criterion(output, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.param_groups[0]['lr'] *= 0.99
    #
    #
    #         cnt += 1
    #         writer.add_scalar('Loss/total_loss', loss, cnt)
    #         # writer.add_image('Input/gt', target[0], cnt)
    #         writer.add_image('Input/fog', images[0], cnt)
    #         writer.add_image('Output/output', output_image[0], cnt)
    #         for name, param in model.named_parameters():
    #             name = name.replace('.', '/')
    #             writer.add_histogram(name, param, cnt)
    epoch = 200
    lr = 0.005
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)
    #
    writer = SummaryWriter()
    criterion = nn.MSELoss()
    cnt = 0
    for _ in range(epoch):
        optimizer.param_groups[0]['lr'] *= 0.99
        model.train()
        train_l_sum, iter_num = 0., 0
        for sample in train_loader:
            x, y, name = sample['fog'], sample['gt'], sample['name']
            y_hat = model(x)
            l = criterion(y_hat, y)
            train_l_sum += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            iter_num += 1
            cnt += 1
            # x_image = np.clip(x.detach().numpy()[0], 0., 1.).transpose((1, 2, 0))
            # y_image = np.clip(y_hat.detach().numpy()[0], 0., 1.).transpose((1, 2, 0))

            x_image = x.detach()
            x_image = torch.clamp(x_image, 0.0, 1.0)
            output_image = y_hat.detach()
            output_image = torch.clamp(output_image, 0.0, 1.0)

            writer.add_scalar('Loss/total_loss', l, cnt)
            # writer.add_image('Input/gt', target[0], cnt)
            writer.add_image('Input/fog', x[0], cnt)
            writer.add_image('Output/output', output_image[0], cnt)
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                writer.add_histogram(name, param, cnt)

            # y_im = np.clip(y_hat.numpy()[0], 0., 1.).transpose((1, 2, 0))
            # cv2.imwrite('./DeFogDataset/output' + name[0], y_hat * 255)
        # train_l.append(train_l_sum / iter_num)


if __name__ == '__main__':
    train()
