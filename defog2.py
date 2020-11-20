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
        fog = cv2.imread(os.path.join(self.root_dir+'/fog', self.imgnames[item]))
        gt = cv2.imread(os.path.join(self.root_dir+'/gt', self.imgnames[item]))
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
    model = Ushape_ResNet()

    root_dir = './DeFogDataset'

    # data_transforms = transforms.Compose([transforms.ToTensor()])

    batch_size = 2
    epoch = 1000
    lr = 0.001

    # train_dataset = MyDeFog(root_dir)
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train_loader, testloader = getDataset(batch_size)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.MSELoss()
    writer = SummaryWriter()
    cnt = 0
    for epoch_ in range(epoch):
        for (id, item) in enumerate(train_loader):
            images = item['fog']
            target = item['gt']
            output = model(images)

            output_image = output.detach()
            output_image = torch.clamp(output_image, 0.0, 1.0)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.param_groups[0]['lr'] *= 0.99


            cnt += 1
            writer.add_scalar('Loss/total_loss', loss, cnt)
            writer.add_image('Input/gt', target[0], cnt)
            writer.add_image('Input/fog', images[0], cnt)
            writer.add_image('Output/output', output_image[0], cnt)
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                writer.add_histogram(name, param, cnt)


if __name__ == '__main__':
    train()
