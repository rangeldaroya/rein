from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F

class AutoencoderPoint2(nn.Module):
    def __init__(self, num_points = 2500, latent_size=512):
        super(AutoencoderPoint2, self).__init__()
        # num_points = 2500
        point_dim = 3
        # latent_size = 512
        self.conv1 = torch.nn.Conv2d(1, 64, (point_dim,1))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 1)
        # self.conv4 = torch.nn.Conv2d(64, 128, 1)
        self.conv5 = torch.nn.Conv2d(64, latent_size, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(latent_size)

        self.maxpool = nn.MaxPool2d((1,num_points),stride=(2,2))
        self.fc1 = nn.Linear(latent_size,latent_size)
        self.fc2 = nn.Linear(latent_size,latent_size)
        self.fc3 = nn.Linear(latent_size,num_points*3)
        self.bn6 = nn.BatchNorm1d(latent_size)
        self.bn7 = nn.BatchNorm1d(latent_size)
        self.bn8 = nn.BatchNorm1d(num_points*3)



    def forward(self, point_cloud):
        batch_size = point_cloud.shape[0]
        num_points = point_cloud.shape[2]
        point_dim = point_cloud.shape[1]
        # print('batch size: ', batch_size)
        # print('num points: ', num_points)
        # print('point dim: ', point_dim)
        end_points = {}
        input_image = point_cloud.view(point_cloud.shape[0], 1, point_cloud.shape[1], point_cloud.shape[2])
        # print('input_image: ', input_image.shape)
        net = F.relu(self.bn1(self.conv1(input_image)))
        # print('net: ', net.shape)
        net = F.relu(self.bn2(self.conv2(net)))
        # print('net: ', net.shape)
        point_feat = F.relu(self.bn3(self.conv3(net)))
        # print('point_feat: ', point_feat.shape)
        # net = F.relu(self.bn4(self.conv4(point_feat)))
        # print('net: ', net.shape)
        net = F.relu(self.bn5(self.conv5(net)))
        # print('net: ', net.shape)
        global_feat = self.maxpool(net)
        # print('global_feat: ', global_feat.shape)
        net = global_feat.view(batch_size, -1)
        # print('net1: ', net.shape)

        end_points['embedding'] = net

        # net = F.relu(self.bn6(self.fc1(net)))
        # net = F.relu(self.bn7(self.fc2(net)))
        # net = self.bn8(self.fc3(net))

        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        net = self.fc3(net)

        net = net.view(batch_size, 3, num_points)
        # net = F.tanh(net)
        # print('net: ', net.shape)

        return net, end_points


class AutoencoderPoint(nn.Module):
    def __init__(self, num_points = 2500, latent_size=512):
        super(AutoencoderPoint, self).__init__()
        # num_points = 2500
        point_dim = 3
        # latent_size = 512
        self.conv1 = torch.nn.Conv2d(1, 64, (point_dim,1))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 1)
        self.conv5 = torch.nn.Conv2d(128, latent_size, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(latent_size)

        self.maxpool = nn.MaxPool2d((1,num_points),stride=(2,2))
        self.fc1 = nn.Linear(latent_size,latent_size)
        self.fc2 = nn.Linear(latent_size,latent_size)
        self.fc3 = nn.Linear(latent_size,num_points*3)
        self.bn6 = nn.BatchNorm1d(latent_size)
        self.bn7 = nn.BatchNorm1d(latent_size)
        self.bn8 = nn.BatchNorm1d(num_points*3)



    def forward(self, point_cloud):
        batch_size = point_cloud.shape[0]
        num_points = point_cloud.shape[2]
        point_dim = point_cloud.shape[1]
        # print('batch size: ', batch_size)
        # print('num points: ', num_points)
        # print('point dim: ', point_dim)
        end_points = {}
        input_image = point_cloud.view(point_cloud.shape[0], 1, point_cloud.shape[1], point_cloud.shape[2])
        # print('input_image: ', input_image.shape)
        net = F.relu(self.bn1(self.conv1(input_image)))
        # print('net: ', net.shape)
        net = F.relu(self.bn2(self.conv2(net)))
        # print('net: ', net.shape)
        point_feat = F.relu(self.bn3(self.conv3(net)))
        # print('point_feat: ', point_feat.shape)
        net = F.relu(self.bn4(self.conv4(point_feat)))
        # print('net: ', net.shape)
        net = F.relu(self.bn5(self.conv5(net)))
        # print('net: ', net.shape)
        global_feat = self.maxpool(net)
        # print('global_feat: ', global_feat.shape)
        net = global_feat.view(batch_size, -1)
        # print('net1: ', net.shape)

        end_points['embedding'] = net

        # net = F.relu(self.bn6(self.fc1(net)))
        # net = F.relu(self.bn7(self.fc2(net)))
        # net = self.bn8(self.fc3(net))

        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        net = self.fc3(net)

        net = net.view(batch_size, 3, num_points)
        # net = F.tanh(net)
        # print('net: ', net.shape)

        return net, end_points


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))

