import os
import torch.nn as nn
from torch.nn import functional as F
import torch
import datetime
import logging
import shutil
import numpy as np



class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=4, epsilon=0.14):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def set_logging_config(logdir):
    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()


    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.Formatter.converter = beijing

    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, ("log.txt"))),
                                  logging.StreamHandler(os.sys.stdout)])


class SE_Block(nn.Module):
    def __init__(self, inchannel):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel),
            nn.ReLU(),
            nn.Linear(inchannel, inchannel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def normalize(w):
    if len(w.shape) == 4:
        d = torch.sum(torch.abs(w), dim=3)
    elif len(w.shape) == 3:
        d = torch.sum(torch.abs(w), dim=2)

    d_re = 1 / torch.sqrt(d + 1e-5)
    d_re[d_re == float('inf')] = 0
    d_matrix = torch.diag_embed(d_re)

    return torch.matmul(torch.matmul(d_matrix, w), d_matrix)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=62, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(self.avg_pool(x).size())
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # print(out.size())
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels=62, ratio=5):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_channels, ratio)
        self.SpatialAttention = SpatialAttention()

    def forward(self, x):
        x = self.ChannelAttention(x) * x
        x = self.SpatialAttention(x) * x
        return x

def feature_trans(subgraph_num, feature):
    if subgraph_num == 7:
        return feature_trans_7(feature)

def feature_trans_7(feature):
    reassigned_feature = torch.cat((
        feature[:, 0:5],

        feature[:, 5:8], feature[:, 14:17], feature[:, 23:26],

        feature[:, 23:26], feature[:, 32:35], feature[:, 41:44],

        feature[:, 7:12], feature[:, 16:21], feature[:, 25:30],
        feature[:, 34:39], feature[:, 43:48],

        feature[:, 11:14], feature[:, 20:23], feature[:, 29:32],

        feature[:, 29:32], feature[:, 38:41], feature[:, 47:50],

        feature[:, 50:62]), dim=1)

    return reassigned_feature