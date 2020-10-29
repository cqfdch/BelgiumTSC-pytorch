from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        #conv unit
        self.conv_3_256 = BasicBlock(in_planes=3, planes=256)
        self.conv_256_64 = BasicBlock(in_planes=256, planes=64)
        self.conv_64_64 = BasicBlock(in_planes=64, planes=64)
        self.conv_64_256 = BasicBlock(in_planes=64, planes=256)
        self.conv_256_128 = BasicBlock(in_planes=256, planes=128)

        #linear unit
        self.linear = nn.Linear(2048, 62)


    def forward(self, x):
        out_0 = self.conv_3_256(x)
        out_1 = self.conv_256_64(out_0)
        out_2 = self.conv_64_64(out_1)
        out_3 = self.conv_64_64(out_2)
        out_4 = F.avg_pool2d(out_3, 2)
        out_5 = self.conv_64_256(out_4)
        out_6 = F.max_pool2d(out_5, 2)
        out_7 = self.conv_256_128(out_6)
        out_8 = F.max_pool2d(out_7, 2)
        out = out_8

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
