import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np

parser = argparse.ArgumentParser(description="pruning for resnet18 by wpy")
parser.add_argument('--loadfile', '-l', default="", help='Path for checkpoint')
parser.add_argument('--prune', '-p', default=0.5, dest='prune', help='Paramters to be pruned')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net', )
