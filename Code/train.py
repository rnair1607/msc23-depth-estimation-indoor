import time
import argparse
import datetime
import sys
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
import threading
from tqdm import tqdm

from model import Model
# from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize


def main():
    # Arguments
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg