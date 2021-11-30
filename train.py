import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from data import *
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

# Parameter
tsk = 'Task 2'
EPOCH_MAX = 30
block = 'LSTM'
optimizer = 'Adam'
dropout = 0
latent_length = 30
batch_size = 309  # 309  340
input_size = 640
hidden1 = 128
hidden2 = 128
hidden3 = 64
hidden4 = 64
learning_rate = 0.00001

device = 'cpu'
data_path_dev = '/data/dev/'
# data_path_additional = '/home/share/dataset/DCASE2020/Dcase2020_task2/data/additional/'
data_path_eval = '/data/eval/'

def init_layer()