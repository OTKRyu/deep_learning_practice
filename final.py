import sys
import numpy as np
import matplotlib.pyplot as plt
from book.dataset.mnist import load_mnist
from book.common.multi_layer_net_extend import MultiLayerNetExtend
from book.common.util import shuffle_dataset
from book.common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

sys.stdin = open('final_data.txt', 'r')
lr = float(input())
weight_decay = float(input())
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[196,100],
                            output_size=10, weight_decay_lambda=weight_decay, use_dropout=True, use_batchnorm=True)
network.params['W1'] = 