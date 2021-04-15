import sys
import numpy as np
import matplotlib.pyplot as plt
from book.dataset.mnist import load_mnist
from book.common.multi_layer_net_extend import MultiLayerNetExtend
from book.common.util import shuffle_dataset
from book.common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[196,100],
                            output_size=10, weight_decay_lambda=weight_decay, use_dropout=True, use_batchnorm=True)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list

sys.stdin = open('fianl_data.txt', 'r')
n = int(input())
if n==0:
    print('you need better hyper parameters. go back to step that find hyper parameters')
else:
    lr = float(input())
    weight_decay = float(input())
    # ================================================
    final_acc = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    plt.plot(fianl_acc)
    plt.show() 