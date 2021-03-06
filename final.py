import sys
import numpy as np
import matplotlib.pyplot as plt
from book.dataset.mnist import load_mnist
from book.common.multi_layer_net_extend import MultiLayerNetExtend
from book.common.util import shuffle_dataset


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)


network = MultiLayerNetExtend(input_size=784, hidden_size_list=[196,100,100,100],
                            output_size=10, use_dropout=True, use_batchnorm=False)
datas = np.load('final_data.npz')
W1 = datas['W1']
b1 = datas['b1']
W2 = datas['W2']
b2 = datas['b2']
W3 = datas['W3']
b3 = datas['b3']
W4 = datas['W4']
b4 = datas['b4']
W5 = datas['W5']
b5 = datas['b5']
network.params['W1'] = W1
network.params['b1'] = b1
network.params['W2'] = W2
network.params['b2'] = b2
network.params['W3'] = W3
network.params['b3'] = b3
network.params['W4'] = W4
network.params['b4'] = b4
network.params['W5'] = W5
network.params['b5'] = b5

print(network.accuracy(x_test,t_test))
    