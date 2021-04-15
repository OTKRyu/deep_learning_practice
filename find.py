import sys
import numpy as np
import matplotlib.pyplot as plt
from book.dataset.mnist import load_mnist
from book.common.multi_layer_net_extend import MultiLayerNetExtend
from book.common.util import shuffle_dataset
from book.common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[196,100],
                            output_size=10, weight_decay_lambda=weight_decay, use_dropout=True, use_batchnorm=True)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100
sys.stdin = open('find_data.txt', 'r')
l_lr, s_lr = map(int, input().split())
if l_lr - s_lr <= 0.01:
    lr_flag = 0
else:
    lr_flag = 1
l_wd, s_wd = map(int, input().split())
if l_wd-s_wd <= 0.01:
    wd_flag = 0
else:
    wd_flag = 1
best_acc = 0
if lr_flag or wd_flag:
    for i in range(optimization_trial):
        # 탐색한 하이퍼파라미터의 범위 지정===============
        weight_decay = np.random.uniform(s_wd, l_lr)
        lr = np.random.uniform(s_lr, l_lr)
        # ================================================
        val_acc_list, train_acc_list = __train(lr, weight_decay)
        if best_acc < val_acc_list[-1]:
            best_acc = val_acc_list[-1]
            best_lr = lr
            best_wd = weight_decay
    if ((l_lr-s_lr)/10)) <= 0.005 and (l_wd-s_wd)/10) <= 0.005:
        sys.stdout = open('final_data.txt', 'w')
        print(2)
        print(f'{best_lr}')
        print(f'{best_wd}')
    else:
        sys.stdout = open('find_data.txt', 'w')
        print(f'{max(0.00000000000001,best_lr-((l_lr-s_lr)/10))} {best_lr+((l_lr-s_lr)/10)}')
        print(f'{max(0.00000000000001,best_wd-((l_wd-s_wd)/10))} {best_wd+((l_wd-s_wd)/10)}')
else:
    print('hyper parameters are already good enough')
