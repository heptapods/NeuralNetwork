# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import json
import struct


from models import MyTwoLayerNeuralNetwork, MyNeuralNetworkTrainer, read_idx

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 读取图像数据
    train_images = read_idx('MNIST/train-images.idx3-ubyte')
    test_images = read_idx('MNIST/t10k-images.idx3-ubyte')

    # 读取标签数据
    train_labels = read_idx('MNIST/train-labels.idx1-ubyte')
    test_labels = read_idx('MNIST/t10k-labels.idx1-ubyte')
    train_images = np.array(train_images)/255
    test_images = np.array(test_images)/255
    I = np.eye(10)
    train_out = np.array([I[x,:] for x in train_labels])
    test_out = np.array([I[x,:] for x in test_labels])
    in_size = len(train_images[0])
    # hidden_size 为隐藏层节点数，learning_rate 为学习率，r为正则化系数
    # hidden_size，learning_rate，r 这三个超参数可以手动调整
    out_size = 10
    hidden_size = 50
    nn = MyTwoLayerNeuralNetwork(in_size, out_size, hidden_size)
    learning_rate = 0.1
    r=0.001
    # 构造训练器 trainer，初始化trainer，然后训练。
    trainer = MyNeuralNetworkTrainer(nn, train_images, train_out, len(train_images), test_images, test_out, 100, learning_rate=learning_rate, r=r)
    trainer.neuralNetworkInitialize(0,0.1)
    # iter_max 为最大迭代次数，可以修改以提高训练速度。
    trainer.train( train_labels=train_labels, test_labels=test_labels,
                   iter_max=1000, model_file=f"results/MNIST_2layer_model_lr={learning_rate}_r={r}_hiddenlayer={hidden_size}.txt",
                  loss_file=f"loss/MNIST_2layer_model_lr={learning_rate}_r={r}_hiddenlayer={hidden_size}.txt")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
