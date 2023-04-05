import numpy as np
import json
import struct


from models import MyTwoLayerNeuralNetwork, MyNeuralNetworkTrainer, read_idx

if __name__=="__main__":
    train_images = read_idx('MNIST/train-images.idx3-ubyte')
    train_labels = read_idx('MNIST/train-labels.idx1-ubyte')
    train_images = np.array(train_images) / 255
    test_images = read_idx('MNIST/t10k-images.idx3-ubyte')
    test_images = np.array(test_images) / 255
    test_labels = read_idx('MNIST/t10k-labels.idx1-ubyte')
    in_size = len(test_images[0])

    # 测试模型时，需要根据模型设置对应的隐藏层节点数 hidden_size
    # 训练好的模型保存在 results 文件下，用 MyTwoLayerNeuralNetwork 类下的 load 方法即可构建相应的网络进行测试。
    out_size = 10
    hidden_size = 50
    nn = MyTwoLayerNeuralNetwork(in_size, out_size, hidden_size)
    nn.load("results/MNIST_2layer_model_lr=0.5_r=0.001_hiddenlayer=50.txt")
    # print(nn)

    wrong = 0
    for train_in, train_out in zip(train_images, train_labels):
        nn.setParameter(input=train_in)
        nn.forwardPass()
        output = nn.output_out
        out_label = np.argmax(output)
        if not out_label == train_out:
            wrong += 1
    print("Accuracy in train set: ",1- wrong / len(train_labels))

    wrong = 0
    for test_in, test_out in zip(test_images, test_labels):
        nn.setParameter(input=test_in)
        nn.forwardPass()
        output = nn.output_out
        out_label = np.argmax(output)
        if not out_label == test_out:
            wrong += 1
    print("Accuracy in test set: ",1- wrong / len(test_labels))
