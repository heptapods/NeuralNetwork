#项目名称
这是一个使用numpy手写的两层全连接神经网络，未使用任何机器学习框架，利用`MNIST`手写数据集进行训练和测试。
##项目结构
```
.
├── MNIST
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
├── __pycache__
│   └── models.cpython-39.pyc
├── loss
│   ├── MNIST_2layer_model_lr=0.1_r=0.001_hiddenlayer=200.txt
│   ├── MNIST_2layer_model_lr=0.1_r=0.001_hiddenlayer=50.txt
│   ├── MNIST_2layer_model_lr=0.5_r=0.001_hiddenlayer=200.txt
│   ├── MNIST_2layer_model_lr=0.5_r=0.001_hiddenlayer=50.txt
│   ├── MNIST_2layer_model_lr=0.5_r=0.01_hiddenlayer=200.txt
│   └── MNIST_2layer_model_lr=0.5_r=0.01_hiddenlayer=50.txt
├── models.py
├── results
│   ├── MNIST_2layer_model_lr=0.1_r=0.001_hiddenlayer=200.txt
│   ├── MNIST_2layer_model_lr=0.1_r=0.001_hiddenlayer=50.txt
│   ├── MNIST_2layer_model_lr=0.2_r=0.002_hiddenlayer=100.txt
│   ├── MNIST_2layer_model_lr=0.5_r=0.001_hiddenlayer=200.txt
│   ├── MNIST_2layer_model_lr=0.5_r=0.001_hiddenlayer=50.txt
│   ├── MNIST_2layer_model_lr=0.5_r=0.01_hiddenlayer=200.txt
│   └── MNIST_2layer_model_lr=0.5_r=0.01_hiddenlayer=50.txt
├── test_model.py
├── train_model.py
└── visualize.py
```
`MNIST`文件下是手写数字的训练和测试样本,该文件太大了无法上传至GitHub，使用期需自行下载 `MNIST`数据集并放在根目录下。

`results`目录下为训练好的神经网络的模型参数。`loss`目录下保存了神经网络的训练过程。可以通过 `json.loads()`读取`results`或`loss`目录下的文件。
###代码文件包括以下内容：
+   `models.py` 包含两个类。`MyTwoLayerNeuralNetwork`是一个2层全链接神经网络类，提供了`forwardPass`（正向传递）和 `backwardPass`（反向传递）方法。`MyNeuralNetworkTrainer`是一个为上述网络提供的训练器采用，损失函数采用平方误差函数加上L2正则化，使用mini-batch随机梯度下降。

+   `train_model.py` 提供了训练神经网络的代码，训练完成后的网络参数会保存在`results`目录下，训练过程会保存在`loss`目录下。

+   `test_model.py`提供了测试神经网络的代码，会打印出神经网络在训练集与测试集上的准确率。
    
+   `visualize.py`提供了训练过程和网络参数可视化代码。

##使用方法
+   若想直接使用该网络可以不去修改`models.py`。

+   用户可以在`train.py`中导入自己的训练集与测试集，文件中的`hidden_size` 为隐藏层节点数，`learning_rate` 为学习率，`r`为正则化系数，这三个超参数可以手动调整。利用 `MyNeuralNetworkTrainer` 类提供的方法（主要使用是构造函数，`neuralNetworkInitialize` 初始化网络参数，和`train`随机梯度下降进行迭代）进行训练即可。建议将训练完成后的网络参数保存在`results`目录下，训练过程会保存在`loss`目录下。

+   用户可以在`test_model.py`中测试训练好的模型。测试模型时，需要根据模型设置对应的隐藏层节点数 `hidden_size`，训练好的模型保存在 `results` 文件下，用 `MyTwoLayerNeuralNetwork` 类下的 `load` 方法导入保存的网络参数即可构建相应的网络进行测试。提供的默认代码会输出模型在`MNIST`数据集上的准确率。

+   用户可以在`visualize.py`中可视化模型训练过程与网络参数，只需修改网络对应的文件即可。

##已训练的模型

results目录下提供了训练好的6个模型，均是在MNIST手写数字样本上训练和测试的，它们分别使用了不同的隐藏层节点数、正则化强度和学习率，可以为用户提供参考。这些模型可以在test_model.py中直接使用。
