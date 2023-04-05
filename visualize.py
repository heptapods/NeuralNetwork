import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

if __name__ =="__main__":
    loss_file = "loss/MNIST_2layer_model_lr=0.5_r=0.001_hiddenlayer=200.txt"
    with open(loss_file, "r") as f:
        train_process = json.loads(f.read())
    train_loss = train_process["train_loss"]
    test_loss = train_process["test_loss"]
    iter = list(range(len(train_loss)))
    plt.plot(iter, train_loss)
    plt.plot(iter, test_loss)
    plt.legend(['train_loss', 'test_loss'])
    plt.title("loss vs iteration")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

    train_accuracy = train_process["train_accuracy"]
    test_accuracy = train_process["test_accuracy"]
    iter = list(range(len(train_accuracy)))
    plt.plot(iter, train_accuracy)
    plt.plot(iter, test_accuracy, '-.', alpha=0.7)
    plt.legend(['train_accuracy', 'test_accuracy'])
    plt.title("accuracy vs iteration")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

    params_file = "results/MNIST_2layer_model_lr=0.5_r=0.001_hiddenlayer=200.txt"
    with open(params_file, "r") as f:
        neural_network_params = json.loads(f.read())
    conn1 = np.array(neural_network_params["conn1"])
    conn2 = np.array(neural_network_params["conn2"])
    bias1 = np.array(neural_network_params["bias1"])
    bias2 = np.array(neural_network_params["bias2"])

    vmax1 = max(np.abs(conn1.max()), np.abs(conn1.min()))
    norm1 = TwoSlopeNorm(vmin=-vmax1, vcenter=0, vmax=vmax1)
    vmax2 = max(np.abs(conn2.max()), np.abs(conn2.min()))
    norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
    cmap = plt.get_cmap('RdBu')
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].set_title('Layer 1 Weights')
    im1 = axs[0].imshow(conn1, cmap=cmap, norm=norm1)
    plt.colorbar(im1, ax=axs[0])

    axs[1].set_title('Layer 2 Weights')
    im2 = axs[1].imshow(conn2, cmap=cmap, norm=norm2)
    plt.colorbar(im2, ax=axs[1])


    plt.show()
