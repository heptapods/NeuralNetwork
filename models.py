import numpy as np
import json
import struct


class MyTwoLayerNeuralNetwork:
    def __init__(self, in_size, out_size, hidden_size):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.input = np.zeros((1, self.in_size))
        self.output_net = np.zeros((1, self.out_size))
        self.output_out = np.zeros((1, self.out_size))
        self.hidden_net = np.zeros((1, self.hidden_size))
        self.hidden_out = np.zeros((1, self.hidden_size))
        self.conn1 = np.zeros((self.in_size, self.hidden_size))
        self.conn2 = np.zeros((self.hidden_size, self.out_size))
        self.bias1 = np.zeros((1,hidden_size))
        self.bias2 = np.zeros((1,out_size))

    def __str__(self):
        res = {}
        res["in_size"] = self.in_size
        res["out_size"] = self.out_size
        res["hidden_size"] = self.hidden_size
        res["input"] = self.input
        res["hidden"] = self.hidden_out
        res["output"] = self.output_out
        res["conn1"] = self.conn1
        res["conn2"] = self.conn2
        res["bias1"] = self.bias1
        res["bias2"] = self.bias2
        return str(res)

    def setParameter(self, **kwargs):
        self.input = kwargs.get("input", self.input)
        self.conn1 = kwargs.get("conn1", self.conn1)
        self.conn2 = kwargs.get("conn2", self.conn2)
        self.bias1 = kwargs.get("bias1", self.bias1)
        self.bias2 = kwargs.get("bias2", self.bias2)

    def forwardPass(self):
        self.hidden_net = self.input.dot(self.conn1) + self.bias1
        self.hidden_out = 1/(1+np.exp(-self.hidden_net))
        self.output_net = self.hidden_out.dot(self.conn2) + self.bias2
        self.output_out = 1/(1+np.exp(-self.output_net))

    def backwardPass(self, real):
        partial_loss__partial_output_out = -(real - self.output_out)
        partial_output_out__partial_output_net = self.output_out * (1-self.output_out)
        partial_output_net__partial_conn2 = self.hidden_out.reshape((-1,1))
        delta_conn2 = partial_loss__partial_output_out * partial_output_out__partial_output_net * partial_output_net__partial_conn2
        delta_bias2 = partial_loss__partial_output_out * partial_output_out__partial_output_net
        partial_output_net__partial_hidden_out = self.conn2.transpose()
        partial_loss__partial_hidden_out = (partial_loss__partial_output_out * partial_output_out__partial_output_net).dot(
            partial_output_net__partial_hidden_out
        )
        partial_hidden_out__partial_hidden_net = self.hidden_out * (1-self.hidden_out)
        partial_hidden_net__partial_conn1 = self.input.reshape((-1,1))
        delta_conn1 = partial_loss__partial_hidden_out * partial_hidden_out__partial_hidden_net * partial_hidden_net__partial_conn1
        delta_bias1 = partial_loss__partial_hidden_out * partial_hidden_out__partial_hidden_net
        return {
            "delta_conn2": delta_conn2,
            "delta_conn1": delta_conn1,
            "delta_bias1": delta_bias1,
            "delta_bias2": delta_bias2
        }

    def save(self, file_name):
        neural_network_params = {
            "conn1": self.conn1.tolist(),
            "conn2": self.conn2.tolist(),
            "bias1": self.bias1.tolist(),
            "bias2": self.bias2.tolist()
        }
        with open(file_name, "w+") as f:
            f.write(json.dumps(neural_network_params))

    def load(self, file_name):
        with open(file_name, "r") as f:
            neural_network_params = json.loads(f.read())
            self.setParameter(
                conn1=np.array(neural_network_params["conn1"]),
                conn2=np.array(neural_network_params["conn2"]),
                bias1=np.array(neural_network_params["bias1"]),
                bias2=np.array(neural_network_params["bias2"])
            )


class MyNeuralNetworkTrainer():
    def __init__(self, nn:MyTwoLayerNeuralNetwork, train_input, train_output, train_size, test_input, test_output, batch_size, learning_rate=0.2, r=0.1):
        self._nn = nn
        self._train_input = train_input
        self._train_output = train_output
        self._test_input = test_input
        self._test_output = test_output
        self._train_size = train_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._r = r

    def neuralNetworkInitialize(self, ave, sigma2):
        self._nn.setParameter(
            conn1=np.random.normal(ave, sigma2, (self._nn.in_size, self._nn.hidden_size)),
            conn2=np.random.normal(ave, sigma2, (self._nn.hidden_size, self._nn.out_size)),
            bias1=np.zeros((1, self._nn.hidden_size)),
            bias2=np.zeros((1, self._nn.out_size))
        )

    # loss = 1/2 * |nn.output - real_output|^2 + 1/2 * r * |nn.parameters|^2
    def getTotalLoss(self, on):
        total_loss = 0
        data_range = zip(self._train_input, self._train_output) if on == "train" else zip(self._test_input, self._test_output)
        for input_data, output_data in data_range:
            self._nn.setParameter(input=input_data)
            self._nn.forwardPass()
            loss = 1/2 * np.sum((output_data - self._nn.output_out) ** 2) + 1/2 * self._r * (
                    np.sum(self._nn.conn1 ** 2) + np.sum(self._nn.conn2 ** 2) + np.sum(self._nn.bias1 ** 2) +
                    np.sum(self._nn.bias2 ** 2))
            total_loss += loss
        return total_loss

    def getAccuracy(self, input, labels):
        correct = 0
        for input_data, output_data in zip(input, labels):
            self._nn.setParameter(input=input_data)
            self._nn.forwardPass()
            if np.argmax(self._nn.output_out)==output_data:
                correct += 1
        return correct/len(labels)

    def SGD(self):
        for input_data, output_data in zip(self._train_input, self._train_output):
            self._nn.setParameter(input=input_data)
            self._nn.forwardPass()
            delta = self._nn.backwardPass(output_data)
            self._nn.conn1 -=  self._learning_rate * (delta["delta_conn1"] + self._r * self._nn.conn1)
            self._nn.conn2 -=  self._learning_rate * (delta["delta_conn2"] + self._r * self._nn.conn2)
            self._nn.bias1 -= self._learning_rate *delta["delta_bias1"]
            self._nn.bias2 -= self._learning_rate * delta["delta_bias2"]

    def batchSGD(self, batch_input, batch_output, r):
        total_delta = {
            "delta_conn1": np.zeros((self._nn.in_size, self._nn.hidden_size)),
            "delta_conn2": np.zeros((self._nn.hidden_size, self._nn.out_size)),
            "delta_bias1": np.zeros((1, self._nn.hidden_size)),
            "delta_bias2": np.zeros((1, self._nn.out_size))
        }
        for input_data, output_data in zip(batch_input, batch_output):
            self._nn.setParameter(input=input_data)
            self._nn.forwardPass()
            delta = self._nn.backwardPass(output_data)
            total_delta["delta_conn1"] += delta["delta_conn1"] + r * self._nn.conn1
            total_delta["delta_conn2"] += delta["delta_conn2"] + r * self._nn.conn2
            total_delta["delta_bias1"] += delta["delta_bias1"] + r * self._nn.bias1
            total_delta["delta_bias2"] += delta["delta_bias2"] + r * self._nn.bias2
        total_delta["delta_conn1"] /= self._batch_size
        total_delta["delta_conn2"] /= self._batch_size
        total_delta["delta_bias1"] /= self._batch_size
        total_delta["delta_bias2"] /= self._batch_size
        self._nn.conn1 -= self._learning_rate * total_delta["delta_conn1"]
        self._nn.conn2 -= self._learning_rate * total_delta["delta_conn2"]
        self._nn.bias1 -= self._learning_rate * total_delta["delta_bias1"]
        self._nn.bias2 -= self._learning_rate * total_delta["delta_bias2"]


    def train(self, train_labels, test_labels, iter_max = 100000,  model_file="MyNeuralNetwork.txt", loss_file = "loss.txt"):
        iter = 0
        beta = 0.9
        r = self._r
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        while iter<iter_max:
            l1 = self.getTotalLoss(on="train")
            l2 = self.getTotalLoss(on="test")
            a1 = self.getAccuracy(self._train_input, train_labels)
            a2 = self.getAccuracy(self._test_input, test_labels)
            print(iter, l1, l2, a1, a2)
            train_loss.append(l1)
            test_loss.append(l2)
            train_accuracy.append(a1)
            test_accuracy.append(a2)
            index = np.random.choice(self._train_size, self._batch_size)
            batch_input = self._train_input[index,:]
            batch_output = self._train_output[index,:]
            self.batchSGD(batch_input, batch_output, r)
            iter += 1
            if iter%(int(iter_max/10)+1) == 0:
                r *= beta
        self._nn.save(model_file)
        with open(loss_file, "w+") as f:
            f.write(json.dumps({
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            }))



def read_idx(filename):
    with open(filename, 'rb') as f:
        # 读取文件头
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number == 2051:
            data_type = 'image'
        elif magic_number == 2049:
            data_type = 'label'
        else:
            raise ValueError('Invalid magic number')

        num_items = struct.unpack('>I', f.read(4))[0]
        if data_type == 'image':
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            data = []
            for i in range(num_items):
                # 读取图像数据
                img_data = []
                for j in range(rows * cols):
                    pixel = struct.unpack('>B', f.read(1))[0]
                    img_data.append(pixel)
                data.append(img_data)
        elif data_type == 'label':
            data = [struct.unpack('>B', f.read(1))[0] for i in range(num_items)]

        return data