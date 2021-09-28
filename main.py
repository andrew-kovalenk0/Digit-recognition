from mnist import MNIST
import numpy as np


def cost_func(a, label, func):
    result = np.empty(a.shape[0])
    if func == 'mse':
        result = (np.square(a - label)).mean(axis=1)
    if func == 'ce':
        pass
    return result


def softmax(Z):
    exp_sum = np.sum(np.exp(Z))
    return np.exp(Z) / exp_sum


def relu(Z):
    return np.maximum(0, Z)


def softmax_backward(Z):
    der = Z.copy()
    der = softmax(der) * (1 - softmax(der))

    return der


class DigitRecognation:
    def __init__(self, train_data, train_labels, loss_label):
        train_data = (train_data - 127.5) / 255.0
        self.train_data = train_data

        train_labels = train_labels[:, np.newaxis]
        buf = np.empty((train_labels.shape[0], 11))
        for i in range(train_labels.shape[0]):
            buf[i] = [0] * 11
            buf[i, train_labels[i]] = 1
        self.train_labels = buf

        self.loss_label = loss_label
        self.w1 = np.random.randint(0, 10, (128, 784))
        self.b1 = np.random.randint(0, 10, 128)
        self.w2 = np.random.randint(0, 10, (11, 128))
        self.b2 = np.random.randint(0, 10, 11)

    def fit(self):
        a1 = np.empty((0, 128))
        z1 = np.empty((0, 128))
        a2 = np.empty((0, 11))
        z2 = np.empty((0, 11))

        for i in range(self.train_data.shape[0]):
            a1 = np.append(a1, [relu(np.dot(self.w1, self.train_data[i]) + self.b1)], axis=0)
            z1 = np.append(z1, [np.dot(self.w1, self.train_data[i]) + self.b1], axis=0)
            a2 = np.append(a2, [softmax(np.dot(self.w2, a1[i]) + self.b2)], axis=0)
            z2 = np.append(z2, [np.dot(self.w2, a1[i]) + self.b2], axis=0)

        cost = cost_func(a2, self.train_labels, self.loss_label)

        s2 = a2 - self.train_labels

    def predict(self):
        pass

if __name__ == '__main__':
    mndata = MNIST(r'data/')
    images, labels = mndata.load_training()
    images, labels = np.array(images), np.array(labels)
    train_images = images[:40000, :]
    train_label = labels[:40000]
    test_images = images[40001:, :]
    test_label = labels[40001:]
    model = DigitRecognation(train_images[0:2, :], train_label[0:2], 'mse')
    model.fit()
