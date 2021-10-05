import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np


def loss(yp, y, func):
    return globals()[func](yp, y)


def mse(yp, y):
    return np.square(yp - y).mean()


def ce(yp, y):
    predictions = np.clip(yp, 1e-12, 1. - 1e-12)
    return -np.sum(y * np.log(predictions + 1e-9)) / predictions.shape[0]


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


def relu(z):
    return np.maximum(z, 0)


def relu_backward(z):
    return z > 0


class DigitRecognation:
    def __init__(self, train_data, train_labels, loss_label):
        train_data = train_data.T / 255
        self.train_data = train_data

        train_labels = train_labels[:, np.newaxis]
        buf = np.empty((train_labels.shape[0], 11))
        for i in range(train_labels.shape[0]):
            buf[i] = [0] * 11
            buf[i, train_labels[i]] = 1
        self.train_labels = buf.T

        self.loss_label = loss_label
        self.w1 = np.random.random((128, 784)) - 0.5
        self.b1 = np.random.random((128, 1)) - 0.5
        self.w2 = np.random.random((11, 128)) - 0.5
        self.b2 = np.random.random((11, 1)) - 0.5

    def fit(self, iterations):
        m = self.train_data.shape[0]
        y = self.train_labels
        alpha = 0.0005
        error = []
        momentum = 0.9
        change_dw2 = np.random.random((11, 128)) * 0.001
        change_db2 = np.random.random((11, 1)) * 0.001
        change_dw1 = np.random.random((128, 784)) * 0.001
        change_db1 = np.random.random((128, 1)) * 0.001
        for i in range(iterations):
            z1 = self.w1 @ self.train_data + self.b1
            a1 = relu(z1)
            z2 = self.w2 @ a1 + self.b2
            a2 = softmax(z2)

            dz2 = a2 - y
            dw2 = 1 / m * dz2 @ a1.T
            db2 = np.array([1 / m * np.sum(dz2, axis=1)]).T
            dz1 = self.w2.T.dot(dz2) * relu_backward(z1)
            dw1 = 1 / m * dz1 @ self.train_data.T
            db1 = np.array([1 / m * np.sum(dz1, axis=1)]).T

            change_w2 = change_dw2 * momentum - alpha * dw2
            change_b2 = change_db2 * momentum - alpha * db2
            change_w1 = change_dw1 * momentum - alpha * dw1
            change_b1 = change_db1 * momentum - alpha * db1

            self.w2 += change_w2
            self.b2 += change_b2
            self.w1 += change_w1
            self.b1 += change_b1

            error.append(loss(a2, y, self.loss_label))
            print(i, error[i])

        plt.plot(range(iterations), error)
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title(f"Learning rate: {alpha}")
        plt.show()

    def predict(self, test_image):
        a1 = relu(self.w1 @ (test_image.T / 255) + self.b1)
        a2 = softmax(self.w2 @ a1 + self.b2)
        return np.argmax(a2, axis=0)


if __name__ == '__main__':
    mndata = MNIST(r'data/')
    images, labels = mndata.load_training()
    images, labels = np.array(images), np.array(labels)
    train_images = images[:40000, :]
    train_label = labels[:40000]
    test_images = images[40001:, :]
    test_labels = labels[40001:]

    model = DigitRecognation(train_images, train_label, 'ce')
    model.fit(100)
    predict = model.predict(test_images[0:10, :])
    print(predict)
    print(test_labels[0:10])
