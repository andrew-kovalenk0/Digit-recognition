import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np


def loss(yp, y, func):
    return globals()[func](yp, y)


def mse(yp, y):
    return np.square(yp - y).mean() / y.shape[1]


def ce(yp, y):
    return -np.sum(y * np.log(yp)) / y.shape[1]


def softmax(z):
    e = np.exp(z)
    return e / (e.sum(axis=0) + 1e-9)


def relu(z):
    return np.maximum(z, 0)


def relu_backward(z):
    return z > 0


def one_hot(y):
    result = np.zeros((y.size, y.max() + 1))
    result[np.arange(y.size), y] = 1
    return result.T


def get_accuracy(prediction, y):
    return f'{int((np.sum(prediction == y) / y.shape)[0] * 100)} %'


class DigitRecognation:
    def __init__(self, train_data, train_labels, valid_x, valid_y, loss_label):
        self.train_data = (train_data.T - 127.5) / 255
        self.valid_data = (valid_x.T - 127.5) / 255
        self.valid_label = one_hot(valid_y)
        self.train_labels = one_hot(train_labels)

        self.loss_label = loss_label
        self.w1 = np.random.uniform(-0.01, 0.01, [128, 784])
        self.b1 = np.random.uniform(-0.01, 0.01, [128, 1])
        self.w2 = np.random.uniform(-0.01, 0.01, [10, 128])
        self.b2 = np.random.uniform(-0.01, 0.01, [10, 1])

    def fit(self, iterations, batches):
        batch_size = int(self.train_data.shape[1] / batches)
        alpha = 0.01
        error_train = np.zeros(iterations)
        error_valid = np.zeros(iterations)

        momentum = 0.9
        change_dw1 = np.random.uniform(-0.01, 0.01, [128, 784])
        change_db1 = np.random.uniform(-0.01, 0.01, [128, 1])
        change_dw2 = np.random.uniform(-0.01, 0.01, [10, 128])
        change_db2 = np.random.uniform(-0.01, 0.01, [10, 1])

        for i in range(iterations):
            for j in range(batches):
                train_data = self.train_data[:, j * batch_size:(j + 1) * batch_size]
                y = self.train_labels[:, j * batch_size:(j + 1) * batch_size]
                z1 = self.w1 @ train_data + self.b1
                a1 = relu(z1)
                z2 = self.w2 @ a1 + self.b2
                a2 = softmax(z2)

                dz2 = a2 - y
                dw2 = 1 / batch_size * dz2 @ a1.T
                db2 = np.array([1 / batch_size * np.sum(dz2, axis=1)]).T
                dz1 = self.w2.T @ dz2 * relu_backward(z1)
                dw1 = 1 / batch_size * dz1 @ train_data.T
                db1 = np.array([1 / batch_size * np.sum(dz1, axis=1)]).T

                change_dw2 = change_dw2 * momentum - alpha * dw2
                change_db2 = change_db2 * momentum - alpha * db2
                change_dw1 = change_dw1 * momentum - alpha * dw1
                change_db1 = change_db1 * momentum - alpha * db1

                self.w2 += change_dw2
                self.b2 += change_db2
                self.w1 += change_dw1
                self.b1 += change_db1

                if j == batches - 1:
                    error_train[i] = loss(a2, y, self.loss_label)
                    a1 = relu(self.w1 @ self.valid_data + self.b1)
                    a2 = softmax(self.w2 @ a1 + self.b2)
                    error_valid[i] = loss(a2, self.valid_label, self.loss_label)

        plt.plot(range(1, iterations + 1), error_train, label='Train')
        plt.plot(range(1, iterations + 1), error_valid, label='Valid')
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title(f"Learning rate: {alpha}")
        plt.legend(loc='best')
        plt.show()

    def predict(self, test_image):
        a1 = relu(self.w1 @ ((test_image.T - 127.5) / 255) + self.b1)
        a2 = softmax(self.w2 @ a1 + self.b2)
        return np.argmax(a2, axis=0)


if __name__ == '__main__':
    mndata = MNIST(r'data/')
    images, labels = mndata.load_training()
    images, labels = np.array(images), np.array(labels)
    train_images = images[:40000, :]
    train_label = labels[:40000]
    test_images = images[40001:41000, :]
    test_labels = labels[40001:41000]
    valid_images = images[41001:, :]
    valid_labels = labels[41001:]

    model = DigitRecognation(train_images, train_label, valid_images, valid_labels, 'ce')
    model.fit(25, 64)

    predictions = model.predict(test_images)
    print(f'\nAccuracy: {get_accuracy(predictions, test_labels)}')
