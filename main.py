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


class Layer:
    def __init__(self, in_size, out_size, batch_size):
        self.w = np.random.uniform(-0.01, 0.01, [out_size, in_size])
        self.b = np.random.uniform(-0.01, 0.01, [out_size, 1])
        self.change_dw = np.random.uniform(-0.01, 0.01, [out_size, in_size])
        self.change_db = np.random.uniform(-0.01, 0.01, [out_size, 1])
        self.z = np.zeros((out_size, batch_size))
        self.a = np.zeros((out_size, batch_size))

    def forward_prop(self, prev_a):
        self.z = self.w @ prev_a + self.b
        self.a = softmax(self.z)

    def backward_prop(self):
        pass
    
    def change_wights(self):
        pass


class DigitRecognation(Layer):
    def __init__(self, in_size, out_size, hiden_layers, hiden_layers_size, iterations, batch_size):
        super().__init__(in_size, out_size, batch_size)
        layers = [Layer(in_size, hiden_layers_size[0], batch_size)]
        for i in range(hiden_layers - 1):
            layers.append(Layer(hiden_layers_size[i], hiden_layers_size[i + 1], batch_size))
        layers.append(Layer(hiden_layers_size[0], out_size, batch_size))
        self.iterations = iterations
        self.batch_size = batch_size
        self.layers = layers
        self.hiden_layers = hiden_layers
        
    def fit(self, train_data, train_labels, valid_x, valid_y, loss_label):
        train_data = (train_data.T - 127.5) / 255
        valid_data = (valid_x.T - 127.5) / 255
        valid_label = one_hot(valid_y)
        train_labels = one_hot(train_labels)
        batches = int(train_data.shape[1] / self.batch_size)
        loss_label = loss_label
        alpha = 0.01
        error_train = np.zeros(self.iterations)
        error_valid = np.zeros(self.iterations)
        momentum = 0.9

        for i in range(self.iterations):
            for j in range(batches):
                train_data_buf = train_data[:, j * self.batch_size:(j + 1) * self.batch_size]
                y = train_labels[:, j * self.batch_size:(j + 1) * self.batch_size]
                self.layers[0].z = self.layers[0].w @ train_data_buf + self.layers[0].b
                self.layers[0].a = relu(self.layers[0].z)
                for li in range(1, self.hiden_layers + 1):
                    self.layers[li].z = self.layers[li].w @ self.layers[li - 1].a + self.layers[li].b
                    self.layers[li].a = softmax(self.layers[li].z)

                dz2 = self.layers[1].a - y
                dw2 = 1 / self.batch_size * dz2 @ self.layers[0].a.T
                db2 = np.array([1 / self.batch_size * np.sum(dz2, axis=1)]).T
                dz1 = self.layers[1].w.T @ dz2 * relu_backward(self.layers[0].z)
                dw1 = 1 / self.batch_size * dz1 @ train_data_buf.T
                db1 = np.array([1 / self.batch_size * np.sum(dz1, axis=1)]).T

                for li in range(self.hiden_layers + 1):
                    self.layers[li].change_dw = self.layers[li].change_dw * momentum - alpha * dw2
                    self.layers[li].change_db = self.layers[li].change_db * momentum - alpha * db2
                    self.layers[li].w += self.layers[li].change_dw
                    self.layers[li].b += self.layers[li].change_db

                if j == batches - 1:
                    error_train[i] = loss(self.layers[1].a, y, loss_label)
                    a1 = relu(self.layers[0].w @ valid_data + self.layers[0].b)
                    a2 = softmax(self.layers[1].w @ a1 + self.layers[1].b)
                    error_valid[i] = loss(a2, valid_label, loss_label)

        plt.plot(range(1, self.iterations + 1), error_train, label='Train')
        plt.plot(range(1, self.iterations + 1), error_valid, label='Valid')
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title(f"Learning rate: {alpha}")
        plt.legend(loc='best')
        plt.show()

    def predict(self, test_image):
        a1 = relu(self.layers[0].w @ ((test_image.T - 127.5) / 255) + self.layers[0].b)
        a2 = softmax(self.layers[1].w @ a1 + self.layers[1].b)
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

    model = DigitRecognation(784, 10, 2, [64, 64], 10, 625)
    model.fit(train_images, train_label, valid_images, valid_labels, 'ce')

    predictions = model.predict(test_images)
    print(f'\nAccuracy: {get_accuracy(predictions, test_labels)}')
