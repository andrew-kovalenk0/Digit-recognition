from mnist import MNIST


if __name__ == '__main__':
    mndata = MNIST(r'')
    images, labels = mndata.load_training()
    print(images, labels)
