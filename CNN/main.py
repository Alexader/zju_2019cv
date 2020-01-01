import torch
import sys
import numpy as np
from torchvision.datasets import mnist, cifar
from torchvision.transforms import ToTensor
import train

def train_minst():
    train_dataset = mnist.MNIST(root='./minist_train', download=False, train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./minst_test', download=False, train=False, transform=ToTensor())
    model = train.TrainTask(train_dataset, test_dataset, 50, 1)
    torch.save(model.state_dict(), "./minst.pt")

def train_cifar10(epoch):
    train_dataset = cifar.CIFAR10(root='./cifar10_train', download=False, train=True, transform=ToTensor())
    test_dataset = cifar.CIFAR10(root='./cifar10_test', download=False, train=False, transform=ToTensor())
    # epoch = 100
    model = train.TrainTask(train_dataset, test_dataset, epoch, 2)
    torch.save(model.state_dict(), "./cifar10.{0}.pt".format(epoch))

def val(test_loader, model):
    correct = 0
    _sum = 0
    for idx, (test_x, test_label) in enumerate(test_loader):
        predict_y = model(test_x.float()).detach()
        predict_ys = np.argmax(predict_y, axis=-1)
        label_np = test_label.numpy()
        _ = predict_ys == test_label
        correct += np.sum(_.numpy(), axis=-1)
        _sum += _.shape[0]

    print('accuracy: {:.2f}'.format(correct / _sum))

if __name__ == "__main__":
    # print("run minst task")
    # train_minst()

    print("run cifar10 task")
    train_cifar10(int(sys.argv[1]))