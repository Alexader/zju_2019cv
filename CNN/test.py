import torch
import sys
import model
from cifar10 import LeNet
import numpy as np
from torchvision.datasets import mnist, cifar
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def val(test_loader, model):
    cifar_test = cifar.CIFAR10("cifar10_test", train=False, transform=ToTensor())
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

def testMinst():
    minst_test = mnist.MNIST("minst_test", train=False, transform=ToTensor())

    batch_size = 256
    minst_model = model.Model()
    minst_model.load_state_dict(torch.load("./minst.pt"))
    test_loader = DataLoader(minst_test, batch_size=batch_size)
    val(test_loader, minst_model)

def testCifar(epoch):
    cifar_test = cifar.CIFAR10("cifar10_test", train=False, transform=ToTensor())

    batch_size = 256
    # epoch = 100
    cifar_model = LeNet()
    cifar_model.load_state_dict(torch.load("./cifar10.{0}.pt".format(epoch)))
    test_loader = DataLoader(cifar_test, batch_size=batch_size)
    val(test_loader, cifar_model)

if __name__ == "__main__":
    
    # cifar_test = mnist.MNIST("./cifar10_test", train=False, transform=ToTensor())

    # test models
    # print("test for minst")
    # testMinst()

    print("test for cifar")
    testCifar(sys.argv[1])

