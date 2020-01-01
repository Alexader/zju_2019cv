import numpy as np
import torch
import matplotlib.pyplot as plt
from model import Model
from cifar10 import LeNet
from torchvision.datasets import mnist, cifar
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

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
    return correct / _sum

def TrainTask(train_dataset, test_dataset, epoch, typ):
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    if typ == 1:
        model = Model()
    else:
        model = LeNet()
    
    sgd = SGD(model.parameters(), lr=1e-1)
    cross_error = CrossEntropyLoss()
    accs = []
    losses = []

    for _epoch in range(epoch):
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            # if idx % 10 == 0:
            #     print('idx: {}, _error: {}'.format(idx, _error))
            _error.backward()
            sgd.step()

        cifar_test = cifar.CIFAR10("cifar10_test", train=False, transform=ToTensor())
        batch_size = 256
        test_loader = DataLoader(cifar_test, batch_size=batch_size)
        acc = val(test_loader, model)
        accs.append(acc)
        losses.append(_error)

    return accs, losses, model

if __name__ == "__main__":
    train_dataset = cifar.CIFAR10(root='./cifar10_train', download=False, train=True, transform=ToTensor())
    test_dataset = cifar.CIFAR10(root='./cifar10_test', download=False, train=False, transform=ToTensor())
    epoch = 100
    acc, loss, model = TrainTask(train_dataset, test_dataset, epoch, 2)

    x1 = range(0, epoch)
    x2 = range(0, epoch)
    y1 = acc
    y2 = loss
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")