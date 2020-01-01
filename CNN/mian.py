import torch
from torchvision.datasets import mnist, cifar
from torchvision.transforms import ToTensor
import train

def train_minst():
    train_dataset = mnist.MNIST(root='./minist_train', download=True, train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./minst_test', download=True, train=False, transform=ToTensor())
    model = train.TrainTask(train_dataset, test_dataset, 50)
    torch.save(model.state_dict(), "./minst.pt")

def train_cifar10():
    train_dataset = cifar.CIFAR10(root='./cifar10_train', download=True, train=True, transform=ToTensor())
    test_dataset = cifar.CIFAR10(root='./cifar10_test', download=True, train=False, transform=ToTensor())
    model = train.TrainTask(train_dataset, test_dataset, 50)
    torch.save(model.state_dict(), "./cifar10.pt")


if __name__ == "__main__":
    print("run minst task")
    train_minst()

    print("run cifar10 task")
    train_cifar10()