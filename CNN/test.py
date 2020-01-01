import torch
import model
from torchvision.datasets import mnist, cifar
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    minst_test = mnist.MNIST("./minst_test", train=False, transform=ToTensor())
    cifar_test = mnist.MNIST("./cifar10_test", train=False, transform=ToTensor())

    # load model
    minst_model = model.Model()
    minst_model.load_state_dict(torch.load("./minst.pt"))

    cifar_model = model.Model()
    cifar_model.load_state_dict(torch.load("./cifar10.pt"))
    cifar_model.train()
