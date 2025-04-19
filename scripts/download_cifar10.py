from torchvision.datasets import CIFAR10
import os
data_dir = "data/cifar10"
CIFAR10(data_dir, train=True, download=True)
CIFAR10(data_dir, train=False, download=True)