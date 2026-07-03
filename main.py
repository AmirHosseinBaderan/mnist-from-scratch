from data.dataset import MNISTDataset
import matplotlib.pyplot as plt
import numpy as np

from data.loader import DataLoader
from nn.linear import Linear
from nn.module import Module
from nn.parameter import Parameter
from transforms.compose import Compose
from transforms.reshape import Reshape
from transforms.to_numpy import ToNumpy

compose = Compose([
    ToNumpy(),
    Reshape(28, 28),
])

dataset = MNISTDataset(
    "data/mnist/train-images.idx3-ubyte",
    "data/mnist/train-labels.idx1-ubyte",
    image_transform=compose,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

linear = Linear(784, 128)

x = np.random.randn(32, 784)

y = linear(x)

grad = np.random.randn(32, 128)

dx = linear.backward(grad)

print(dx.shape)

print(linear.weight.grad.shape)

print(linear.bias.grad.shape)