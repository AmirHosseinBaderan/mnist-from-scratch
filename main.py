from data.dataset import MNISTDataset
import matplotlib.pyplot as plt
import numpy as np

from data.loader import DataLoader
from nn.linear import Linear
from nn.module import Module
from nn.parameter import Parameter
from nn.relu import ReLU
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

relu = ReLU()
x = np.array([
    [-2, -1, 0, 1, 2],
    [-2, -1, 0, 4, 3]
])

y = relu(x)
print(y)

grad = np.ones_like(y)

dx = relu.backward(grad)

print(dx)