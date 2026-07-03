from data.dataset import MNISTDataset
import matplotlib.pyplot as plt
import numpy as np

from data.loader import DataLoader
from nn.cross_entropy import CrossEntropyLoss
from nn.linear import Linear
from nn.module import Module
from nn.parameter import Parameter
from nn.relu import ReLU
from nn.sequential import Sequential
from transforms.compose import Compose
from transforms.reshape import Reshape
from transforms.to_numpy import ToNumpy
from nn.sgd import SGD

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

w = Parameter(np.array([1.0, 2.0]))

w.grad = np.array([0.1, 0.2])

optimizer = SGD([w], learning_rate=0.5)

optimizer.step()

print(w.data)
