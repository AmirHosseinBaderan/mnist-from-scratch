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

criterion = CrossEntropyLoss()

logits = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])

targets = np.array([0, 1])

loss = criterion(logits, targets)

print(loss)

grad = criterion.backward()

print(grad)
print(grad.shape)