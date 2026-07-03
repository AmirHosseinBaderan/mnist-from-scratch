from data.dataset import MNISTDataset
import matplotlib.pyplot as plt

from data.loader import DataLoader
from transformers.compose import Compose
from transformers.reshape import Reshape
from transformers.to_numpy import ToNumpy

compose = Compose([
    ToNumpy(),
    Reshape(28,28),
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

for images,labels in loader:
    print(images.shape)
    print(labels.shape)