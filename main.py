from dataset import MNISTDataset
import matplotlib.pyplot as plt

from transformers.reshape import Reshape
from transformers.to_numpy import ToNumpy

dataset = MNISTDataset(
    "data/mnist/train-images.idx3-ubyte",
    "data/mnist/train-labels.idx1-ubyte",
)

image, label = dataset[0]
to_numpy = ToNumpy()
reshape = Reshape(28, 28)

image = to_numpy(image)
image = reshape(image)

plt.imshow(image,cmap="gray")
plt.show()
