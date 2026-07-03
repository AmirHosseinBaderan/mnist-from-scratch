from dataset import MNISTDataset

dataset = MNISTDataset(
    "data/mnist/train-images.idx3-ubyte",
    "data/mnist/train-labels.idx1-ubyte",
)


print(len(dataset))

image, label = dataset[0]
print(label)