from dataset import MNISTDataset

dataset = MNISTDataset(
    "data/mnist/train-images.idx3-ubyte",
    "data/mnist/train-labels.idx1-ubyte",
)


magic,count,rows,cols = dataset.read_image_header()

print("Magic:",magic)
print("Count:",count)
print("Rows:",rows)
print("Cols:",cols)

label_magic,label_count = dataset.read_label_header()
print("Label Magic:",label_magic)
print("Label Count:",label_count)