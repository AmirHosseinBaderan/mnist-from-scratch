import numpy as np

def default_collate(batch):
    images = []
    labels = []

    for image,label in batch:
        images.append(image)
        labels.append(label)

    images = np.stack(images)
    labels = np.array(labels,dtype=np.int64)

    return images,labels