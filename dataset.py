import struct

class MNISTDataset:

    def __init__(self,image_path,label_path):
        self.image_path = image_path
        self.label_path = label_path