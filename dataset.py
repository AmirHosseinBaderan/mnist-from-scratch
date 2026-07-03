import struct


class MNISTDataset:

    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    def read_image_header(self):
        with open(self.image_path, 'rb') as f:
            magic = struct.unpack(">I", f.read(4))[0]
            count = struct.unpack(">I", f.read(4))[0]
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]

        return magic, count, rows, cols

    def read_label_header(self):
        with open(self.label_path, 'rb') as f:
            magic = struct.unpack(">I", f.read(4))[0]
            count = struct.unpack(">I", f.read(4))[0]

        return magic, count

    def read_first_image(self):
        with open(self.image_path, 'rb') as f:
            # header
            f.read(16)

            image = f.read(28 * 28)

        return image
