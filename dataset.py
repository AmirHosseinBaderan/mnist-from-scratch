import struct


class MNISTDataset:

    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

        (
            self.image_magic,
            self.image_count,
            self.rows,
            self.cols
        ) = self._read_image_header()

        (
            self.label_magic,
            self.label_count
        ) = self._read_label_header()

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):
        image_offset = 16 + idx * self.rows * self.cols
        label_offset = 8 + idx

        with open(self.image_path, 'rb') as f:
            f.seek(image_offset)

            image = f.read(self.rows * self.cols)

        with open(self.label_path, 'rb') as f:
            f.seek(label_offset)
            label = f.read(1)[0]

        return image, label

    def _read_image_header(self):
        with open(self.image_path, 'rb') as f:
            magic = struct.unpack(">I", f.read(4))[0]
            count = struct.unpack(">I", f.read(4))[0]
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]

        return magic, count, rows, cols

    def _read_label_header(self):
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
