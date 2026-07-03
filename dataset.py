import struct


class MNISTDataset:

    def __init__(self, image_path, label_path):
        self.image_file = open(image_path, "rb")
        self.label_file = open(label_path, "rb")

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

    def __del__(self):
        self.image_file.close()
        self.label_file.close()

    def __getitem__(self, idx):
        image_offset = 16 + idx * self.rows * self.cols
        label_offset = 8 + idx

        self.image_file.seek(image_offset)
        image = self.image_file.read(self.rows * self.cols)

        self.label_file.seek(label_offset)
        label = self.label_file.read(1)[0]

        return image, label

    def _read_image_header(self):
        magic = struct.unpack(">I", self.image_file.read(4))[0]
        count = struct.unpack(">I", self.image_file.read(4))[0]
        rows = struct.unpack(">I", self.image_file.read(4))[0]
        cols = struct.unpack(">I", self.image_file.read(4))[0]

        return magic, count, rows, cols

    def _read_label_header(self):
        magic = struct.unpack(">I", self.label_file.read(4))[0]
        count = struct.unpack(">I", self.label_file.read(4))[0]

        return magic, count
