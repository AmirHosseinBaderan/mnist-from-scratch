import struct


class MNISTDataset:
    IMAGE_HEADER_SIZE = 16
    LABEL_HEADER_SIZE = 8

    def __init__(
            self,
            image_path,
            label_path,
            image_transform=None,
            label_transform=None,
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform

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
        self._validate_index(idx)

        image = self._read_image(idx)
        label = self._read_label(idx)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return image, label

    def _validate_index(self,idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Dataset index out of range.")

    def _read_image(self,idx):
        image_offset = self.IMAGE_HEADER_SIZE + idx * self.rows * self.cols
        self.image_file.seek(image_offset)
        image = self.image_file.read(self.rows * self.cols)

        return image

    def _read_label(self,idx):
        label_offset = self.LABEL_HEADER_SIZE + idx
        self.label_file.seek(label_offset)
        label = self.label_file.read(1)[0]

        return label

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
