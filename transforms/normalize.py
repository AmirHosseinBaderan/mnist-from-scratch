import numpy as np


class Normalize:

    def __call__(self, image):
        return image.astype(np.float32) / 255.0
