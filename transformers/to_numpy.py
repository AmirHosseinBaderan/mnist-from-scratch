
import numpy as np


class ToNumpy:
    def __call__(self, image):
        return np.frombuffer(image, dtype=np.uint8)
