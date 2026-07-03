class Reshape:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def __call__(self, image):
        return image.reshape(self.rows, self.cols)
