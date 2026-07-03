class Compose:
    def __init__(self, transformers):
        self.transformers = transformers

    def __call__(self, image):
        for transformer in self.transformers:
            image = transformer(image)

        return image
