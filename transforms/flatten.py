
class Flatten:

    def __call__(self, x):
        return x.reshape(-1)