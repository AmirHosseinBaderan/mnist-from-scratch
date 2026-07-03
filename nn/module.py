class Module:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

        for child in self.children():
            child.eval()

    def children(self):
        return []