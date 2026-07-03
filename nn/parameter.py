class Parameter:
    def __init__(self, data,requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Parameter (shape={self.data.shape})"