from nn.module import Module


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
