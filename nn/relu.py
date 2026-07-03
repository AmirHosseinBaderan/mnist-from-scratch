from nn.module import Module
import numpy as np


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None

    def forward(self, X):
        self.input = X
        output = np.maximum(0, X)
        return output

    def backward(self, grad_output):
        return grad_output * (self.input > 0)
