import numpy as np

from nn.module import Module
from nn.parameter import Parameter


class Linear(Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            np.random.randn(in_features, out_features) * 0.01
        )

        self.bias = (
            Parameter(np.zeros(out_features))
            if bias
            else None
        )

        # cache for backward
        self.input = None

    def forward(self, X):
        self.input = X

        output = X @ self.weight.data

        if self.bias is not None:
            output += self.bias.data

        return output

    def parameters(self):
        parameters = [self.weight]

        if self.bias is not None:
            parameters.append(self.bias)

        return parameters

    def backward(self, grad_output):

        grad_input = grad_output @ self.weight.data.T

        self.weight.grad = self.input.T @ grad_output

        if self.bias is not None:
            self.bias.grad = grad_output.sum(axis=0)

        return grad_input
