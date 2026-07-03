from nn.module import Module
import numpy as np


class CrossEntropyLoss(Module):

    def __init__(self):
        super().__init__()
        self.probabilities = None
        self.targets = None
        self.batch_size = None

    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        probabilities = self._softmax(logits)

        self.probabilities = probabilities
        self.targets = targets
        self.batch_size = len(targets)

        return self._cross_entropy(probabilities, targets)

    def backward(self):
        one_hot = np.zeros_like(self.probabilities)
        one_hot[np.arange(self.batch_size), self.targets] = 1.0
        grad = (self.probabilities - one_hot) / self.batch_size

        return grad

    def _softmax(self, logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)

        return exp / np.sum(exp, axis=1, keepdims=True)

    def _cross_entropy(self, probabilities, targets):
        correct = probabilities[np.arange(self.batch_size), targets]
        correct = np.clip(correct, 1e-15, 1.0)
        loss = -np.log(correct)

        return np.mean(loss)
