# Neural Network from Scratch: A Complete Course

Welcome to this comprehensive course on building a neural network from scratch! This document will guide you through each component of the MNIST-from-scratch project, explaining not just *what* each class does, but *how* it works and *why* it's designed that way.

---

## Table of Contents

1. [Configuration](#configuration)
2. [Data Pipeline](#data-pipeline)
3. [Transform Classes](#transform-classes)
4. [Neural Network Core](#neural-network-core)
5. [Training and Evaluation](#training-and-evaluation)

---

## Configuration

### `config.py`

This file contains all the hyperparameters and file paths needed for the project. Think of it as the control panel for your neural network.

```python
TRAIN_IMAGES = "data/mnist/train-images.idx3-ubyte"
TRAIN_LABELS = "data/mnist/train-labels.idx1-ubyte"
TEST_IMAGES = "data/mnist/t10k-images.idx3-ubyte"
TEST_LABELS = "data/mnist/t10k-labels.idx1-ubyte"

INPUT_SIZE = 784      # 28 * 28 pixels
HIDDEN_SIZE = 128     # Neurons in hidden layer
NUM_CLASSES = 10      # Digits 0-9

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01

IMAGE_ROWS = 28
IMAGE_COLS = 28
```

**Key Concepts:**
- **INPUT_SIZE**: Each MNIST image is 28×28 = 784 pixels. We flatten it to a 1D vector.
- **HIDDEN_SIZE**: The number of neurons in our hidden layer. More neurons = more capacity to learn complex patterns.
- **NUM_CLASSES**: We have 10 output classes (digits 0-9).
- **LEARNING_RATE**: How big steps we take when updating weights. Too small = slow learning. Too big = overshooting.

---

## Data Pipeline

### `data/dataset.py` - MNISTDataset Class

**What is it?**
The `MNISTDataset` class is responsible for loading and parsing the MNIST dataset files. It reads the binary IDX format used by MNIST.

**How does it work?**

```python
class MNISTDataset:
    IMAGE_HEADER_SIZE = 16
    LABEL_HEADER_SIZE = 8
```

The MNIST files have headers that store metadata. The image header contains:
- Magic number (identifies file type)
- Number of images
- Image dimensions (rows and columns)

**Key Methods:**

1. `__init__`: Opens the files and reads headers
2. `__len__`: Returns total number of samples
3. `__getitem__`: Loads a single image-label pair by index
4. `_read_image`: Uses `seek()` to jump to the correct position and reads raw bytes
5. `_read_label`: Reads a single byte representing the digit

**Why is this important?**
Understanding how to read data is fundamental. Real-world datasets come in many formats, and you need to know how to parse them efficiently.

---

### `data/loader.py` - DataLoader Class

**What is it?**
The `DataLoader` creates batches of data for training. Instead of processing one image at a time, we process multiple images together (batch processing).

**How does it work?**

```python
class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None):
```

- **batch_size**: How many samples to process together
- **shuffle**: Randomize order each epoch (important for training)
- **collate_fn**: How to combine individual samples into a batch

**The Iterator Pattern:**
The DataLoader implements `__iter__` and `__next__`, making it iterable. This allows us to use it in a for loop:

```python
for images, labels in train_loader:
    # Process batch
```

**Why batch processing?**
1. More efficient computation (vectorization)
2. Better gradient estimates (averaging over multiple samples)
3. Enables parallel processing on GPUs

---

### `data/collate.py` - default_collate Function

**What is it?**
A simple function that combines a list of (image, label) pairs into batch arrays.

**How does it work?**

```python
def default_collate(batch):
    images = []
    labels = []
    
    for image, label in batch:
        images.append(image)
        labels.append(label)
    
    images = np.stack(images)  # Stack into 2D array
    labels = np.array(labels, dtype=np.int64)
    
    return images, labels
```

**Key Insight:**
`np.stack` creates a new dimension, turning a list of 784-element vectors into a 2D array of shape `(batch_size, 784)`.

---

## Transform Classes

### `transforms/compose.py` - Compose Class

**What is it?**
A container that chains multiple transformations together.

**How does it work?**

```python
class Compose:
    def __init__(self, transformers):
        self.transformers = transformers
    
    def __call__(self, image):
        for transformer in self.transformers:
            image = transformer(image)
        return image
```

**Why use it?**
It allows us to create a pipeline:
```python
transform = Compose([
    ToNumpy(),
    Reshape(28, 28),
    Flatten(),
    Normalize(),
])
```

Each image flows through all transformations in sequence.

---

### `transforms/to_numpy.py` - ToNumpy Class

**What is it?**
Converts raw bytes from the file into a NumPy array.

**How does it work?**

```python
class ToNumpy:
    def __call__(self, image):
        return np.frombuffer(image, dtype=np.uint8)
```

`np.frombuffer` interprets raw bytes as an array of unsigned 8-bit integers (0-255), which are the pixel values.

---

### `transforms/reshape.py` - Reshape Class

**What is it?**
Reshapes a 1D array into a 2D matrix.

**How does it work:**

```python
class Reshape:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    
    def __call__(self, image):
        return image.reshape(self.rows, self.cols)
```

**Example:**
- Input: `[0, 1, 2, ..., 783]` (784 elements)
- Output: `[[0, 1, 2, ...], [28, 29, 30, ...], ...]` (28×28 matrix)

---

### `transforms/flatten.py` - Flatten Class

**What is it?**
Converts a 2D image back to 1D for the neural network.

**How does it work:**

```python
class Flatten:
    def __call__(self, x):
        return x.reshape(-1)
```

The `-1` tells NumPy to automatically calculate the size: 28×28 = 784.

**Why flatten?**
Neural network layers work with 1D vectors. Each pixel becomes a feature.

---

### `transforms/normalize.py` - Normalize Class

**What is it?**
Scales pixel values from [0, 255] to [0, 1].

**How does it work:**

```python
class Normalize:
    def __call__(self, image):
        return image.astype(np.float32) / 255.0
```

**Why normalize?**
- Neural networks work better with small values
- Prevents numerical instability
- Makes gradients more stable during training

---

## Neural Network Core

### `nn/parameter.py` - Parameter Class

**What is it?**
A wrapper for learnable values (weights and biases) in the network.

**How does it work:**

```python
class Parameter:
    def __init__(self, data, requires_grad=True):
        self.data = data      # The actual values
        self.grad = None      # Gradient (computed during backward pass)
        self.requires_grad = requires_grad
```

**Key Concepts:**
- **data**: The parameter values (e.g., weights)
- **grad**: How much to change the parameter (computed by backpropagation)
- **requires_grad**: Whether to compute gradients for this parameter

**Why a separate class?**
It allows us to track gradients and update parameters systematically.

---

### `nn/module.py` - Module Class (Base Class)

**What is it?**
The base class for all neural network components. Every layer inherits from this.

**How does it work:**

```python
class Module:
    def __init__(self):
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError  # Must be implemented by subclass
    
    def parameters(self):
        return []  # Return list of parameters
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
        for child in self.children():
            child.eval()
```

**Key Methods:**
- `__call__`: Allows using the object like a function: `layer(input)`
- `forward`: The actual computation (implemented by each layer)
- `parameters`: Returns all learnable parameters
- `train()` / `eval()`: Set training mode (for layers like Dropout, BatchNorm)

---

### `nn/linear.py` - Linear Class

**What is it?**
A fully connected (dense) layer. Each input connects to each output.

**How does it work?**

```python
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        # Initialize weights with small random values
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * 0.01
        )
        self.bias = Parameter(np.zeros(out_features)) if bias else None
```

**Forward Pass:**
```python
def forward(self, X):
    self.input = X  # Cache for backward pass
    output = X @ self.weight.data  # Matrix multiplication
    if self.bias is not None:
        output += self.bias.data
    return output
```

**Mathematical Formula:**
```
output = input × weight + bias
```

Where:
- `input` shape: (batch_size, in_features)
- `weight` shape: (in_features, out_features)
- `output` shape: (batch_size, out_features)

**Backward Pass:**
```python
def backward(self, grad_output):
    # Gradient w.r.t. input
    grad_input = grad_output @ self.weight.data.T
    
    # Gradient w.r.t. weights
    self.weight.grad = self.input.T @ grad_output
    
    # Gradient w.r.t. bias
    if self.bias is not None:
        self.bias.grad = grad_output.sum(axis=0)
    
    return grad_input
```

**Key Insight:**
- We cache `self.input` during forward pass for use in backward pass
- The chain rule gives us: `grad_input = grad_output × weight^T`
- Weight gradient: `grad = input^T × grad_output`

---

### `nn/relu.py` - ReLU Class

**What is it?**
The Rectified Linear Unit activation function. It introduces non-linearity.

**How does it work:**

```python
class ReLU(Module):
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)  # max(0, x)
```

**Mathematical Formula:**
```
ReLU(x) = max(0, x)
```

**Why do we need it?**
Without activation functions, a neural network would just be a linear transformation, no matter how many layers it has. ReLU allows the network to learn non-linear patterns.

**Backward Pass:**
```python
def backward(self, grad_output):
    return grad_output * (self.input > 0)
```

**Key Insight:**
- If input > 0: gradient passes through unchanged
- If input ≤ 0: gradient is zero (no learning happens for negative values)

---

### `nn/sequential.py` - Sequential Class

**What is it?**
A container that chains multiple layers together.

**How does it work:**

```python
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
```

**Example:**
```python
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
)
```

**Forward Flow:**
```
input → Linear(784, 128) → ReLU → Linear(128, 10) → output
```

**Backward Flow:**
```
grad → Linear(128, 10) → ReLU → Linear(784, 128) → input_grad
```

**Why reverse for backward?**
Gradients flow in the opposite direction of the forward pass (backpropagation).

---

### `nn/losses/cross_entropy.py` - CrossEntropyLoss Class

**What is it?**
Measures how well the model's predictions match the true labels.

**How does it work:**

**Step 1: Softmax**
Converts logits (raw scores) to probabilities:

```python
def _softmax(self, logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)  # Numerical stability
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)
```

**Mathematical Formula:**
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**Step 2: Cross-Entropy**
Measures the difference between predicted and true distributions:

```python
def _cross_entropy(self, probabilities, targets):
    correct = probabilities[np.arange(self.batch_size), targets]
    correct = np.clip(correct, 1e-15, 1.0)  # Prevent log(0)
    loss = -np.log(correct)
    return np.mean(loss)
```

**Mathematical Formula:**
```
CE = -log(probability of correct class)
```

**Step 3: Backward Pass**
```python
def backward(self):
    one_hot = np.zeros_like(self.probabilities)
    one_hot[np.arange(self.batch_size), self.targets] = 1.0
    grad = (self.probabilities - one_hot) / self.batch_size
    return grad
```

**Key Insight:**
The gradient of cross-entropy with softmax simplifies beautifully to:
```
grad = (probabilities - one_hot) / batch_size
```

---

## Training and Evaluation

### `optim/sgd.py` - SGD Class

**What is it?**
The optimizer that updates model parameters using Stochastic Gradient Descent.

**How does it work:**

```python
class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate
    
    def step(self):
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.data -= self.learning_rate * parameter.grad
    
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
```

**The Update Rule:**
```
parameter = parameter - learning_rate × gradient
```

**Why zero_grad?**
Gradients accumulate, so we need to reset them before each backward pass.

---

### `train.py` - Training Functions

**What is it?**
Contains the training loop logic.

**Key Functions:**

1. `train_one_epoch`: Trains for one full pass through the data
2. `validate_one_epoch`: Evaluates on validation data
3. `train`: Main training loop

**Training Loop:**
```python
for epoch in range(epochs):
    # Training phase
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        grad = criterion.backward()
        model.backward(grad)
        optimizer.step()
    
    # Validation phase
    validation_loss, validation_accuracy = evaluate(...)
```

**Key Steps:**
1. **Forward pass**: Compute predictions
2. **Compute loss**: Measure error
3. **Zero gradients**: Reset gradients
4. **Backward pass**: Compute gradients
5. **Update weights**: Apply SGD

---

### `evaluate.py` - Evaluation Function

**What is it?**
Computes loss and accuracy on a dataset.

**How does it work:**

```python
def evaluate(model, test_loader, criterion):
    model.eval()  # Set to evaluation mode
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        predictions = np.argmax(outputs, axis=1)
        correct += np.sum(predictions == labels)
        total += len(labels)
        total_loss += loss
    
    return total_loss / len(test_loader), correct / total
```

**Key Metrics:**
- **Loss**: Average cross-entropy loss
- **Accuracy**: Percentage of correct predictions

---

### `predict.py` - Prediction Function

**What is it?**
Shows predictions for a few samples.

**How does it work:**

```python
def predict(model, data_loader, count=10):
    model.eval()
    
    for images, labels in data_loader:
        outputs = model(images)
        predictions = np.argmax(outputs, axis=1)
        
        for image, prediction, label in zip(images, predictions, labels):
            print(f"Prediction: {prediction} | Label: {label}")
```

**Key Function:**
- `np.argmax(outputs, axis=1)`: Returns the index of the highest probability (predicted class)

---

## Summary

This project demonstrates the complete pipeline of a neural network:

1. **Data**: Load and preprocess images
2. **Model**: Define architecture (Linear → ReLU → Linear)
3. **Loss**: Measure prediction quality
4. **Backward**: Compute gradients
5. **Optimize**: Update parameters

Each class is designed to be modular and reusable, following the same patterns as popular frameworks like PyTorch, but with all the complexity exposed for learning purposes.

---

## Exercises for Students

1. **Modify the architecture**: Try different hidden sizes
2. **Add more layers**: Add another Linear+ReLU layer
3. **Change activation**: Implement Sigmoid or Tanh instead of ReLU
4. **Add momentum**: Implement SGD with momentum
5. **Visualize weights**: Plot the learned weight matrices

Happy learning! 🎓