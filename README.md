# MNIST From Scratch

A complete neural network implementation for handwritten digit recognition built entirely from scratch using only NumPy. This project demonstrates the fundamental concepts of deep learning by implementing a simple feedforward neural network to classify MNIST digits without relying on high-level frameworks like PyTorch or TensorFlow.

## What is MNIST?

The MNIST (Modified National Institute of Standards and Technology) dataset is a classic benchmark in machine learning, consisting of 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels, and the task is to classify each image into one of the 10 digit classes.

## Project Overview

This project implements a complete machine learning pipeline from data loading to training and evaluation, including:

- **Data Loading**: Custom MNIST dataset reader that parses the IDX binary format
- **Data Transforms**: Image preprocessing pipeline (normalization, reshaping, flattening)
- **Neural Network Layers**: Linear (fully connected) and ReLU activation layers
- **Loss Function**: Cross-entropy loss with softmax
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Training Loop**: Complete training and validation pipeline

## Model Architecture

The neural network is a simple feedforward network with the following architecture:

```
Input (784 features) → Linear (128 units) → ReLU → Linear (10 units) → Output
```

- **Input Layer**: 784 neurons (28×28 pixels flattened)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each digit class)

## Project Structure

```
mnist-from-scratch/
├── main.py              # Entry point - model setup and training
├── train.py             # Training loop implementation
├── evaluate.py          # Model evaluation function
├── predict.py           # Prediction and visualization
├── config.py            # Configuration and hyperparameters
├── data/
│   ├── dataset.py       # MNIST dataset loader
│   ├── loader.py        # DataLoader for batching
│   └── collate.py       # Batch collation utility
├── nn/
│   ├── module.py        # Base Module class
│   ├── parameter.py     # Parameter class for weights
│   ├── linear.py        # Linear layer implementation
│   ├── relu.py          # ReLU activation
│   ├── sequential.py    # Sequential container
│   └── losses/
│       └── cross_entropy.py  # Cross-entropy loss
├── optim/
│   └── sgd.py           # SGD optimizer
└── transforms/
    ├── compose.py       # Compose multiple transforms
    ├── to_numpy.py      # Convert to NumPy array
    ├── reshape.py       # Reshape image dimensions
    ├── flatten.py       # Flatten image to 1D
    └── normalize.py     # Normalize pixel values
```

## How It Works

### 1. Data Loading
The `MNISTDataset` class reads the original MNIST IDX format files directly using Python's `struct` module, without any external dependencies. Images are loaded on-demand and transformed through a pipeline.

### 2. Data Transformation
Images go through a transformation pipeline:
1. **ToNumpy**: Convert raw bytes to NumPy array
2. **Reshape**: Reshape to 28×28 matrix
3. **Flatten**: Flatten to 784-dimensional vector
4. **Normalize**: Scale pixel values from [0, 255] to [0, 1]

### 3. Forward Pass
- Input images are passed through the first Linear layer (784 → 128)
- ReLU activation introduces non-linearity
- Second Linear layer produces 10 output logits

### 4. Loss Computation
Cross-entropy loss measures the difference between predicted probabilities and true labels. The softmax function converts logits to probabilities.

### 5. Backward Pass
Automatic differentiation is implemented manually:
- Gradients flow from the loss back through each layer
- Each layer computes its own gradients (weights and bias)
- Gradients are stored in the `grad` attribute of each Parameter

### 6. Optimization
SGD updates each parameter using:
```
parameter.data = parameter.data - learning_rate * parameter.grad
```

## Usage

```bash
# Install dependencies
pip install numpy

# Run training
python main.py
```

## Configuration

Edit `config.py` to modify:
- `BATCH_SIZE`: Number of samples per batch (default: 32)
- `EPOCHS`: Number of training epochs (default: 10)
- `LEARNING_RATE`: Learning rate for SGD (default: 0.01)
- `HIDDEN_SIZE`: Hidden layer size (default: 128)

## Learning Outcomes

This project helps you understand:
- How neural networks work under the hood
- The mathematics behind forward and backward propagation
- How optimizers update model parameters
- The importance of data preprocessing
- How to implement automatic differentiation manually

## Requirements

- Python 3.7+
- NumPy

## License

MIT License