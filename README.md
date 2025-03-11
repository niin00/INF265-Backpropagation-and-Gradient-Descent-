# INF265-Backpropagation-and-Gradient-Descent-

# Manual Backpropagation & Gradient Descent

## Overview

This repository contains the implementation and experiments for a deep learning project that demonstrates a custom implementation of backpropagation and gradient descent. The project focuses on a simplified version of the CIFAR-10 dataset by distinguishing between two classes: "cat" and "car." Both manual and built-in training routines (using PyTorch’s SGD) were implemented, and the results were verified through gradient checking.

**Author:** Ninad Hagi  
**Date:** 21 February 2025

## Project Details

### Objectives
- **Backpropagation Implementation:**  
  Develop a custom neural network (`MyNet`) that stores intermediate tensors and computes gradients manually.  
  - Implemented manual gradient computation following standard equations.
  - Verified the gradients by comparing with PyTorch’s autograd (`loss.backward()`) and through finite-difference gradient checking.
  
- **Gradient Descent Training:**  
  Train a multilayer perceptron (MLP) model (`MyMLP`) on a two-class subset of CIFAR-10.
  - Model architecture: Input size 768 (16×16×3), two hidden layers (128 and 32 units) with ReLU activations, and an output layer with 2 units.
  - Two training approaches were used:  
    1. **Built-in PyTorch Optimizer:** Using `torch.optim.SGD`.
    2. **Manual Updates:** Explicitly applying the weight-update equation.

### Key Results
- **Hyperparameters:**
  - **Batch Size:** 256
  - **Epochs:** 30
  - **Learning Rate:** 0.01
  - **Momentum:** 0
  - **Weight Decay:** 0

- **Performance:**
  - **Training Accuracy:** 97%
  - **Validation Accuracy:** 91%
  - **Test Accuracy:** 91%

Both training methods yielded equivalent results, validating the accuracy of the manual backpropagation and update routines.

## Repository Structure

```
INF265_Project_1/
├── README.md          # This file
├── src/               # Source code directory
│   ├── mynet.py       # Custom neural network implementation (MyNet)
│   ├── mymlp.py       # MLP model implementation (MyMLP)
│   ├── train.py       # Training scripts for both SGD and manual updates
│   └── utils.py       # Utility functions including backpropagation and gradient checking
└── data/              # Data handling and dataset (subset of CIFAR-10)
```

*Note: Adjust the structure as needed based on your organization.*

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/INF265_Project_1.git
   cd INF265_Project_1
   ```

2. **Set Up a Virtual Environment and Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install PyTorch (if not already installed):**
   ```bash
   pip install torch torchvision
   ```

## Usage

- **Training with PyTorch’s SGD:**
  ```bash
  python src/train.py --mode sgd
  ```

- **Training with Manual Updates:**
  ```bash
  python src/train.py --mode manual
  ```

For more configuration options, run:
```bash
python src/train.py --help
```

## Detailed Implementation

### Backpropagation
- **Custom Neural Network (MyNet):**
  - Stores layer-wise tensors (`z[l]`, `a[l]`) and computes forward passes.
- **Manual Gradient Computation:**
  - Implements the gradient calculation for weights and biases.
  - Validated via PyTorch’s autograd and finite-difference gradient checking.
- **Loss Function:**
  - Mean Squared Error (MSE) was used for testing gradient computations.

### Gradient Descent
- **Dataset Preparation:**
  - CIFAR-10 subset: Only “cat” and “car” images resized to 16×16.
  - Deterministic splits (shuffle disabled) for reproducibility.
- **MLP Architecture (MyMLP):**
  - Input dimension: 768
  - Hidden layers: 128 and 32 units with ReLU activation.
  - Output layer: 2 units (no activation).
- **Training Approaches:**
  - **Built-in:** Using `torch.optim.SGD` to update parameters.
  - **Manual:** Updating parameters via:  
    `θ ← θ - α (∇θL + λθ)`
- **Results:**
  - Both training methods achieved nearly identical accuracy, confirming the correctness of the manual implementation.

## Contributing

Contributions are welcome! Please fork the repository and open a pull request for any improvements, bug fixes, or new features.
