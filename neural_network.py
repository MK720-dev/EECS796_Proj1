#  _*_ coding: utf-8 _*_
""""
===============================================================================
Neural Network Class Skeleton for Project 1 (Math 796)
===============================================================================
Author:       Malek Kchaou
Course:       MATH 796 - Machine Learning & Optimization
Project:      Project 1 - Subproject 1
File:         neural_network.py
-------------------------------------------------------------------------------
Description:
    This class defines a modular architecture for building, training, and
    evaluating feedforward neural networks of arbitrary depth and width.
    The class is designed to integrate seamlessly with multiple optimization
    algorithms (SGD, BFGS, Adam, Adagrad, etc.) to compare their performance
    across shallow and deep networks as required by Subproject 1.

    The structure supports:
        • Dynamic construction of fully connected architectures.
        • Customizable activation functions and initialization strategies
          (He, Xavier, etc.).
        • Flexible training loops supporting k-fold cross-validation.
        • Integration with both custom and TensorFlow optimizers through
          a unified API.

    The design emphasizes transparency and educational clarity, allowing
    experimentation with different learning rate strategies, convergence
    behaviors, and generalization patterns.
-------------------------------------------------------------------------------
Created:      10-10-2025
Dependencies:
    - NumPy
    - Matplotlib
    - Scikit-learn for KFold cross-validation
    - (Optional) TensorFlow, for optimizer integration
-------------------------------------------------------------------------------
Notes:
    - All layer connections, activations, and gradients are implemented
      explicitly in NumPy to visualize the inner workings of backpropagation.
    - The modular design allows incremental development and testing of each
      component (forward pass, backpropagation, optimization, etc.).
===============================================================================
"""


from multiprocessing import Value
import numpy as np  # type: ignore
import matplotlib 
matplotlib.use('Agg') # disable GUI backend
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from optimizers import GradientDescent, BFGS, Adam, RMSProp
from optimizers import LineSearch

class NeuralNetwork:

    def __init__(self, input_size, hidden_layers, output_size, activation='relu', seed=None, leaky_alpha=0.01):
        """
        Initialize the neural network structure.
    
        Parameters
        __________
        input_size: int
            Number of input features.
        hidden_layers : list[int]
            List of integers specifying the number of neurons in each hidden layer.
        output_size : int
            Number of output neurons (depends on classification/regression task).
        activation : str
            Activation function to use ('relu', 'sigmoid', 'tanh', etc.).
        seed : int, optional
            Random seed for reproducibility.        
        
        """

        if seed:
            np.random.seed(seed)

        # Architecture
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        self.leaky_alpha = float(leaky_alpha)

        # Loss function choice
        self.loss_name = "mse" #default
        self.loss_fn = None
        self.loss_derivative_fn = None

        # Will be initialized later 
        self.layers = None 
        self.weights = {}
        self.biases = {}

        # Training placeholders
        self.loss_histories = []
        self.optimizer = None
        self.learning_rate = None

        # Build architecture 
        self._build_network()
    
    def _build_network(self):
        """Build layer structure and initialize weights/biases."""
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.layers = layer_sizes
        print(f"Network architecture: {layer_sizes}")

        # Initialize Parameters 
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize weights and biases according to chosen activation."""
        for i in range(1, len(self.layers)):
            n_in = self.layers[i-1]
            n_out = self.layers[i]

            if self.activation_name.lower() in  ['relu', 'leakyrelu']:
                # He initialization 
                std = np.sqrt(2.0/n_in)
            elif self.activation_name.lower() in ['sigmoid', 'tanh']:
                # Xavier initialization
                std = np.sqrt(2.0/(n_out + n_in))
            else:
                # Default small random initialization
                std = 0.01

            self.weights[f"W{i}"] = np.random.randn(n_out, n_in) * std
            self.biases[f"b{i}"] = np.zeros((n_out,1))

        print("Weights and biases initialized.")

    def _activation(self, z):
        """
        Compute activation based on selected type.
        ________
        Parameters:
            z: pre_activation vector for current layer 

        """
        if self.activation_name == 'relu':
            return np.maximum(0,z)
        elif self.activation_name == 'leakyrelu':
            a = self.leaky_alpha
            return np.where(z > 0, z, a*z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def _activation_derivative(self, z):
        """
        Compute activation derivative based on selected type.
        ________
        Parameters:
            z: pre_activation vector for current layer 
        """
        if self.activation_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_name == 'leakyrelu':
            a = self.leaky_alpha
            grad = np.ones_like(z)
            grad[z < 0] = a
            return grad
        elif self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z)**2
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    # ----------------------- LOSS FUNCTIONS -----------------------
    def _set_default_loss(self):
        """Set default loss to MSE."""
        self.set_loss("mse")

    def set_loss(self, loss_name=None):
        """
        Set the loss function and its derivative for training.

        Parameters
        ----------
        loss_name : str
            Either 'mse' or 'cross_entropy'
        """
        loss_name = loss_name.lower()
        if loss_name == None:
            loss_name = self.loss_name
        self.loss_name = loss_name

        if loss_name == "mse":
            # Mean Squared Error
            def loss_fn(y_pred, y_true):
                m = y_true.shape[1]
                return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

            def loss_derivative(y_pred, y_true):
                m = y_true.shape[1]
                return (1 / m) * (y_pred - y_true)
        elif loss_name == "cross_entropy":
            # Cross-Entropy for softmax outputs
            def loss_fn(y_pred, y_true):
                eps = 1e-12
                y_pred = np.clip(y_pred, eps, 1 - eps) # prevents taking log(0) since it's undefined 
                return -np.mean(np.sum(y_true * np.log(y_pred), axis = 0))

            def loss_derivative(y_pred, y_true):
                # For softmax + cross-entropy combination, derivative simplifies:
                return (y_pred - y_true) / y_true.shape[1]

        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

        self.loss_fn = loss_fn
        self.loss_derivative_fn = loss_derivative
        print(f"Loss function set to: {self.loss_name.upper()}")


    # ----------------------- FORWARD PASS -----------------------

    def forward(self, X):
        """
        Forward pass through all layers.
        Applies activation to all hidden layers.
        For the output layer:
            - uses linear output for regression (MSE loss)
            - uses sigmoid/softmax for classification (cross-entropy loss)

        Details:

        W_l * X_(l-1) + b_l = Z_l
        A_l = activation(Z_l)
        
        W_l is the weights matrix of the layer l
        X_(l-1) is the input matrix from previous layer (l-1) of shape n*m where n is the number of features 
        and m is the number of samples
        b_l is the bias vector that will be applied column wise to each sample in W_l * X_(l-1)
        Z_l is matrix of pre-activation column vectors where each column corresponds to its respective input 
        sample
        A_l is the matrix of the post-activation column vectors 
        _________
        Parameters:
            X: Input data matrix where columns are samples and rows are features
        ---------
        Returns:
            y_hat: final network output
            cache: dictionary with intermediate values for backprop
        """
        cache = {"A0": X}
        A = X

        L = len(self.layers) - 1

        for i in range(1, len(self.layers)):
            Z = np.dot(self.weights[f"W{i}"], A) + self.biases[f"b{i}"]
            if i < L:
                A = self._activation(Z)
            else:
                if self.loss_name == "cross_entropy":
                    # Multi-Class Softmax
                    if self.output_size > 1:
                        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) # as exponentials grow very fast and could cause overflow, to stabilize computation, we shift all logits by subtracting the maximum -> doesn't change result 
                        A = expZ / np.sum(expZ, axis=0, keepdims=True) #Softmax
                    # Binary Classification Sigmoid 
                    else:
                        A = 1 / (1 + np.exp(-Z)) # sigmoid
                elif self.loss_name == "mse":
                    A = Z
                else:
                    raise ValueError("Loss not set or unsupported loss type.")

            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = A
        
        return A, cache 

    # ----------------------- Backward PASS -----------------------

    def backward(self, X, y, cache):
        """
        Perform backpropagation to compute gradients of weights and biases.
        Parameters:
            X: Input data (n_input, m)
            y: True labels (n_output, m)
            cache: dictionary from forward pass containing A and Z values
        Returns:
            grads: dictionary containing dW and db for each layer
        """
        grads = {}
        m = X.shape[1]
        L = len(self.layers) - 1  # Number of weight layers 

        # Output Layer Error
        A_final = cache[f"A{L}"]
        Z_final = cache[f"Z{L}"]

        # Compute derivative of loss w.r.t. output activation
        dA = self.loss_derivative_fn(A_final, y)
        if self.loss_name == "cross_entropy":
            dZ = dA
        else:
            dZ = dA * self._activation_derivative(Z_final)
        grads[f"dW{L}"] = np.dot(dZ, cache[f"A{L-1}"].T)
        grads[f"db{L}"] = np.sum(dZ, axis=1, keepdims=True)

        # Propagate backward through hidden layers 
        for l in range(L-1, 0, -1):
            Z = cache[f"Z{l}"]
            dA = np.dot(self.weights[f"W{l+1}"].T, dZ)
            dZ = dA * self._activation_derivative(Z)
            grads[f"dW{l}"] = np.dot(dZ, cache[f"A{l-1}"].T)
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)

        return grads

    def compute_loss(self, y_pred, y_true):
        """
        Compute the Mean Squared Error (MSE) loss.
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted output of shape (n_output, m)
        y_true : np.ndarray
            True labels of shape (n_output, m)
        Returns
        -------
        float
            Mean Squared Error loss value
        """
        if self.loss_fn is None:
            raise ValueError("Loss function not set. Call set_loss('mse') or set_loss('cross_entropy').")
        return self.loss_fn(y_pred, y_true)
        
    def set_optimizer(self, optimizer):
        """
        Attach an optimizer object (e.g., SGD, Adam, BFGS) to this network.
        """
        self.optimizer = optimizer
        print(f"Optimizer set to: {type(optimizer).__name__}")

    def update_parameters(self, grads, lr):
        """
        Update parameters (weights and biases) using computed gradients.
        If an optimizer is provided, delegate update to it.
    
        Parameters
        ----------
        grads : dict
            Dictionary of gradients for each layer: dW1, db1, ...
        lr : float
            Learning rate for gradient descent (used only if no optimizer is set)
        """
        if self.optimizer is not None:
            # Optimizer will handle parameter updates 
            self.optimizer(self.weights, self.biases, grads)
        else:
            # Standard gradient descent update
            for i in range(1, len(self.layers)):
                self.weights[f"W{i}"] -= lr * grads[f"dW{i}"]
                self.biases[f"b{i}"] -= lr * grads[f"db{i}"]

    def train(self, X, y, epochs=1000, lr=0.01, batch_size=None, verbose=True):
        """
        Train the neural network using forward and backward propagation.
        Supports either manual updates or external optimizer.
        Supports stochastic and mini-batch updates. 
        
        Automatically adapts to the optimizer in use:
            - For first-order methods (SGD, Adam, etc.), uses batch or mini-batch updates.
            - For second-order methods (BFGS), enforces full-batch training.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_features, m)
        y : np.ndarray
            True labels of shape (n_outputs, m)
        epochs : int, optional
            Number of training iterations
        lr : float, optional
            Learning rate (ignored if optimizer is set)
        batch_size: int, optional
            Number of sample to be processed
        verbose : bool, optional
            Whether to print training progress
        """
        optimizer_name = type(self.optimizer).__name__ if self.optimizer else "None"
        print(f"\nTraining with optimizer: {optimizer_name}")
        print(f"Epochs: {epochs}, Learning rate: {lr}, Batch size: {batch_size}\n")

        loss_history = []

        #------------------------------ BFGS MODE------------------------------------
        if isinstance(self.optimizer, BFGS):
            print("Using full-batch mode for BFGS (batch_size ignored).")

            for epoch in range(1, epochs + 1):
                self.optimizer(self, X, y)
                y_pred, _ = self.forward(X)
                loss = self.compute_loss(y_pred, y)
                loss_history.append(loss)

                if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                    print(f"Epoch {epoch}/{epochs} | Loss: {loss:.6f}")

        #------------------------------ FIRST ORDER MODE------------------------------------
        else:
            m = X.shape[1]

            for epoch in range(1, epochs+1):

                indices = np.arange(m)
                np.random.shuffle(indices)

                if batch_size is None:
                    batch_size = m # default to full batch GD

                for start in range(0, m, batch_size):

                    end = start + batch_size
                    batch_idx = indices[start:end]
                    X_batch, y_batch = X[:, batch_idx], y[:, batch_idx]

                    # Forward Pass
                    y_pred, cache = self.forward(X_batch)
                    # Backward pass 
                    grads = self.backward(X_batch, y_batch, cache)

                    # Parameter update
                    self.update_parameters(grads, lr)

                # Compute Loss
                y_pred_full, _ = self.forward(X)
                loss = self.compute_loss(y_pred_full, y)
                loss_history.append(loss)

                # Oprional progress printout 
                if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                    print(f"Epoch {epoch}/{epochs} | Loss: {loss:.6f}") 
            
        self.loss_histories.append(loss_history)
        print("Training complete.")


    def cross_validate(self, X, y, k=5, epochs=1000, lr=0.01, batch_size=None, seed=None, verbose=True):
        """
        Perform k-fold cross-validation on the dataset.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_features, m)
        y : np.ndarray
            True labels of shape (n_outputs, m)
        k : int
            Number of folds (default: 5)
        epochs : int
            Number of epochs for each fold
        lr : float
            Learning rate
        batch_size: int, optional
            Number of sample to be processed
            - If None or >= m → Full batch gradient descent
            - If 1 → Stochastic gradient descent
            - Otherwise → Mini-batch gradient descent
        seed : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress

        Returns
        -------
        dict
            Dictionary containing per-fold and average training/validation losses.
        """

        if seed is not None:
            np.random.seed(seed)

        m = X.shape[1]
        indices = np.arange(m)
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        fold_results = {
                        "K": k,
                        "train_losses": [], 
                        "val_losses": [], 
                        "loss_histories": [], 
                        "avg_train_loss": None, 
                        "avg_val_loss": None,
                        "std_val_loss": None,
                        "metric_name": None, 
                        "train_metrics": [], 
                        "val_metrics": [],
                        "avg_train_metric": None,
                        "avg_val_metric": None
                        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
            # Split into training and validation
            X_train, X_val = X[:, train_idx], X[:, val_idx]
            y_train, y_val = y[:, train_idx], y[:, val_idx]
            
            # Re-initialize parameters for each fold
            self._initialize_parameters()

            # Train on this fold's training data
            self.train(X_train, y_train, epochs=epochs, lr=lr, batch_size=batch_size, verbose=False)
            
            # Compute Losses
            y_train_pred = self.predict(X_train)
            y_val_pred = self.predict(X_val)
            train_loss = self.compute_loss(y_train_pred, y_train)
            val_loss = self.compute_loss(y_val_pred, y_val)

            # Compute accuracy for classification tasks for current fold
            if self.loss_name == "cross_entropy":
                metric_name = "accuracy"
                train_metric = self.compute_accuracy(y_train_pred, y_train)
                val_metric  = self.compute_accuracy(y_val_pred, y_val)
                fold_results["metric_name"] = metric_name
                fold_results["train_metrics"].append(train_metric)
                fold_results["val_metrics"].append(val_metric)
            elif self.loss_name == "mse":
                metric_name = "mse"
                fold_results["metric_name"] = metric_name
                fold_results["train_metrics"].append(train_loss)
                fold_results["val_metrics"].append(val_loss)

            else:
                metric_name = None
                train_metric = None
                val_metric = None

            fold_results["train_losses"].append(train_loss)
            fold_results["val_losses"].append(val_loss)
            

            if verbose:
                print(f"Fold {fold}/{k} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n")
        
        # Collecting loss histories after training is done
        fold_results["loss_histories"] = self.loss_histories.copy()

        # Compute average and variance across folds
        avg_train = np.mean(fold_results["train_losses"])
        avg_val = np.mean(fold_results['val_losses'])
        std_val = np.std(fold_results['val_losses'])
        fold_results["avg_train_loss"] = avg_train
        fold_results["avg_val_loss"] = avg_val
        fold_results["std_val_loss"] = std_val

        # Computer average training and validation metric values
        avg_train_metric = np.mean(fold_results["train_metrics"])
        avg_val_metric = np.mean(fold_results["val_metrics"])
        fold_results["avg_train_metric"] = avg_train_metric
        fold_results["avg_val_metric"] = avg_val_metric

        if verbose:
            print(f"\nCross-validation training summary (K={k}):")
            print(f"Average Train Loss: {avg_train:.6f}")
            print(f"Average Val Loss: {avg_val:.6f} +/- {std_val:.6f}")

        return fold_results

    def predict(self, X):
        """
        Make predictions after training.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_features, m)
        Returns
        -------
        np.ndarray
            Predicted outputs of shape (n_outputs, m)
        """
        y_pred, _ = self.forward(X)
        return y_pred

    def compute_accuracy(self, y_pred, y_true):
        """
        Compute classification accuracy.
        For one-hot encoded targets, takes the argmax of predictions and labels.
        """
        # multi-class one-hot encoded prediction 
        if y_true.shape[0] > 1:
            y_pred_labels = np.argmax(y_pred, axis=0)
            y_true_labels = np.argmax(y_true, axis=0)
        else:
            y_pred_labels = (y_pred > 0.5).astype(int)
            y_true_labels = y_true.astype(int)

        return np.mean(y_pred_labels==y_true_labels)

    def plot_loss(self, fold_idx=None, title="Loss Convergence", show=True, save_path=None):
        """
        Plot loss convergence across one or multiple training runs (folds).

        Parameters
        ----------
        fold_idx : int or None
            If None, plots all folds; otherwise plots only the given fold (1-indexed).
        title : str
            Title of the plot.
        show : bool
            Whether to display the plot.
        save_path : str or None
            File path to save the plot image.
        """
        if not hasattr(self, "loss_histories") or len(self.loss_histories) == 0:
            print("No training history found. Train the model before plotting.")
            return

        plt.figure(figsize=(9, 5))

        if fold_idx is None:
            # Plot all folds
            for i, losses in enumerate(self.loss_histories, 1):
                plt.plot(losses, label=f"Fold {i}")
        else:
            # Plot only selected fold (1-indexed)
            if 1 <= fold_idx <= len(self.loss_histories):
                plt.plot(self.loss_histories[fold_idx - 1], label=f"Fold {fold_idx} (Training)")
            else:
                print("Invalid fold index.")
                return

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Loss plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
