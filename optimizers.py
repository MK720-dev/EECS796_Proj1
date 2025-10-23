"""
===============================================================================
Optimizers Module
===============================================================================
Author:       Malek Kchaou
Course:       MATH 796 - Machine Learning & Optimization
Project:      Project 1 
File:         optimizers.py
-------------------------------------------------------------------------------
Description:
This module defines the structure for all optimizer classes used in training
artificial neural networks for Subproject 1: "Compare Learning Rate Techniques
and Deep vs Shallow Networks."

It provides:
    • An abstract Optimizer base class defining the common interface for all
      optimization algorithms (step() and reset_state()).
    • Custom NumPy-based implementations of optimizers such as SGD and Adam.
    • A factory class (OptimizerFactory) to dynamically create either custom
      or TensorFlow optimizers based on user parameters.

This modular design allows seamless switching between the different hand-coded algorithms
-------------------------------------------------------------------------------
Usage Example:
    from optimizers import OptimizerFactory

    # Create custom (NumPy) Adam optimizer
    opt = OptimizerFactory.create("adam", lr=0.001, use_tensorflow=False)

    # During training:
    opt.step(weights, biases, grads)
-------------------------------------------------------------------------------
Created: 10-12-2025
Dependencies:
    - NumPy
    - Inspect
-------------------------------------------------------------------------------
Notes:
    - All optimizers conform to a common API:
          step(weights, biases, grads)
          reset_state()
    - This structure ensures compatibility with both shallow and deep network
      architectures implemented in the NeuralNetwork class.
===============================================================================
"""

import numpy as np
import inspect

# Optional TensorFlow import (guarded for environments without TensorFlow)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================================
# Base Optimizer Interface
# ============================================================================

class Optimizer:
    """
    Abstract base class for all optimizers.

    Defines a consistent interface used by NeuralNetwork:
        - step(weigths, biases, grads)
        - reset_state()
    """

    def __init__(self, lr=0.01):
        """
        Initialize the optimizer.

        Parameters
        ----------
        lr : float
            Learning rate.
        """
        self.lr = lr

    def step(self, weights, biases, grads):
        """
        Perform one optimization step (update parameters).

        Parameters
        ----------
        weights : dict[str, np.ndarray]
            Model weights (e.g., "W1", "W2", ...).
        biases : dict[str, np.ndarray]
            Model biases (e.g., "b1", "b2", ...).
        grads : dict[str, np.ndarray]
            Gradients computed during backpropagation (e.g., "dW1", "db1", ...).
        """

        raise NotImplementedError("Optimizer step() must be implemented by subclasses.")

    def reset_state(self):
        """
        Reset any stateful parameters (e.g., momentum, moving averages).
        Called at the start of each new training session or fold.
        """
        pass

# ============================================================================
# Custom Optimizers 
# ============================================================================
class GradientDescent(Optimizer):
    """
    Basic (Vanilla) Gradient Descent Optimizer.
    -------------------------------------------------------------------
    Applies parameter updates based on provided gradients.
    Works with stochastic, mini-batch, or full-batch modes depending
    on the batch_size used in the training loop.
    """

    def __init__(self, lr=0.01):
        """
        Initialize the GradientDescent optimizer.

        Parameters
        ----------
        lr : float
            Fixed learning rate η for all updates.
        """
        super().__init__(lr)

    #-------------------------------------------------------------------------
    def step(self, weights, biases, grads):
        """
        Perform one GradientDescent update using computed gradients.

        Parameters
        ----------
        weights : dict[str, np.ndarray]
            Model weights { "W1": ..., "W2": ... }.
        biases : dict[str, np.ndarray]
            Model biases { "b1": ..., "b2": ... }.
        grads : dict[str, np.ndarray]
            Gradient dictionary from backprop { "dW1": ..., "db1": ... }.
        """
        for i in range(1, len(weights)+1):
            weights[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            biases[f"b{i}"] -= self.lr * grads[f"db{i}"]

    def __call__(self, weights, biases, grads):
        """
        Make the optimizer callable, so it can be used as:
            self.optimizer(self.weights, self.biases, grads)
        """
        self.step(weights, biases, grads)


class BFGS(Optimizer):
    """
    BFGS Optimizer (Quasi-Newton)
    -----------------------------
    Uses approximate inverse Hessian updates and line search to perform
    second-order optimization without explicitly computing the Hessian.
    """
    def __init__(self, line_search, tol=1e-5, max_iter=100):
        """
        Parameters
        ----------
        line_search : LineSearch
            LineSearch instance providing step sizes via Wolfe conditions.
        tol : float
            Tolerance for gradient norm stopping criterion.
        max_iter : int
            Maximum number of quasi-Newton updates.
        """
        self.line_search = line_search
        self.tol = tol 
        self.max_iter = max_iter 


    # ---------------------------------------------------------
    def _flatten_params(self, weights, biases):
        """Flatten parameter dicts into a single vector."""
        flat = []
        for i in range(1, len(weights) + 1):
            flat.append(weights[f"W{i}"].ravel())
            flat.append(biases[f"b{i}"].ravel())
        return np.concatenate(flat)

    def _unflatten_params(self, x, template_weights, template_biases):
         """Reconstruct dicts of W and b from flattened vector."""
         weights, biases = {}, {}
         pos = 0
         for i in range(1, len(template_weights) + 1):
             w_shape = template_weights[f"W{i}"].shape
             b_shape = template_biases[f"b{i}"].shape
             w_size = np.prod(w_shape)
             b_size = np.prod(b_shape)
             weights[f"W{i}"] = x[pos:pos+w_size].reshape(w_shape)
             pos += w_size
             biases[f"b{i}"] = x[pos:pos+b_size].reshape(b_shape)
             pos += b_size

         return weights, biases

    # ---------------------------------------------------------
    def step(self, network, X, y):
        """
        Perform one BFGS optimization pass on the neural network.
        This method assumes network.forward/backward compute f(x) and g(x).
        """
        # Flatten current parameters 
        x = self._flatten_params(network.weights, network.biases)

        # Initial function value and gradient 
        y_pred, cache = network.forward(X)
        loss = network.compute_loss(y_pred, y)
        grads = network.backward(X, y, cache)
        L = len(network.layers) - 1
        grads_W = {f"W{i}": grads[f"dW{i}"] for i in range(1, L+1)}
        grads_b = {f"b{i}": grads[f"db{i}"] for i in range(1, L+1)}
        g = self._flatten_params(grads_W, grads_b)

        n = len(x)
        H = np.eye(n)

        for k in range(self.max_iter):

            # Search direction
            p = -H @ g

            # Line search (using the LineSearch class)
            alpha = self.line_search.find_alpha(
                f=lambda z: self._evaluate_loss(network, X, y, z),
                grad_f=lambda z: self._evaluate_grad(network, X, y, z, p),
                x = x, 
                p = p
                )

            # Step
            x_new = x + alpha * p

            # Compute new loss and gradient 
            loss_new, g_new = self._compute_loss_and_grad(network, X, y, x_new)

            # Convergence check 
            if np.linalg.norm(g_new) < self.tol:
                print(f"Converged at iteration {k} with |g|={np.linalg.norm(g_new):.2e}")
                break

            # BFGS update 
            s = x_new - x
            yk = g_new - g 
            rho = 1.0 / (yk.T @ s + 1e-12)

            I = np.eye(n)
            H = (I - rho * np.outer(s, yk)) @ H @ (I - rho * np.outer(yk, s)) + rho * np.outer(s, s)

            # Update state 
            x, g = x_new, g_new

        network.weights, network.biases = self._unflatten_params(x, network.weights, network.biases)

    # ---------------------------------------------------------
    def _compute_loss_and_grad(self, network, X, y, x):
        """Compute loss and flattened gradient given vector x."""
        # Unflatten vector x back into weight and bias dictionaries
        weights, biases = self._unflatten_params(x, network.weights, network.biases)
        network.weights = weights
        network.biases = biases

        # Compute forward, loss, and gradients at these parameters
        y_pred, cache = network.forward(X)
        loss = network.compute_loss(y_pred, y)
        grads = network.backward(X, y, cache)

        # Flatten gradients into a single vector
        L = len(network.layers) - 1
        grads_W = {f"W{i}": grads[f"dW{i}"] for i in range(1, L+1)}
        grads_b = {f"b{i}": grads[f"db{i}"] for i in range(1, L+1)}
        g = self._flatten_params(grads_W, grads_b)

        return loss, g

    def _evaluate_loss(self, network, X, y, x):
        """Helper for line search to evaluate f(x + αp)."""
        weights, biases = self._unflatten_params(x,  network.weights, network.biases)
        network.weights = weights
        network.biases = biases

        y_pred, _ = network.forward(X)
        return network.compute_loss(y_pred, y)

    def _evaluate_grad(self,network, X, y, x, p):
        """Helper for line search directional derivative."""
        _, g = self._compute_loss_and_grad(network, X, y, x)
        return g 

    # ---------------------------------------------------------
    def __call__(self, network, X, y):
        """Make optimizer callable to integrate with NeuralNetwork."""
        self.step(network, X, y)


class Adam(Optimizer):
    """
    Adam Optimizer
    Based on: Kingma & Ba (2015) - Adam: A Method for Stochastic Optimization
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0    
        self.m_w, self.v_w = {}, {}
        self.m_b, self.v_b = {}, {}

    def step(self, weights, biases, grads):
        """
        Args:
            weights (dict): e.g., {"W1": ..., "W2": ...}
            biases  (dict): e.g., {"b1": ..., "b2": ...}
            grads (dict):   e.g., {"dW1": ..., "db1": ..., ...}
        Returns:
            updated weights and biases
        """
        if not self.m_w:
            # Initialize moments as zero arrays
            for key in weights.keys():
                self.m_w[key] = np.zeros_like(weights[key])
                self.v_w[key] = np.zeros_like(weights[key])
            for key in biases.keys():
                self.m_b[key] = np.zeros_like(biases[key])
                self.v_b[key] = np.zeros_like(biases[key])

        self.t += 1
        # === Update weights ===
        for key in weights.keys():
            dkey = "d" + key  # e.g., "dW1"
            g = grads[dkey]

            # Moments
            self.m_w[key] = self.beta1 * self.m_w[key] + (1 - self.beta1) * g
            self.v_w[key] = self.beta2 * self.v_w[key] + (1 - self.beta2) * (g ** 2)

            # Bias Correction
            m_hat = self.m_w[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w[key] / (1 - self.beta2 ** self.t)

            # Parameter update
            weights[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # === Update biases ===
        for key in biases.keys():
            dkey = "d" + key  # e.g., "db1"
            g = grads[dkey]

            # Moments
            self.m_b[key] = self.beta1 * self.m_b[key] + (1 - self.beta1) * g
            self.v_b[key] = self.beta2 * self.v_b[key] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m_b[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b[key] / (1 - self.beta2 ** self.t)

            # Parameter update
            biases[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights, biases

     # ---------------------------------------------------------
    def __call__(self, weights, biases, grads):
        """Make optimizer callable to integrate with NeuralNetwork."""
        self.step(weights, biases, grads)

class RMSProp(Optimizer):
    """
    RMSProp optimizer for neural networks with separate weight and bias dictionaries.
    Based on: G. Hinton, "Neural Networks for Machine Learning" (Coursera, 2012)
    """
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon

        # Exponential moving averages of squared gradients
        self.sq_w = {}  # for weights
        self.sq_b = {}  # for biases

    def step(self, weights, biases, grads):
        """
        Perform one RMSProp update step.

        Args:
            weights (dict): e.g., {"W1": ..., "W2": ...}
            biases  (dict): e.g., {"b1": ..., "b2": ...}
            grads (dict):   e.g., {"dW1": ..., "db1": ..., ...}
        Returns:
            updated weights and biases
        """
        #Initialize running averages if needed
        if not self.sq_w:
            for key in weights.keys():
                self.sq_w[key] = np.zeros_like(weights[key])
            for key in biases.keys():
                self.sq_b[key] = np.zeros_like(biases[key])

        # === Update weights ===
        # === Update weights ===
        for key in weights.keys():
            dkey = "d" + key
            g = grads[dkey]

            # Update exponential moving average of squared gradients
            self.sq_w[key] = self.beta * self.sq_w[key] + (1 - self.beta) * (g ** 2)

            # Update weights
            weights[key] -= self.lr * g / (np.sqrt(self.sq_w[key]) + self.epsilon)

        # === Update biases ===
        for key in biases.keys():
            dkey = "d" + key
            g = grads[dkey]

            # Update exponential moving average of squared gradients
            self.sq_b[key] = self.beta * self.sq_b[key] + (1 - self.beta) * (g ** 2)

            # Update biases
            biases[key] -= self.lr * g / (np.sqrt(self.sq_b[key]) + self.epsilon)

        return weights, biases

    #-------------------------------------------------------------------
    def __call__(self, weights, biases, grads):
        """Make optimizer callable to integrate with NeuralNetwork."""
        self.step(weights, biases, grads)


# ============================================================================
# Optimizer Factory
# ============================================================================
class OptimizerFactory:
    """
    Factory class for creating optimizer instances.

    Supports both:
        - Custom NumPy-based optimizers (SGD, Adam, etc.)
        - TensorFlow’s built-in optimizers (SGD, Adam, RMSprop, Adagrad)
    """

    @staticmethod
    def _safe_init(OptimizerClass, **kwargs):
        """
        Dynamically filters keyword arguments to match the optimizer’s constructor.
        This prevents TypeErrors when passing unused parameters.
        """
        valid_args = inspect.signature(OptimizerClass.__init__).parameters
        filtered = {k:v for k, v in kwargs.items() if k in valid_args}
        return OptimizerClass(**filtered)

    @staticmethod
    def create(optimizer_name="GradientDescent", **kwargs):
        """
        Create and return an optimizer instance.

        Parameters
        ----------
        
        Create an optimizer instance (custom or TensorFlow).

        Parameters
        ----------
        optimizer_name : str
            Name of optimizer ('gradientdescent', 'adam', 'rmsprop', 'bfgs', etc.)
        **kwargs :
            Any optimizer parameters (lr, line_search, tol, max_iter, etc.)
            Irrelevant ones will be ignored automatically.

        Returns
        -------
        Optimizer
            An instance of the requested optimizer.
        
        """
        name = optimizer_name.lower()

        # ------------------------------
        # Custom Optimizers
        # ------------------------------
        custom_optimizers = {
            "gradientdescent": GradientDescent,
            "adam": Adam,
            "rmsprop": RMSProp,
            "bfgs": BFGS
        }

        if name not in custom_optimizers:
            raise ValueError(f"Custom optimizer '{optimizer_name}' not recognized.")

        return OptimizerFactory._safe_init(custom_optimizers[name], **kwargs)

# ============================================================================
# Line Search Utility
# ============================================================================
class LineSearch:
    """
    Implements Backtracking and Wolfe line search to find adaptive step sizes
    for gradient-based optimizers (e.g., Gradient Descent, MiniBatchGD, BFGS).
    """
    def __init__(self, c1=1e-4, c2=0.9, tau=0.5, max_iter=50, use_wolfe=False):
        """
        Parameters
        ----------
        c1 : float
            Armijo condition constant (sufficient decrease).
        c2 : float
            Curvature condition constant (used if use_wolfe=True).
        tau : float
            Reduction factor for step size during backtracking (e.g., 0.5 halves the step each iteration).
        max_iter : int
            Maximum number of backtracking steps.
        use_wolfe : bool
            Whether to enforce Wolfe conditions (if False, Armijo condition only).
        """
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.max_iter = max_iter
        self.use_wolfe = use_wolfe

    def find_alpha(self, f, grad_f, x, p, f_x=None, grad_x=None, alpha_init=1.0):
        """
        Perform line search with interpolation to find alpha satisfying Armijo
        (and optionally Wolfe) conditions.

        Parameters
        ----------
        f : callable
            Objective function f(x).
        grad_f : callable
            Gradient function ∇f(x).
        x : np.ndarray
            Current parameter vector.
        p : np.ndarray
            Descent direction (typically -∇f(x) or calculated via Quasi-Newton).
        f_x : float, optional
            f(x) if already known.
        grad_x : np.ndarray, optional
            ∇f(x) if already known.
        alpha_init : float, optional
            Initial step length (default = 1.0).

        Returns
        -------
        float
            Step length α satisfying Armijo (and optionally Wolfe) conditions.
        """
        phi0 =  f_x if f_x is not None else f(x)
        g0 = grad_x if grad_x is not None else grad_f(x)
        dphi0 = np.dot(g0.T, p) # directional derivative at alpha = 0

        alpha_prev, phi_prev = 0.0, phi0
        alpha = alpha_init
        phi = f(x + alpha * p)

        for i in range(self.max_iter):
            # Armijo sufficient decrease 
            if phi <= phi0 + self.c1 * alpha * dphi0:
                if not self.use_wolfe:
                    return alpha
                # Wolfe curvature condition
                g_new = grad_f(x + alpha * p)
                dphi = np.dot(g_new.T, p)
                if dphi >= self.c2 * dphi0:
                    return alpha

            # ---- Quadratic or cubic interpolation step ----
            if i == 0:
                # Quadratic model on first iteration
                alpha_new =  - (dphi0 * alpha**2) / (2 * (phi - phi0 - dphi0 * alpha))
            else:
                # Cubic model (safer, uses previous point)
                alpha_new = self._cubic_interpolate(alpha_prev, phi_prev, alpha, phi, dphi0)

            # Safeguards: keep alpha_new inside reasonable bounds
            alpha_new = max(0.1 * alpha, min(alpha_new, 0.5 * alpha))

            # Prepare next iteration
            alpha_prev, phi_prev = alpha, phi 
            alpha = alpha_new
            phi = f(x + alpha * p)

        return alpha  # Fallback if no Armijo/Wolfe condition satisfied


    # -----------------------------------------------------------------------
    def _cubic_interpolate(self, a, fa, b, fb, fprime0):
        """
        Perform cubic interpolation between two points (a, fa), (b, fb)
        with slope f'(0) = fprime0.
        Returns the new alpha estimate minimizing the cubic model.
        """
        d1 = fprime0 + (fb - fa) / (b - a)
        d2 = (d1**2 - fprime0 * (fb - fa) / (b - a)) ** 0.5
        if np.isnan(d2):
            return b / 2  # fallback to halving if numerical issue
        alpha_cubic = b - (b - a) * ((fb - fa) / ((fb - fa) + (b - a) * (d2 - d1)))
        return alpha_cubic





                    
