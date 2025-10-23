#  _*_ coding: utf-8 _*_
"""
===============================================================================
Optimizer Comparison Script
===============================================================================
Author:       Malek Kchaou
Course:       MATH 796 - Machine Learning & Optimization
Project:      Project 1
File:         test_optimizer_comparison.py
-------------------------------------------------------------------------------
Description:
This script benchmarks and compares the performance of custom NumPy-based 
optimizers (GradientDescent, BFGS, Adam, RMSProp) against TensorFlow's 
built-in optimizers using the OptimizerFactory.

It evaluates:
    - Full-batch, stochastic (batch_size=1), and mini-batch (batch_size=20)
      variants of Gradient Descent.
    - Quasi-Newton (BFGS), Adam, and RMSProp custom implementations.
    - TensorFlow's SGD, Adam, and RMSProp optimizers for reference.
    - Loss convergence speed and stability on a synthetic regression task.

The results are visualized via Matplotlib loss curves, allowing direct
comparison of convergence behavior and final training performance between
hand-coded and TensorFlow optimizers.
-------------------------------------------------------------------------------
Usage Example:
    # Run directly from terminal
    $ python test_optimizer_comparison.py

    # Expected output:
    - Console summary of optimizer performance
    - Loss convergence plots for all optimizers
-------------------------------------------------------------------------------
Created: 10-17-2025
Dependencies:
    - NumPy
    - TensorFlow
    - Matplotlib
    - optimizer_factory.py
    - optimizers.py
-------------------------------------------------------------------------------
Notes:
    - The test will be carried on shallow and deep networks using different datasets
    - The datasets are the following:
        - A synthetic regression dataset
        - Higham's "Deep Learning: An Introduction for Applied Mathematicians" dataset
        - A chosen Kaggle dataset
    - This script supports reproducible experiments by separating optimizer 
      testing logic from implementation code.
===============================================================================
"""


from neural_network import NeuralNetwork
from optimizers import OptimizerFactory, LineSearch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=only errors

import re


def generate_synthetic_data(n_samples=200, seed=42):
    np.random.seed(seed)
    X = np.linspace(-3, 3, n_samples).reshape(-1,1)
    y = 2*X + np.sin(X) + 0.3 * np.random.rand(n_samples, 1)
   
    return X.T, y.T, "Synthetic"

def load_higham_data():
    # Input features
    x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
    x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])
    X = np.vstack((x1, x2))   # shape (2, 10)

    # Labels: first half 0s, second half 1s
    n = X.shape[1]
    y = np.hstack((np.zeros((1, n//2)), np.ones((1, n//2))))

    return X, y, "Higham Toy"

def load_kaggle_dataset(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values.reshape(-1,1)

    le = LabelEncoder()
    y_int = le.fit_transform(y_raw.ravel())
    num_classes = len(np.unique(y_int))
    y = np.eye(num_classes)[y_int] # Transform label for each sample from simple int to vector of shape (num_classes, 1) 

    return X.T, y.T, "Kaggle Dataset"

def load_all_datasets():
    datasets = []
    datasets.append(generate_synthetic_data())
    datasets.append(load_higham_data())

    kaggle_path = "UCI_Wall_Following_Robot_Sensor_Dataset\sensor_readings_24.csv"
    try:
        datasets.append(load_kaggle_dataset(kaggle_path))
    except FileNotFoundError:
        print(f"[Warning] Kaggle dataset not found at {kaggle_path}. Skipping.")
    return datasets

def run_tensorflow_experiment_cv(X, y, dataset_name, arch, optimizer_name,
                                 epochs=300, batch_size=32, lr=0.01, k=5, seed=42):
    """
    Run TensorFlow experiment with K-fold cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_features, n_samples)
    y : np.ndarray
        Labels or regression targets (n_outputs, n_samples)
    dataset_name : str
        Dataset name for reporting.
    arch : list[int]
        Hidden layer architecture, e.g. [64, 32, 16].
    optimizer_name : str
        One of ['sgd', 'adam', 'rmsprop', 'adagrad'].
    epochs : int
        Training epochs per fold.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    k : int
        Number of folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Summary dictionary with average and std of train/val losses.
    """

    # ----------------------------------------------------------
    # Prepare data (transpose to (samples, features))
    # ----------------------------------------------------------
    X_tf = X.T
    y_tf = y.T
    print(f"Input Shape: {X_tf.shape} | Output Shape: {y_tf.shape}")
    print(f"Expected number of samples: {X_tf.shape[0]}")
    n_outputs = y_tf.shape[1]

    if n_outputs == 1:
        loss_fn = "mse"
        metrics = ["mse"]
        is_classification = False
    else:
        loss_fn = "categorical_crossentropy"
        metrics = ["accuracy"]
        is_classification = True

    # ----------------------------------------------------------
    # Setup cross-validation
    # ----------------------------------------------------------
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    fold_train_losses = []
    fold_val_losses = []
    fold_train_metrics = []
    fold_val_metrics = []
    fold_histories = []

    # ----------------------------------------------------------
    # Loop through folds
    # ----------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tf), start=1):
        print(f"   → Fold {fold}/{k}")

        X_train, X_val = X_tf[train_idx], X_tf[val_idx]
        y_train, y_val = y_tf[train_idx], y_tf[val_idx]

        # ------------------------------------------------------
        # Build model anew for each fold
        # ------------------------------------------------------
        model = Sequential()
        model.add(Dense(arch[0], activation="tanh", input_shape=(X_tf.shape[1],)))
        for h in arch[1:]:
            model.add(Dense(h, activation="tanh"))

        if is_classification:
            model.add(Dense(n_outputs, activation="softmax"))
        else:
            model.add(Dense(n_outputs, activation=None))

        # Optimizer selection
        optimizers = {
            "sgd": tf.keras.optimizers.SGD(learning_rate=lr),
            "gradientdescent": tf.keras.optimizers.SGD(learning_rate=lr), #alias for sgd
            "adam": tf.keras.optimizers.Adam(learning_rate=lr),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=lr),
            "adagrad": tf.keras.optimizers.Adagrad(learning_rate=lr)
        }
        if optimizer_name not in optimizers:
            raise ValueError(f"Unsupported TensorFlow optimizer: {optimizer_name}")

        model.compile(optimizer=optimizers[optimizer_name],
                      loss=loss_fn,
                      metrics=metrics)

        # ------------------------------------------------------
        # Train on this fold
        # ------------------------------------------------------
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0
        )

        # Final training and validation loss
        fold_train_loss = history.history["loss"][-1]
        fold_val_loss = history.history["val_loss"][-1]


        metric_name = metrics[0]
        fold_train_metric = history.history[metric_name][-1]
        fold_val_metric = history.history[f"val_{metric_name}"][-1]

        fold_train_losses.append(fold_train_loss)
        fold_val_losses.append(fold_val_loss)
        fold_train_metrics.append(fold_train_metric)
        fold_val_metrics.append(fold_val_metric)

        # Store full loss history for convergence plotting
        fold_histories.append((history.history["loss"], history.history["val_loss"]))

    # ----------------------------------------------------------
    # Aggregate results
    # ----------------------------------------------------------
    return {
        "Dataset": dataset_name,
        "Architecture": str(arch),
        "Optimizer": f"{optimizer_name} (TF)",
        "BatchSize": batch_size,
        "AvgTrainLoss": np.mean(fold_train_losses),
        "AvgValLoss": np.mean(fold_val_losses),
        "StdValLoss": np.std(fold_val_losses),
        "Metric": metrics[0],
        "TrainMetric": np.mean(fold_train_metrics),
        "ValMetric": np.mean(fold_val_metrics),
        "K": k,
        "loss_histories": fold_histories
    }


def plot_tf_convergence(loss_histories, optimizer_name, dataset_name, arch_name, save_dir="results/plots"):
    """
    Plot training/validation loss curves across folds for TensorFlow optimizers.

    Parameters
    ----------
    loss_histories : list of tuples
        Each element: (train_loss_list, val_loss_list)
    optimizer_name : str
        Optimizer name (e.g., "Adam (TF)")
    dataset_name : str
        Name of the dataset
    arch_name : str
        Architecture type ("Shallow" or "Deep")
    save_dir : str
        Directory for saving plots
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(9, 5))

    for fold_idx, (train_loss, val_loss) in enumerate(loss_histories, start=1):
        plt.plot(train_loss, label=f"Fold {fold_idx} (Training)")
        
    plt.title(f"Loss Convergence — {optimizer_name} on {dataset_name} ({arch_name}) [TensorFlow]")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"{dataset_name}_{arch_name}_{optimizer_name.replace(' ', '_')}_TF_loss.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"TensorFlow convergence plot saved to: {save_path}")


def main():
    # ---------------------
    # 1. Load All Datasets
    #----------------------
    datasets = load_all_datasets()

    # -----------------------------
    # 2. Define Neural Network Arch
    # -----------------------------
    shallow_arch = [16]       # 1 hidden layer
    deep_arch = [64, 32, 16]  # 3 hidden layers

    # ------------------------------
    # 3. Define Optimizers to test
    # ------------------------------
    optimizer_variants = [
        ("GradientDescent_Full", "gradientdescent", None, True, True),
        ("GradientDescent_MiniBatch", "gradientdescent", 20, True, True),
        ("SGD", "gradientdescent", 1, True, True),
        ("Adam", "adam", None, True, True),
        ("RMSProp", "rmsprop", None, True, True),
        ("BFGS", "bfgs", None, True, False),
        ("Adagrad", "adagrad", None, False, True)
    ] # different optimizers with their respective batch size setting (label, opt_name, batch_size, custom_exists, tf_exists)

    use_tensorflow = [False, True] # Testing both implementations

    # ----------------------
    # 4. Traning Parameters
    # ----------------------
    K = 5
    epochs = 100
    lr = 0.01
    tol = 1e-5
    max_iter = 100
    line_search = LineSearch(c1=1e-4, c2=0.9, tau=0.5, use_wolfe=True)

    # -----------------------
    # 5. Running experiments
    # -----------------------
    experiment_results = []
    tf_results = []

    for (X, y, name) in datasets:
        print(f"\n======================")
        print(f"Dataset: {name}")
        print(f"Input Size: {X.shape[0]} | Ouput Size: {y.shape[0]}")
        print(f"\n======================")

        for (arch_name, arch) in [("Shallow", shallow_arch), ("Deep", deep_arch)]:
            print(f"\nArchitecture: {arch_name} ({arch})")

            for label, opt_name, bs, custom_exists, tf_exists in optimizer_variants:
                
                if label == "BFGS" and len(arch) > 1:
                    print("\nSkipping BFGS for deep network since it's computationally infeasible.")
                    continue

                for tf_flag in use_tensorflow:
                    if not tf_flag:
                        if custom_exists:
                            try: 
                                nn = NeuralNetwork(X.shape[0], arch, y.shape[0], activation="tanh")

                                # Choose loss function (MSE for regression, cross-entropy for classification)
                                if y.shape[0] > 1:
                                    nn.set_loss("cross_entropy")
                                else:
                                    nn.set_loss("mse")

                                # Attach optimizer to neural network 
                                optimizer = OptimizerFactory.create(
                                        optimizer_name=opt_name,
                                        lr=lr,
                                        line_search=line_search,
                                        tol=tol,
                                        max_iter=max_iter
                                    )
                            
                                nn.set_optimizer(optimizer)

                                # Train using K-Fold cross validation
                                results = nn.cross_validate(X, y, k=K, epochs=epochs, lr=lr, batch_size=bs, verbose=False)

                                # Extract training and testing metrics
                                avg_train = results["avg_train_loss"]
                                avg_val = results["avg_val_loss"]
                                std_val = results["std_val_loss"]
                                metric_name = results["metric_name"]
                                avg_train_metric = results["avg_train_metric"]
                                avg_val_metric = results["avg_val_metric"]


                                experiment_results.append({
                                                        "Dataset": name,
                                                        "Architecture": arch_name,
                                                        "Optimizer": f"{label} (Custom)",
                                                        "BatchSize": bs if bs else "Full",
                                                        "AvgTrainLoss": avg_train,
                                                        "AvgValLoss": avg_val,
                                                        "StdValLoss": std_val,
                                                        "Metric": metric_name,           
                                                        "TrainMetric": avg_train_metric,      
                                                        "ValMetric": avg_val_metric,         
                                                        "K": K
                                                    })

                                # Plot and save convergence
                                """optimizer_label = f"{label} (Custom)"
                                optimizer_label_clean = re.sub(r'[^A-Za-z0-9_-]', '', optimizer_label.replace(' ', '_'))
                                plots_dir = "results/plots"
                                os.makedirs(plots_dir, exist_ok=True)

                                plot_title = f"Loss Convergence — {optimizer_label} on {name} ({arch_name})"
                                
                                save_path = os.path.join(
                                    plots_dir,
                                    f"{name}_{arch_name}_{optimizer_label_clean}_loss.png"
                                )"""

                                # Save convergence plot (all folds together)
                                #nn.plot_loss(title=plot_title, show=False, save_path=save_path)


                            except Exception as e: 
                                print(f"\n[WARNING] Skipped {opt_name.upper()} for {arch_name}: {e}\n")
                    else:
                        if tf_exists:
                            try:
                                result = run_tensorflow_experiment_cv(
                                    X, y, name, arch,
                                    optimizer_name=opt_name,
                                    epochs=epochs,
                                    batch_size=bs,
                                    lr=lr,
                                    k=K
                                )

                                # drop loss_histories for key alignment pruposes with the results dict from the custom experiments 
                                # if more key are added to dict then this needs to change
                                tf_results.append(dict(list(result.items())[:-1]))
                                print(f"{arch_name} | {opt_name.upper()} — Train Loss: {result['AvgTrainLoss']:.5f}, "
                                      f"Val Loss: {result['AvgValLoss']:.5f}")

                                """if "loss_histories" in result:
                                    plot_tf_convergence(
                                        result["loss_histories"],
                                        optimizer_name=f"{label}",
                                        dataset_name=name,
                                        arch_name=arch_name
                                    )"""
                            except Exception as e:
                                print(f"[WARNING] Skipped {opt_name.upper()} for {arch_name}: {e}")

    results_path = "results/optimizer_comparison_summary.csv"            
    df_results = pd.DataFrame(experiment_results) 
    tf_results = pd.DataFrame(tf_results)
    all_results = pd.concat([df_results, tf_results], axis=0, ignore_index=True)

    
    print("\nUpdated Results (Custom + TF):")
    print(all_results.round(6).to_string(index=False))

    all_results.to_csv(results_path, index=False)
    print(f"\nAll results saved to:\n{results_path}")


if __name__=="__main__":
    main()
