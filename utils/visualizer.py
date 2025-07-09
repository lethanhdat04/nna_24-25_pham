import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython import get_ipython
from scipy.stats import pearsonr

def plot_results_1d(X_train, y_train, X_test, y_test, y_pred, 
                    history, title,
                    highlight_points=[],
                    plot_size=(12, 5),
                    save_path: str = None):
    """
    Plot training history and function approximation results for 1D data.
    
    :param X_train: training data input
    :param y_train: training data output
    :param X_test: test data input
    :param y_test: test data output
    :param y_pred: predicted output
    :param history: training history
    :param title: plot title
    :param plot_size: size of the plot, default=(12, 5)
    :param save_path: path to save the plot, default=None
    
    :return: None
    """

    plt.figure(figsize=plot_size)
    
    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # Plot function approximation
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, label='Train Data', color='blue', alpha=0.5)
    plt.scatter(X_test, y_test, label='Test Data', color='green', alpha=0.5)
    plt.scatter(X_test, y_pred, label='Predictions', color='red', alpha=0.5)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend()

    # Highlight points
    for point in highlight_points:
        plt.axvline(x=point, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        if get_ipython():
            plt.show()
        else:
            print('Plot displayed in non-interactive mode. Saving to file instead. Please provide a save_path.')

    plt.close()

def plot_approximations(xs, ys, approx_axon, approx_relu, func_label="f(x)"):
    plt.figure(figsize=(12, 5))

    # Axon plot
    plt.subplot(1, 2, 1)
    plt.plot(xs, ys, label=f'Target: ${func_label}$')
    plt.plot(xs, approx_axon, '--', label='Axon Approximation')
    plt.title("Axon Approximation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    # ReLU network plot
    plt.subplot(1, 2, 2)
    plt.plot(xs, ys, label=f'Target: ${func_label}$')
    plt.plot(xs, approx_relu, '--', label='ReLU Network')
    plt.title("Traditional ReLU Network")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()