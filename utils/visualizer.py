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

def analyze_and_plot_errors(target_function, function_name, 
                           latex_name, xs, ys, approx_axon, approx_relu):
    """
    Complete error analysis and visualization for a single function
    """
    
    print(f"Computing errors for {function_name}...")
    
    # Compute errors
    error_axon = np.abs(ys - approx_axon)
    error_relu = np.abs(ys - approx_relu)
    
    # Compute statistics
    mae_axon = np.mean(error_axon)
    mae_relu = np.mean(error_relu)
    max_error_axon = np.max(error_axon)
    max_error_relu = np.max(error_relu)
    rmse_axon = np.sqrt(np.mean(error_axon**2))
    rmse_relu = np.sqrt(np.mean(error_relu**2))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_vals = xs.flatten()
    ax.plot(x_vals, error_axon, 'b-', linewidth=2, 
            label=f'Growing Axons (MAE: {mae_axon:.4f})', alpha=0.8)
    ax.plot(x_vals, error_relu, 'r--', linewidth=2, 
            label=f'Traditional ReLU (MAE: {mae_relu:.4f})', alpha=0.8)
    
    ax.set_title(f'Approximation Error: $f(x) = {latex_name}$', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('|f(x) - f\'(x)|', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    plt.show()
    
    # Return complete results
    return {
        'function': function_name,
        'latex': latex_name,
        'mae_axon': mae_axon,
        'mae_relu': mae_relu,
        'max_error_axon': max_error_axon,
        'max_error_relu': max_error_relu,
        'rmse_axon': rmse_axon,
        'rmse_relu': rmse_relu,
        'xs': xs,
        'ys': ys,
        'error_axon': error_axon,
        'error_relu': error_relu,
        'figure': fig,
        'axis': ax
    }