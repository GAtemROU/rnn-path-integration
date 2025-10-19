import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_data_distribution(data, columns, title=None, output_path=None):
    """
    Plots the distribution of specified columns in the dataset.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    columns (list): List of column names to plot.
    title (str): Title of the plot.
    output_path (str, optional): If provided, saves the plot to the specified path.
    """
    num_cols = len(columns)
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 4))
    if title:
        fig.suptitle(title)
        fig._suptitle.set_fontsize(25)
    for i, col in enumerate(columns):
        axes[i].hist(data[col], bins=30, alpha=0.7, color='blue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def smart_subplots(n_plots, max_cols=7, figsize_per_plot=(4, 4)):
    """
    Creates a grid of subplots based on the number of plots required.
    
    Parameters:
    n_plots (int): Total number of subplots needed.
    max_cols (int): Maximum number of columns in the subplot grid.
    figsize_per_plot (tuple): Size of each subplot (width, height).
    
    Returns:
    fig, axes: Matplotlib figure and axes array.
    """
    if n_plots <= 0:
        raise ValueError("n_plots must be >= 1")

    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax in axes[n_plots:]:
        ax.axis('off')

    return fig, axes
