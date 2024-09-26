import math
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from matplotlib.patches import Ellipse
from uncertainty_loss import uncertainty_function_at_t


def plot_losses(train_loss, test_loss, dataset_name, seed, plot_title="Loss Plot"):
    """
    Plots the training and testing losses over epochs and saves the plot in the specified directory.

    Args:
        train_loss (list or array): A list or array containing the training loss values per epoch.
        test_loss (list or array): A list or array containing the testing loss values per epoch.
        plot_title (str): The title for the plot (default is 'Loss Plot').
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b', label='Training Loss', linestyle='-', marker='o')
    plt.plot(epochs, test_loss, 'r', label='Testing Loss', linestyle='-', marker='x')

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'{plot_title} - Training and Testing Loss over Epochs', fontsize=16)

    plt.grid(True)
    plt.legend(fontsize=12)

    save_path = f"plots/{dataset_name}/{seed}/{plot_title}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def plot_points_with_uncertainty_shapes(
        true_points, pred_points, coefficients, q, dataset_name, seed, iteration, title):
    num_steps = 100
    t_values = torch.linspace(0, 2 * math.pi, num_steps)

    plt.figure(figsize=(8, 8))

    colors = plt.cm.get_cmap('tab10', len(true_points))

    for i, (true_point, pred_point) in enumerate(zip(true_points, pred_points)):
        x_true, y_true = true_point
        x_pred, y_pred = pred_point
        coeffs = torch.tensor(coefficients[i], dtype=torch.float32)

        color = colors(i)
        plt.plot(x_true, y_true, 'o', color=color, markersize=8)
        plt.plot(x_pred, y_pred, 's', color=color, markersize=6)

        uncertainty_x = []
        uncertainty_y = []

        for t in t_values:
            uncertainty_radius = torch.tensor(q) * uncertainty_function_at_t(coeffs, t)
            uncertainty_radius = uncertainty_radius.item()
            new_x = x_pred + uncertainty_radius * math.cos(t.item())
            new_y = y_pred + uncertainty_radius * math.sin(t.item())
            uncertainty_x.append(new_x)
            uncertainty_y.append(new_y)

        plt.plot(uncertainty_x, uncertainty_y, '--', color=color, alpha=0.6)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)

    if iteration == -1:
        save_path = f"plots/{dataset_name}/{seed}/MVSVRC_uncertainty_shape_test.png"
    else:
        save_path = f"plots/{dataset_name}/{seed}/MVSVRC_uncertainty_shape_train_epoch_{iteration}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def draw_ellipsoids(covs, scale, pred_list, true_list, dataset_name, seed, title="Uncertainty Plot"):
    num_samples = len(pred_list)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    colors = plt.cm.get_cmap('tab10', num_samples)

    for i in range(num_samples):
        pred = pred_list[i]
        true = true_list[i]
        cov = covs[i]

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        width, height = 2 * np.sqrt(eigenvalues * scale)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        ell = Ellipse(xy=pred, width=width, height=height, angle=angle, edgecolor=colors(i),
                      facecolor='none', lw=2, linestyle='--')
        ax.add_patch(ell)

        ax.scatter(*pred, color=colors(i), marker='s', s=80)
        ax.scatter(*true, color=colors(i), marker='o', s=80)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.title(title)

    save_path = f"plots/{dataset_name}/{seed}/NLE_uncertainty_ellipsoid_test.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
