import matplotlib.pyplot as plt
from typing import List
import numpy as np


def plot_site_occupations(atomic_positions: np.ndarray,
                          atomic_symbols: List[str],
                          site_occupations: np.ndarray,
                          title: str = "Polaron Site Occupations"):
    """
    Plot atomic positions colored by polaron site occupations.

    Args:
        atomic_positions: Nx3 array of atomic positions
        atomic_symbols: list of atomic symbols
        site_occupations: array of length N with occupation probabilities
        title: plot title
    """
    # 2D projection (x vs y)
    x = atomic_positions[:, 0]
    y = atomic_positions[:, 1]
    colors = site_occupations
    sizes = 300  # marker size

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x, y, s=sizes, c=colors, cmap='Reds', edgecolors='k')

    # Annotate symbols
    for i, sym in enumerate(atomic_symbols):
        plt.text(x[i], y[i], sym, ha='center', va='center', color='white', weight='bold')

    plt.colorbar(scatter, label='Polaron Occupation Probability')
    plt.xlabel('x (Å)')
    plt.ylabel('y (Å)')
    plt.title(title)
    plt.axis('equal')
    plt.show()
