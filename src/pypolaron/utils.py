import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np
from pymatgen.core import Structure
from pathlib import Path


def plot_site_occupations(
    atomic_positions: np.ndarray,
    atomic_symbols: List[str],
    site_occupations: np.ndarray,
    title: str = "Polaron Site Occupations",
):
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
    scatter = plt.scatter(x, y, s=sizes, c=colors, cmap="Reds", edgecolors="k")

    # Annotate symbols
    for i, sym in enumerate(atomic_symbols):
        plt.text(
            x[i], y[i], sym, ha="center", va="center", color="white", weight="bold"
        )

    plt.colorbar(scatter, label="Polaron Occupation Probability")
    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title(title)
    plt.axis("equal")
    plt.show()

def parse_aims_plus_u_params(dftu_str: str) -> Dict[str, List[Tuple[int, str, float]]]:
    """
    Parses a DFT+U string (e.g., 'Ti:3d:4.5') into the FHI-aims 'plus_u' structure.

    FHI-aims format required: {"Element": [(n_quantum, 'l_char', U_value)]}
    The orbital character must be extracted from the orbital string (e.g., '3d' -> 'd').
    """
    if not dftu_str:
        return {}

    parsed_u = {}

    # Example format expected: "Mg:3d:2.65,Ti:3d:4.5" (J term is ignored/set to 0 for AIMS U)
    for term in dftu_str.split(','):
        parts = term.strip().split(':')

        if len(parts) < 3:
            print(f"Skipping malformed DFT+U term (requires Element:Orbital:U): {term}")
            continue

        element_sym = parts[0].strip()
        orbital_str = parts[1].strip()
        u_value = float(parts[2].strip())

        # 1. Extract n (principal quantum number) and l (angular momentum character)
        try:
            # Orbital is typically in the form '3d', '2p', etc.
            n_quantum = int(orbital_str[0])
            l_char = orbital_str[1].lower()

            if l_char not in ['s', 'p', 'd', 'f']:
                print(f"Invalid orbital character '{l_char}' in term {term}. Skipping.")
                continue

        except (IndexError, ValueError):
            print(f"Could not parse orbital string '{orbital_str}'. Skipping term {term}.")
            continue

        # 2. Assemble the required tuple structure
        u_tuple = (n_quantum, l_char, u_value)

        # 3. Assemble the required dictionary structure: List of tuples
        # Note: If the element is already present, we append the tuple (supporting multiple U per element)
        if element_sym not in parsed_u:
            parsed_u[element_sym] = []

        parsed_u[element_sym].append(u_tuple)

    return parsed_u

