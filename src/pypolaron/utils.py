import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, TypedDict
import numpy as np
from pymatgen.core import Structure
from pathlib import Path
from dataclasses import dataclass, field


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

@dataclass(frozen=True)  # frozen=True makes the settings immutable, which is good practice for configurations
class DftSettings:
    """
    Configuration parameters for the DFT polaron calculation.
    """
    # Core DFT Parameters
    functional: str = "hse06"
    dft_code: str = "aims"
    calc_type: str = "relax-atoms"

    # Execution/Environment Parameters
    run_dir_root: str = "polaron_runs"
    species_dir: Optional[str] = None
    do_submit: bool = False
    run_pristine: bool = False

    # Structural/Model Parameters
    supercell: Tuple[int, int, int] = (2, 2, 2)
    add_charge: int = -1  # TODO: obsolete argument

    # Spin/U/Hybrid Parameters
    spin_moment: float = 1.0
    set_site_magmoms: bool = False
    alpha: float = 0.25
    hubbard_parameters: Optional[str] = None
    fix_spin_moment: Optional[float] = None
    disable_elsi_restart: bool = False

    # Analysis Parameters
    do_bader: bool = True
    potential_axis: int = 2
    dielectric_eps: float = 10.0
    auto_analyze: bool = True

class DftParameters(TypedDict):
    dft_code: str
    functional: str
    calc_type: str
    supercell: Tuple[int, int, int]
    aims_command: str
    species_dir: Optional[str]
    run_dir_root: str
    do_submit: bool
    set_site_magmoms: bool
    spin_moment: float
    run_pristine: bool
    alpha: float
    hubbard_parameters: Optional[str]
    fix_spin_moment: Optional[float]
    disable_elsi_restart: bool
