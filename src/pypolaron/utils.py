import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, TypedDict, Union
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

    attractor_elements: str = None
    aims_command: str = None

def generate_occupation_matrix_content(
        scell: Structure, polaron_indices: List[int]
) -> str:
    """
    Generates the content for FHI-aims' occupation_matrix_control.txt file.

    We assume 5x5 matrices (d-orbitals) for the U term, and that the polaron
    is seeded by putting 1.0 in the first spin-up diagonal element.
    """
    matrix_size = 5  # Typical for d-orbitals (5x5 matrix)
    content = []
    empty_matrix = '\n'.join([' '.join(['0.00000'] * matrix_size)] * matrix_size)

    for i, site in enumerate(scell.sites):

        is_polaron_site = i in polaron_indices
        atom_index_in_aims = i + 1

        for spin in [1, 2]:
            header = (
                f"occupation matrix (subspace #           {atom_index_in_aims} , spin           {spin} )"
            )
            content.append(header)

            matrix_lines = []

            if is_polaron_site and spin == 1:  # Spin 1 = Spin Up (Seeding the polaron)
                # Seed the polaron by placing 1.0 in the first diagonal element
                seeded_matrix = [['0.00000'] * matrix_size for _ in range(matrix_size)]
                seeded_matrix[0][0] = '1.00000'

                for row in seeded_matrix:
                    matrix_lines.append(' '.join(row))

            else:  # Spin 2 (Spin Down) or Non-polaron site
                matrix_lines.append(empty_matrix)

            content.extend(matrix_lines)

    return '\n'.join(content)

def create_attractor_structure(
        scell: Structure,
        target_site_index: Union[int, List[int]],
        attractor_elements: Union[str, List[str]]
) -> Structure:
    """
    Creates the intermediate structure for the electron attractor method.
    Substitutes the original atom (M^n+) at the target polaron site with a
    specific attractor element (e.g., V^5+ -> Cr^3+).
    """
    if isinstance(target_site_index, int):
        target_site_index = [target_site_index]

    if isinstance(attractor_elements, str):
        elements_list = [e.strip().capitalize() for e in attractor_elements.split(',') if e.strip()]
    else:
        elements_list = [e.strip().capitalize() for e in attractor_elements]

    if len(elements_list) == 1 and len(target_site_index) > 1:
        elements_list = elements_list * len(target_site_index)

    if len(elements_list) != len(target_site_index):
        raise ValueError("The number of attractor elements must match the number of target sites.")

    attractor_structure = scell.copy()
    for target_site, element_symbol in zip(target_site_index, elements_list):
        attractor_structure.replace(target_site, element_symbol)

    return attractor_structure


