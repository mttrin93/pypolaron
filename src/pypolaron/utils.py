import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, TypedDict, Union
import numpy as np
import socket
import getpass
import logging
import re
import time
import subprocess
from pymatgen.core import Structure
from pyfhiaims.geometry import AimsGeometry
from pathlib import Path
from dataclasses import dataclass, field



def setup_pypolaron_logger(
    name: str = 'pypolaron',
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    hostname = socket.gethostname()
    if format_str is None:
        log_format = f'%(asctime)s %(levelname).1s - ({hostname}) - %(message)s'
    else:
        log_format = format_str

    if not logging.getLogger(name).handlers:
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt="%Y/%m/%d %H:%M:%S"
        )

    return logging.getLogger(name)

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

    attractor_elements: Optional[Union[str, List[str]]] = None
    aims_command: str = None

@dataclass(frozen=True)  # frozen=True makes the policy immutable, which is ideal
class WorkflowPolicy:
    """
    Configuration parameters defining the job execution policy for the PolaronWorkflow.
    """
    # Job Scheduler Parameters
    ntasks: int = 72
    walltime: str = "02:00:00"
    scheduler: str = "slurm"

    # Rerun/Stability Policy
    max_retries: int = 3

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

def run_job_and_wait(
        script_path: Path,
        log: logging.Logger,
        scheduler: str = "slurm",
):
    """
    Executes a job submission script and waits synchronously for its completion.

    Args:
        script_path (Path): Path to the executable job script (e.g., run_aims.sh).
        scheduler (Optional[str]): The scheduler type ('slurm' or None).

    Raises:
        RuntimeError: If the job submission or execution fails.
    """
    if scheduler and scheduler.lower() == 'slurm':
        # --- 1. SLURM Submission and Polling ---

        # 1a. Submit the job using sbatch
        log.info(f"Submitting SLURM job: {script_path}")
        try:
            submission_result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                check=True  # Raise exception on non-zero exit code for submission
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SLURM submission failed for {script_path.parent.name}: {e.stderr}")

        # Expected sbatch output format: "Submitted batch job 1234567"
        match = re.search(r'Submitted batch job (\d+)', submission_result.stdout)
        if not match:
            raise RuntimeError(f"Could not extract Job ID from sbatch output: {submission_result.stdout}")

        job_id = match.group(1)
        log.info(f"SLURM Job ID: {job_id}. Polling status...")

        last_state = None

        # 1b. Poll status until job finishes (sacct is generally more reliable than squeue)
        while True:
            # Check the job state using sacct
            try:
                # Use --noheader and -P for cleaner parsing
                status_result = subprocess.run(
                    ['sacct', '-j', job_id, '--format=State', '-P', '--noheader'],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                # sacct sometimes fails temporarily; log and continue polling
                log.warning(f"sacct command failed: {e.stderr}. Retrying in 10s.")
                time.sleep(10)
                continue

            # The output should contain the state (e.g., RUNNING, PENDING, COMPLETED, FAILED)
            state = status_result.stdout.strip()
            if not state:
                time.sleep(5)
                continue

            first_line = next((line for line in state.splitlines() if line.strip()), "")
            state_token = first_line.split()[0].rstrip('+').upper() if first_line else ""

            if state_token != last_state:
                log.info(f"Job {job_id} is currently {state_token}")

            last_state = state_token

            if state_token in ['RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING']:
                time.sleep(30)
            elif state_token == 'COMPLETED':
                log.info(f"SLURM Job {job_id} finished successfully.")
                return state_token
                # break
            # elif state_token in ['FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']:
            elif state_token in ['FAILED', 'CANCELLED', 'NODE_FAIL']:
                raise RuntimeError(f"SLURM Job {job_id} failed with state: {state_token}")
            elif state_token == "TIMEOUT":
                log.info(f"SLURM Job {job_id} failed with state: {state_token}")
                return state_token
            else:
                log.info(f"Job {job_id} in unexpected state: {state_token}. Waiting 30s.")
                time.sleep(30)

    else:
        # --- 2. Local/Bash Execution ---

        log.info(f"Executing local job: {script_path}")
        try:
            # Run the script directly using bash
            # check=True will raise CalledProcessError if the script exits with non-zero status
            subprocess.run(
                ['bash', str(script_path)],
                check=True,
                cwd=script_path.parent,  # Set CWD to the job directory
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            log.info(f"Local job finished successfully.")
        except subprocess.CalledProcessError as e:
            # Capture the output of the failed job for debugging
            output_message = e.output.strip() if e.output else "No output captured."
            raise RuntimeError(f"Local job failed for {script_path.parent.name}. Output: {output_message}")

def read_final_geometry(
        dft_code: str,
        job_directory: Path,
        log: logging.Logger,
) -> Optional[Structure]:
    """
    Reads the final relaxed structure from a completed DFT job directory.
    """
    dft_code = dft_code.lower()

    if dft_code == "aims":
        # FHI-AIMS output for relaxed geometry is typically geometry.in.next_step
        final_geom_path = job_directory / "geometry.in.next_step"
        aims_output_file = job_directory / "aims.out"
        # match = re.search("Have a nice day.", aims_output_file)
        if final_geom_path.exists(): #and match:
            log.info(f"Reading AIMS geometry from {final_geom_path.name}")
            try:
                # Need to use the Aims-specific parser if the file isn't POSCAR-like
                return AimsGeometry.from_file(final_geom_path).structure
            except Exception as e:
                log.error(f"Error parsing AIMS geometry: {e}")
                return None
        else:
            return None
    elif dft_code == "vasp":
        # VASP output for relaxed geometry
        final_geom_path = job_directory / "CONTCAR"
        if final_geom_path.exists():
            log.info(f"Reading VASP CONTCAR from {final_geom_path.name}")
            try:
                # Use standard pymatgen parser
                return Structure.from_file(str(final_geom_path))
            except Exception as e:
                log.error(f"Error parsing VASP CONTCAR: {e}")
                return None
        else:
            return None
    else:
        log.error(f"Final geometry file not found for {dft_code.upper()} in {job_directory.name}.")
        return None

def is_job_completed(dft_code: str, job_directory: Path) -> bool:
    """
    Checks if a calculation run has already successfully completed based on the
    presence of key output files and a successful termination string.

    Args:
        dft_code (str): The DFT code used ('aims' or 'vasp').
        job_directory (Path): The directory where the calculation was run.

    Returns:
        bool: True if key output files are found AND the termination message is present.
    """
    dft_code = dft_code.lower()

    if dft_code == 'aims':
        required_files = ["geometry.in.next_step", "aims.out"]
        log_file = job_directory / "aims.out"
        success_string = "Have a nice day."

    elif dft_code == 'vasp':
        required_files = ["CONTCAR", "OUTCAR"]
        log_file = job_directory / "OUTCAR"
        success_string = "reached required accuracy"

    else:
        return False

    for filename in required_files:
        if not (job_directory / filename).exists():
            return False

    if log_file.exists():
        try:
            success = False
            with open(log_file, 'r') as f:
                for line in f:
                    if line.lstrip().startswith(success_string):
                        success = True
            return success
        except Exception as e:
            print(f"Warning: Could not read log file {log_file.name} for content check: {e}")
            return False

    return False
