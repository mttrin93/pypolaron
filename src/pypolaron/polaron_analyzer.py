import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from ase.io import read
from numpy import ndarray, dtype, generic
from ase.io.cube import read_cube
import subprocess
import json
import os

from pypolaron.formation_energy_calculator import FormationEnergyCalculator

class PolaronAnalyzer:
    """
    High-level analyzer for FHI-aims polaron calculations.

    Stores default parameters such as dielectric constant, Fermi level (e_fermi),
    system volume, etc., and provides methods to parse outputs, analyze
    charge localization, and compute formation energies with corrections.
    """
    def __init__(self,
                 epsilon: Optional[float] = None,
                 fermi_energy: float = 0.0,
                 volume_ang3: Optional[float] = None,
                 exclude_radius: float = 2.0):
        self.epsilon = epsilon
        self.fermi_energy = fermi_energy
        self.volume_ang3 = volume_ang3
        self.exclude_radius = exclude_radius

    def parse_aims_total_energy(self, aims_out_path: str) -> float:
        """
        Parse total energy (in eV) from a FHI-aims output file using ASE.
        Falls back to regex parsing if ASE reading fails.
        """
        p = Path(aims_out_path)
        # --- Attempt to read using ASE ---
        try:
            atoms = read(p, format="aims-output")
            if hasattr(atoms, "get_potential_energy"):
                return atoms.get_potential_energy()
        except Exception as ase_err:
            print(f"[ASE parser warning] Failed to read {aims_out_path} with ASE: {ase_err}")

        text = p.read_text()
        # try some common patterns; energies in aims often in eV with 'Total energy of the DFT part' or 'Total energy'
        pattern = r"\|\s*Electronic free energy\s*:\s*([-\d\.Ee+]+)\s*eV"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            # take the last matched line
            return float(matches[-1])
        else:
            raise RuntimeError(f"Could not find electronic free energy in {aims_out_path}")

    def parse_aims_mulliken_population(self, aims_out_path: str) -> Dict[int, float]:
        """
        Parse Mulliken population table from FHI-aims output.
        Returns dict mapping atom_index (0-based) -> population (float, number of electrons assigned to atom).
        """
        #TODO: write the same function but in the case of hirshfeld population
        p = Path(aims_out_path)
        text = p.read_text()
        lines = text.splitlines()

        # Find start of Mulliken block
        start_idx = None
        for i, line in enumerate(lines):
            if "Summary of the per-atom charge analysis" in line:
                start_idx = i
                break
        if start_idx is None:
            # fallback: look for table with headers like "atom" and "electrons"
            for i, line in enumerate(lines):
                if re.search(r"^\s*\|?\s*atom\s+electrons", line, flags=re.IGNORECASE):
                    start_idx = i
                    break
        if start_idx is None:
            return {}  # no Mulliken block found

        data = {}
        for line in lines[start_idx + 2:]:
            if line.strip() == "" or "----" in line:
                break
            # extract atom index and electrons
            cols = line.split()
            if len(cols) < 3:
                continue
            try:
                atom_idx = int(cols[1]) - 1  # convert 1-based to 0-based
                electrons = float(cols[2])
            except ValueError:
                continue
            data[atom_idx] = electrons

        return data

    def read_aims_cube(self, file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Read a total_density.cube file using ASE’s read_cube.

        Returns:
        - data: 3D numpy array of the grid values
        - origin: 3-array of origin (Angstrom)
        - axes: 3x3 array of axis vectors (Angstrom)
        - natoms: integer number of atoms
        - atom_list: numpy array of shape (natoms,4) with (Z, x, y, z) in Angstrom

        ASE automatically reads Gaussian-style cube files and returns an object
        with `data`, `origin`, `axes`, and `atoms` attributes.
        """
        # file_path = Path(file_path)
        file_path = Path(file_path)
        with open(file_path, "r") as f:
            cube = read_cube(f)

        # Extract grid data
        data = cube['data'].copy()  # shape (nx, ny, nz)
        origin = cube['origin'].copy()  # 3-array, specifying the cube_data origin
        axes = np.array(cube['spacing'])  # 3x3 array, representing voxel size

        # Extract atoms
        number_of_atoms = len(cube['atoms'])
        atom_list = np.zeros((number_of_atoms, 4), dtype=float)
        for i, atom in enumerate(cube['atoms']):
            atom_list[i, 0] = atom.number  # atomic number
            atom_list[i, 1:4] = atom.position  # x, y, z in Angstrom

        return data, origin, axes, number_of_atoms, atom_list

    def run_bader(self,
                  charge_density_path: str,
                  output_dir: Optional[str] = None,
                  reference: Optional[str] = None,
                  bader_executable_path: Optional[str] = None) -> Dict:
        """
        Run external 'bader' program (Henkelman) on a charge-density file.
        Accepts CHGCAR or .cube (Gaussian cube) files. If 'reference' provided, passes -ref <reference>.
        Returns parsed results as dict: { 'atom_charges': [q1, q2, ...], 'ACF': parsed_text, 'raw_stdout': ... }
        Requires 'bader' binary on PATH.
        """
        p = Path(charge_density_path)
        if not p.exists():
            raise FileNotFoundError(f"Charge density file not found: {charge_density_path}")

        outdir = Path(output_dir) if output_dir else p.parent
        outdir.mkdir(parents=True, exist_ok=True)

        # --- add bader path to PATH if provided ---
        if bader_executable_path:
            os.environ["PATH"] = str(Path(bader_executable_path).parent) + os.pathsep + os.environ["PATH"]

        cmd = [bader_executable_path or "bader", str(p)]
        if reference:
            cmd += ["-ref", str(reference)]
        # run in output dir (bader writes ACF.dat / other files here)
        try:
            proc = subprocess.run(cmd, cwd=str(outdir), capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"bader failed: {e.stderr}\nCommand: {' '.join(cmd)}")

        # parse typical output: bader prints "Atom   Charge   ..."
        # We will look for ACF.dat.
        acf_path = outdir / "ACF.dat"
        atomic_bader_charges = []
        if acf_path.exists():
            # ACf.dat typical format: index  x  y  z  charge  min_dist  atomic_vol
            with open(acf_path, "r") as fh:
                for line in fh:
                    # Skip empty lines or separators
                    if not line or line.startswith('#') or line.startswith('-'):
                        continue
                    # Skip footer lines (non-numeric first column)
                    parts = line.split()
                    if not parts[0].isdigit():
                        continue
                    try:
                        # the charge column is at index 4
                        charge_val = float(parts[4])
                        atomic_bader_charges.append(charge_val)
                    except (ValueError, IndexError):
                        # skip malformed lines
                        continue
        else:
            return {"warning": "ACF.dat not found; Bader analysis not possible"}

        return {"atomic_charges": atomic_bader_charges, "acf_path": str(acf_path)}

    def charge_difference_from_bader_analysis(
            self,
            pristine_cube_path: str,
            charged_cube_path: str,
            site_supercell_index: int,
            bader_executable_path: Optional[str] = None,
            tmp_dir: str = "./"
    ) -> float:
        """
        Compute charge difference (charged - neutral) for a given atom index using Bader analysis.
        Runs the external 'bader' program on the provided cube files.
        Returns delta_charge: positive means more electrons in the charged calculation near the atom.
        """
        #TODO: here we calculate the differences in bader charges, try to calculate the difference in magmoms (charged - neutral)
        # check if the code works when site_supercell_index is a list of ints

        # --- run bader for both charge densities ---
        bader_neutral = self.run_bader(
            charge_density_path=pristine_cube_path,
            output_dir=os.path.join(tmp_dir, "neutral"),
            bader_executable_path=bader_executable_path
        )
        bader_charged = self.run_bader(
            charge_density_path=charged_cube_path,
            output_dir=os.path.join(tmp_dir, "charged"),
            bader_executable_path=bader_executable_path
        )

        # --- extract atomic charges ---
        charges_neutral = bader_neutral.get("atomic_charges", [])
        charges_charged = bader_charged.get("atomic_charges", [])

        if not charges_neutral or not charges_charged:
            raise ValueError("Bader analysis failed to produce atomic charges for one or both cubes.")

        if len(charges_neutral) != len(charges_charged):
            raise ValueError(
                f"Number of atoms differ between neutral ({len(charges_neutral)}) and charged ({len(charges_charged)}) runs."
            )

        # --- compute charge difference for the selected atom ---
        delta_q = charges_charged[site_supercell_index] - charges_neutral[site_supercell_index]

        return delta_q

    # TODO: CHECK THIS FUNCTION
    def compute_spin_ipr_from_spin_cube(self, spin_cube_path: str) -> float:
        """
        Approximate IPR from a spin-density cube: IPR = sum(rho^2) / (sum(rho))^2
        This returns a single number for the entire cell; larger => more localized spin density.
        """
        grid, origin, axes, nat, atoms = self.read_aims_cube(spin_cube_path)
        vals = grid.flatten()
        s = np.sum(vals)
        if abs(s) < 1e-12:
            return 0.0
        ipr = np.sum(vals**2) / (s**2)
        return float(ipr)

    # TODO: CHECK THIS FUNCTION
    def potential_alignment(self, neutral_pot_cube: str, charged_pot_cube: str,
                            atom_coords: np.ndarray,
                            exclude_radius: float = 2.0) -> float:
        """
        Compute potential alignment delta by averaging electrostatic potential in regions far from atoms in both cubes
        and returning (phi_charged - phi_neutral). Potentials should be in eV in the cube.
        This is simplistic: we exclude spheres of radius 'exclude_radius' around all atoms, then average remaining grid points.
        """
        grid_n, origin_n, axes_n, nat_n, atoms_n = PolaronAnalyzer.read_aims_cube(neutral_pot_cube)
        grid_c, origin_c, axes_c, nat_c, atoms_c = PolaronAnalyzer.read_aims_cube(charged_pot_cube)
        # compute mask of 'far' grid points (not within exclude_radius of any atom)
        nx, ny, nz = grid_n.shape
        ix, iy, iz = np.arange(nx), np.arange(ny), np.arange(nz)
        I, J, K = np.meshgrid(ix, iy, iz, indexing="ij")
        coords = ( origin_n.reshape((1, 1, 1, 3)) + I[..., None]*axes_n[0] +
                   J[..., None]*axes_n[1] + K[..., None]*axes_n[2] )
        mask_far = np.ones(coords.shape[:-1], dtype=bool)
        for a in atom_coords:
            d = np.linalg.norm(coords - np.array(a).reshape((1, 1, 1, 3)), axis=-1)
            mask_far &= (d > exclude_radius)
        if mask_far.sum() == 0:
            # fallback: average whole cell
            phi_n = np.mean(grid_n)
            phi_c = np.mean(grid_c)
        else:
            phi_n = np.mean(grid_n[mask_far])
            phi_c = np.mean(grid_c[mask_far])
        # alignment to apply to energy: -q * delta_phi (but sign conventions vary). We'll return delta_phi = phi_ch - phi_neu
        return float(phi_c - phi_n)

    def analyze_polaron_run(self,
                            pristine_out: str, charged_out: str,
                            atom_coords: np.ndarray,
                            site_index_supercell: int,
                            total_charge: int,
                            pristine_cube_path: Optional[str] = None,
                            charged_cube_path: Optional[str] = None,
                            spin_cube: Optional[str] = None,
                            pot_cube_pristine: Optional[str] = None,
                            pot_cube_charged: Optional[str] = None,
                            bader_executable_path: Optional[str] = None) -> Dict[str, float]:
        """
        High-level analysis:
        - Parse total energies from aims outputs
        - Parse Mulliken populations (if present) and compute delta on chosen site
        - Optionally approximate Bader via sphere integration on provided charge-density cubes
        - Compute spin IPR from spin-density cube
        - Compute formation energy (raw), and add corrections:
            - potential alignment (if potential cubes provided)
            - Makov-Payne monopole correction (if epsilon given)
        Returns dict with metrics and energies.
        """
        # TODO: site_index_supercell can be also a list of ints in the multipolaronic case
        results = {}

        # --- Parse total energies ---
        energy_pristine = self.parse_aims_total_energy(pristine_out)
        energy_charged = self.parse_aims_total_energy(charged_out)
        results["E_pristine_eV"] = energy_pristine
        results["E_charged_eV"] = energy_charged

        # --- Initialize formation energy calculator ---
        formation_energy_calculator = FormationEnergyCalculator(
            total_charge=total_charge,
            fermi_energy=self.fermi_energy,
            epsilon=self.epsilon,
            volume_ang3=self.volume_ang3,
        )

        # --- Compute raw formation energy ---
        results["raw_formation_energy_eV"] = formation_energy_calculator.raw(
            E_polaron=energy_charged, E_pristine=energy_pristine,
        )

        # --- Mulliken populations
        mulliken_pristine = self.parse_aims_mulliken_population(pristine_out)
        mulliken_charged = self.parse_aims_mulliken_population(charged_out)
        if mulliken_pristine and mulliken_charged:
            population_pristine = mulliken_pristine.get(site_index_supercell)
            population_charged = mulliken_charged.get(site_index_supercell)

            results["mulliken_pristine"] = population_pristine
            results["mulliken_charged"] = population_charged

            # Compute delta population (charged - pristine)
            results["delta_mulliken_population"] = population_charged - population_pristine

        # approximate sphere integration (Bader-like)
        if pristine_cube_path and charged_cube_path:
            try:
                delta_q = self.charge_difference_from_bader_analysis(pristine_cube_path=pristine_cube_path,
                                                                     charged_cube_path=charged_cube_path,
                                                                     bader_executable_path=bader_executable_path,
                                                                     site_supercell_index=site_index_supercell)
                results["delta_charge_sphere_integral"] = delta_q
            except Exception as e:
                results["delta_charge_sphere_integral_error"] = str(e)

        # spin IPR
        if spin_cube:
            try:
                ipr = self.compute_spin_ipr_from_spin_cube(spin_cube)
                results["spin_ipr"] = ipr
            except Exception as e:
                results["spin_ipr_error"] = str(e)

        # corrections
        corrections = {}
        total_corr = 0.0

        # potential alignment
        if pot_cube_pristine and pot_cube_charged:
            try:
                delta_phi = self.potential_alignment(pot_cube_pristine, pot_cube_charged, atom_coords)
                corrections["potential_alignment_delta_phi_eV"] = delta_phi
                # energy correction for formation energy is - q * delta_phi (if mu_e referenced to same potential)
                corrections["potential_alignment_energy_eV"] = - total_charge * delta_phi
                total_corr += corrections["potential_alignment_energy_eV"]
            except Exception as e:
                corrections["potential_alignment_error"] = str(e)

        # Makov-Payne
        if self.epsilon is not None and self.volume_ang3 is not None:
            try:
                mp = formation_energy_calculator.makov_payne_correction()
                corrections["makov_payne_eV"] = mp
                total_corr += mp
            except Exception as e:
                corrections["makov_payne_error"] = str(e)

        results["corrections"] = corrections
        results["total_corrections_eV"] = total_corr
        results["formation_energy_corrected_eV"] = results["raw_formation_energy_eV"] + total_corr

        return results
