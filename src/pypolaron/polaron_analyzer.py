import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union, Literal
from ase.io.cube import read_cube
import logging

from pypolaron.formation_energy_calculator import FormationEnergyCalculator
from pypolaron.utils import parse_aims_total_energy, parse_aims_atomic_properties, \
    calculate_property_difference, get_localization_metrics, parse_total_spin_moment

# TODO: in the future run_postprocess script add the condition that if both mulliken and
#  hirshfled are not found, run a new scf calculation

class PolaronAnalyzer:
    """
    High-level analyzer for FHI-aims polaron calculations.

    Stores default parameters such as dielectric constant, Fermi level (e_fermi),
    system volume, etc., and provides methods to parse outputs, analyze
    charge localization, and compute formation energies with corrections.
    """

    def __init__(
        self,
        epsilon: Optional[float] = None,
        fermi_energy: float = 0.0,
        volume_ang3: Optional[float] = None,
        exclude_radius: float = 2.0,
    ):
        self.epsilon = epsilon
        self.fermi_energy = fermi_energy
        self.volume_ang3 = volume_ang3
        self.exclude_radius = exclude_radius

    def read_aims_cube(
        self, file_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
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
        data = cube["data"].copy()  # shape (nx, ny, nz)
        origin = cube["origin"].copy()  # 3-array, specifying the cube_data origin
        axes = np.array(cube["spacing"])  # 3x3 array, representing voxel size

        # Extract atoms
        number_of_atoms = len(cube["atoms"])
        atom_list = np.zeros((number_of_atoms, 4), dtype=float)
        for i, atom in enumerate(cube["atoms"]):
            atom_list[i, 0] = atom.number  # atomic number
            atom_list[i, 1:4] = atom.position  # x, y, z in Angstrom

        return data, origin, axes, number_of_atoms, atom_list

    # TODO: CHECK THIS FUNCTION
    def potential_alignment(
        self,
        neutral_pot_cube: str,
        charged_pot_cube: str,
        atom_coords: np.ndarray,
        exclude_radius: float = 2.0,
    ) -> float:
        """
        Compute potential alignment delta by averaging electrostatic potential in regions far from atoms in both cubes
        and returning (phi_charged - phi_neutral). Potentials should be in eV in the cube.
        This is simplistic: we exclude spheres of radius 'exclude_radius' around all atoms, then average remaining grid points.
        """
        grid_n, origin_n, axes_n, nat_n, atoms_n = PolaronAnalyzer.read_aims_cube(
            neutral_pot_cube
        )
        grid_c, origin_c, axes_c, nat_c, atoms_c = PolaronAnalyzer.read_aims_cube(
            charged_pot_cube
        )
        # compute mask of 'far' grid points (not within exclude_radius of any atom)
        nx, ny, nz = grid_n.shape
        ix, iy, iz = np.arange(nx), np.arange(ny), np.arange(nz)
        I, J, K = np.meshgrid(ix, iy, iz, indexing="ij")
        coords = (
            origin_n.reshape((1, 1, 1, 3))
            + I[..., None] * axes_n[0]
            + J[..., None] * axes_n[1]
            + K[..., None] * axes_n[2]
        )
        mask_far = np.ones(coords.shape[:-1], dtype=bool)
        for a in atom_coords:
            d = np.linalg.norm(coords - np.array(a).reshape((1, 1, 1, 3)), axis=-1)
            mask_far &= d > exclude_radius
        if mask_far.sum() == 0:
            # fallback: average whole cell
            phi_n = np.mean(grid_n)
            phi_c = np.mean(grid_c)
        else:
            phi_n = np.mean(grid_n[mask_far])
            phi_c = np.mean(grid_c[mask_far])
        # alignment to apply to energy: -q * delta_phi (but sign conventions vary). We'll return delta_phi = phi_ch - phi_neu
        return float(phi_c - phi_n)

    def calculate_atomic_property_difference(
            self,
            pristine_data: Dict[str, Dict[int, float]],
            polaron_data: Dict[str, Dict[int, float]],
            site_indices: Union[int, List[int]],
            property_key: Literal["mulliken_charges", "mulliken_spins", "hirshfeld_charges", "hirshfeld_spins"],
    ) -> Union[float, Dict[int, float]]:
        """
        Computes the difference (charged - pristine) for a specified atomic property
        (charge or spin moment) at the given site index(es).

        Args:
            pristine_data: results dictionary for prisitine run.
            polaron_data: results dictionary for polaron run.
            site_indices: The 0-based index (or list of indices) of the atom(s) to analyze.
            property_key: The property to extract ('mulliken_charge', 'hirshfeld_spin', etc.).

        Returns:
            The difference value(s) (float or Dict[int, float]).
        """

        pristine_property_map = pristine_data.get(property_key, {})
        charged_property_map = polaron_data.get(property_key, {})

        if not pristine_property_map or not charged_property_map:
            raise ValueError(f"Could not find or parse '{property_key}' data in one or both output files.")

        delta = calculate_property_difference(
            data_polaron=charged_property_map,
            data_pristine=pristine_property_map,
            site_indices=site_indices,
        )

        return delta

    # TODO: add logs for polaron eigenvalues, distances to cbm, vbm -> to do in run_workflow below
    def analyze_polaron_localization(
            self,
            eigenvalues_data: Dict[str, List[Tuple[float, float]]],
            log: logging.Logger,
            number_of_polarons: int,
            polaron_type: str,
            charged_out_path: str,
            localization_distance_threshold: float = 0.1,
    ) -> Dict[str, Union[float, str]]:
        """
        Calculates band edge gaps and determines the polaron localization channel
        based on the largest split-off energy (localization gap).

        Args:
            eigenvalues_data: Dict with keys 'spin_up', 'spin_down' containing lists
                              of (Energy_eV, Occupation).
            log: The logger instance.
            localization_distance_threshold: Minimum gap size (in eV) required to label a state as localized.

        Returns:
            A dictionary of results including HOMO, LUMO, localization metrics, and the result summary.
        """
        total_spin_moment = parse_total_spin_moment(charged_out_path)

        if total_spin_moment is None:
            log.warning("Total spin moment could not be parsed. Cannot determine spin distribution.")
            number_of_states_up = int(number_of_polarons / 2)
            number_of_states_down = int(number_of_polarons / 2)
        else:
            # N_up = (N_polaron + N_spin) / 2
            # N_down = (N_polaron - N_spin) / 2

            number_of_states_up = int((number_of_polarons + total_spin_moment) / 2)
            number_of_states_down = int((number_of_polarons - total_spin_moment) / 2)

            if number_of_states_up < 0 or number_of_states_down < 0:
                log.warning("Inconsistent spin and polaron numbers detected. Cannot determine spin distribution.")
                number_of_states_up = number_of_polarons
                number_of_states_down = 0

        analysis_results = {}

        E_homo_up, E_lumo_up, E_split_vbm_up, E_split_cbm_up, gap_up = get_localization_metrics(
            spin_channel="Spin Up",
            data=eigenvalues_data.get("spin_up", []),
            log=log,
            number_of_states=number_of_states_up,
            polaron_type=polaron_type,
        )

        E_homo_down, E_lumo_down, E_split_vbm_down, E_split_cbm_down, gap_down = get_localization_metrics(
            spin_channel="Spin Down",
            data=eigenvalues_data.get("spin_down",[]),
            log=log,
            number_of_states=number_of_states_down,
            polaron_type=polaron_type,
        )

        analysis_results["E_HOMO_Up_eV"] = E_homo_up
        analysis_results["E_LUMO_Up_eV"] = E_lumo_up
        analysis_results["E_Split_VBM_Up_eV"] = E_split_vbm_up
        analysis_results["E_Split_CBM_Up_eV"] = E_split_cbm_up

        analysis_results["E_HOMO_Down_eV"] = E_homo_down
        analysis_results["E_LUMO_Down_eV"] = E_lumo_down
        analysis_results["E_Split_VBM_Down_eV"] = E_split_vbm_down
        analysis_results["E_Split_CBM_Down_eV"] = E_split_cbm_down

        if number_of_states_up !=0  and gap_up < localization_distance_threshold:
            analysis_results["Warning_Up_Channel"] = (
                f"Localization gap in the up channel ({gap_up:.3f} eV) is small. "
                "The small polaron(s) may not be well-localized or may be delocalized in the structure."
            )
        else:
            analysis_results["Warning_Up_Channel"] = "The small polaron(s) is(are) well localized"

        if number_of_states_down !=0  and gap_down < localization_distance_threshold:
            analysis_results["Warning_Down_Channel"] = (
                f"Localization gap in the down channel ({gap_down:.3f} eV) is small. "
                "The small polaron(s) may not be well-localized or may be delocalized in the structure."
            )
        else:
            analysis_results["Warning_Down_Channel"] = "The small polaron(s) is(are) well localized"

        return analysis_results

    def analyze_polaron_run(
        self,
        pristine_out: str,
        charged_out: str,
        atom_coords: np.ndarray,
        site_index_supercell: int,
        total_charge: int,
        log: logging.Logger,
        pristine_cube_path: Optional[str] = None,
        charged_cube_path: Optional[str] = None,
        spin_cube: Optional[str] = None,
        pot_cube_pristine: Optional[str] = None,
        pot_cube_charged: Optional[str] = None,
    ) -> Dict[str, float]:
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
        energy_pristine = parse_aims_total_energy(pristine_out)
        energy_charged = parse_aims_total_energy(charged_out)
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
            E_polaron=energy_charged,
            E_pristine=energy_pristine,
        )

        # --- Mulliken populations
        pristine_atomic_properties_dict = parse_aims_atomic_properties(
            aims_out_path=pristine_out,
            log=log
        )

        polaron_atomic_properties_dict = parse_aims_atomic_properties(
            aims_out_path=charged_out,
            log=log
        )

        if (hasattr(pristine_atomic_properties_dict, "mulliken_charges") and
                hasattr(polaron_atomic_properties_dict, "mulliken_charges")):

            delta_mulliken_charges = self.calculate_atomic_property_difference(
                pristine_data=pristine_atomic_properties_dict,
                polaron_data=polaron_atomic_properties_dict,
                site_indices=site_index_supercell,
                property_key="mulliken_charges",
            )

            results["delta_mulliken_charges"] = delta_mulliken_charges

            delta_mulliken_spins = self.calculate_atomic_property_difference(
                pristine_data=pristine_atomic_properties_dict,
                polaron_data=polaron_atomic_properties_dict,
                site_indices=site_index_supercell,
                property_key="mulliken_spins",
            )

            results["delta_mulliken_spins"] = delta_mulliken_spins

        if (hasattr(pristine_atomic_properties_dict, "hirshfeld_charges") and
                hasattr(polaron_atomic_properties_dict, "hirshfeld_charges")):
            delta_hirshfeld_charges = self.calculate_atomic_property_difference(
                pristine_data=pristine_atomic_properties_dict,
                polaron_data=polaron_atomic_properties_dict,
                site_indices=site_index_supercell,
                property_key="hirshfeld_charges",
            )

            results["delta_hirshfeld_charges"] = delta_hirshfeld_charges

            delta_hirshfeld_spins = self.calculate_atomic_property_difference(
                pristine_data=pristine_atomic_properties_dict,
                polaron_data=polaron_atomic_properties_dict,
                site_indices=site_index_supercell,
                property_key="hirshfeld_spins",
            )

            results["delta_hirshfeld_spins"] = delta_hirshfeld_spins

        # corrections
        corrections = {}
        total_corr = 0.0

        # potential alignment
        if pot_cube_pristine and pot_cube_charged:
            try:
                delta_phi = self.potential_alignment(
                    pot_cube_pristine, pot_cube_charged, atom_coords
                )
                corrections["potential_alignment_delta_phi_eV"] = delta_phi
                # energy correction for formation energy is - q * delta_phi (if mu_e referenced to same potential)
                corrections["potential_alignment_energy_eV"] = -total_charge * delta_phi
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
        results["formation_energy_corrected_eV"] = (
            results["raw_formation_energy_eV"] + total_corr
        )

        return results
