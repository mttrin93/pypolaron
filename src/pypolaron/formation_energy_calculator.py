from typing import Dict, Tuple, List, Optional
import numpy as np

from pymatgen.analysis.defects.corrections import freysoldt
from pymatgen.io.vasp.outputs import Locpot, Poscar
from pymatgen.core import Structure
from ase.io.cube import read_cube
from pymatgen.io.ase import AseAtomsAdaptor
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from ase import Atoms

class FormationEnergyCalculator:
    def __init__(self, total_charge: int, fermi_energy: float, epsilon: Optional[float] = None, volume_ang3: Optional[float] = None):
        """
        q: integer charge (e.g. -1 for electron polaron)
        mu_e: electron chemical potential referenced to VBM (eV)
        epsilon: dielectric constant (optional, for Makov-Payne)
        volume_ang3: cell volume in Angstrom^3 (optional, for Makov-Payne)
        """
        self.total_charge = total_charge
        self.fermi_energy = fermi_energy
        self.epsilon = epsilon
        self.volume_ang3 = volume_ang3

    def raw(self, E_polaron: float, E_pristine: float) -> float:
        """
        Basic formation energy (no corrections).
        E_polaron, E_pristine in eV, q integer charge (e.g. -1 for extra electron).
        mu_e: electron chemical potential (in eV) referenced to VBM (so if evaluating at CBM set accordingly).
        """
        return E_polaron - E_pristine - self.total_charge * self.fermi_energy

    def makov_payne_correction(self, alpha: float = 2.837297) -> float:
        """
        Rough Makov-Payne monopole correction in eV:
          E_mp = q^2 * alpha / (2 * epsilon * L)
        where L = (V)^(1/3) in Angstroms, alpha is Madelung constant (2.837 for cubic), epsilon is static dielectric constant.
        Note: This is a very rough estimate; FNV is preferred.
        """
        if self.epsilon is None or self.epsilon <= 0:
            raise ValueError("epsilon (dielectric constant) must be provided and >0 for Makov-Payne")
        L = (self.volume_ang3)**(1.0/3.0)
        # convert to eV: units check - here we assume atomic units converted; use empirical prefactor:
        # Use formula in eV: (q**2 * alpha) / (2 * epsilon * L) * (e^2 / (4*pi*epsilon0)) in eV*Angstrom units.
        # Numeric constant e^2/(4*pi*epsilon0) = 14.3996454784255 eV·Å
        pref = 14.3996454784255
        E_mp = (self.total_charge ** 2) * alpha * pref / (2.0 * self.epsilon * L)
        return float(E_mp)

    def potential_alignment_energy(self, delta_phi: float) -> float:
        """Energy correction: -q * delta_phi"""
        return -self.total_charge * delta_phi

    # TODO: to improve, have a look at https://doped.readthedocs.io/en/latest/doped.corrections.html and
    #https://doped.readthedocs.io/en/latest/_modules/doped/corrections.html#get_freysoldt_correction
    # TODO: implement other corrections: Kumagai, Falletta
    def get_freysoldt_correction_from_aims_cubes(self,
                                                 pristine_cube_path: str,
                                                 charged_cube_path: str,
                                                 structure: Union[Atoms, Structure],
                                                 site_supercell_index: int,
                                                 # axis: Optional[int] = None,
                                                 # plot: bool = False,
                                                 # filename_plot: Optional[str] = None,
                                                 # verbose: bool = True,
                                                 **kwargs
                                                 ) -> Dict:
        """
        Wrapper to compute Freysoldt correction (FNV) using pymatgen, from Gaussian cube files.

        Args:
            pristine_cube_path (str): Path to pristine potential (.cube)
            charged_cube_path (str): Path to charged potential (.cube)
            structure (Structure or ASE Atoms): Atomic structure (supercell)
            total_charge (int): Defect charge (q)
            dielectric (float or tensor): Total dielectric constant or tensor
            defect_fractional_coordinates (np.ndarray): Fractional coordinates of defect site
            axis (int): Axis for optional plotting (0, 1, 2)
            plot (bool): Whether to produce FNV potential plots
            filename_plot (str): Optional filename to save the plot
            verbose (bool): Whether to print the correction summary

        Returns:
            Dict: {
                "E_corr_eV": total correction (float),
                "E_alignment_eV": potential alignment correction,
                "E_image_eV": image-charge correction,
                "metadata": full CorrectionResult metadata
            }
        """

        # --- Convert ASE structure if needed ---
        if not isinstance(structure, Structure):
            structure = AseAtomsAdaptor().get_structure(structure)

        # --- Convert cube files to pymatgen Locpot-like objects ---
        # AimsCube is in pymatgen >=2023. If unavailable, you can parse manually with your read_cube()
        file_pristine_cube = Path(pristine_cube_path)
        with open(file_pristine_cube, "r") as f:
            pristine_cube = read_cube(f)

        file_charged_cube = Path(charged_cube_path)
        with open(file_charged_cube, "r") as f:
            charged_cube = read_cube(f)

        # Build "Locpot-like" objects (pymatgen expects potential grids)
        locpot_bulk = Locpot(
            poscar=Poscar(structure),
            data={"total": pristine_cube['data']},
        )
        locpot_defect = Locpot(
            poscar=Poscar(structure),
            data={"total": charged_cube['data']},
        )

        # --- Call pymatgen FNV correction ---
        fnv = freysoldt.get_freysoldt_correction(
            q=self.total_charge,
            dielectric=self.epsilon,
            defect_locpot=locpot_defect,
            bulk_locpot=locpot_bulk,
            lattice=structure.lattice,
            defect_frac_coords=structure[site_supercell_index].frac_coords,
            **kwargs
        )

        # # --- Print summary ---
        # if verbose:
        #     print(f"[FNV] Total correction: {fnv.correction_energy:.3f} eV")
        #     if "E_align" in fnv.metadata:
        #         print(f"[FNV] Alignment: {fnv.metadata['E_align']:.3f} eV")
        #     if "E_image" in fnv.metadata:
        #         print(f"[FNV] Image charge: {fnv.metadata['E_image']:.3f} eV")
        #
        # # --- Optionally plot ---
        # if plot:
        #     import matplotlib.pyplot as plt
        #     axis_label_dict = {0: "a-axis", 1: "b-axis", 2: "c-axis"}
        #     direction = axis or 2
        #     plot_data = fnv.metadata["plot_data"][direction]
        #     plt.figure(figsize=(5, 3))
        #     plt.plot(plot_data["x"], plot_data["potential"], label="ΔV(z)")
        #     plt.xlabel(axis_label_dict[direction])
        #     plt.ylabel("ΔV (eV)")
        #     plt.legend()
        #     if filename_plot:
        #         plt.savefig(filename_plot, dpi=300, bbox_inches="tight")
        #     plt.show()
        #
        return {
            "E_corr_eV": float(fnv.correction_energy),
            "E_alignment_eV": float(fnv.metadata.get("E_align", np.nan)),
            "E_image_eV": float(fnv.metadata.get("E_image", np.nan)),
            "metadata": fnv.metadata,
        }

    def corrected(self, raw_energy: float, delta_phi: Optional[float] = None) -> float:
        corr = 0.0
        if delta_phi is not None:
            corr += self.potential_alignment_energy(delta_phi)
        if self.epsilon is not None and self.volume_ang3 is not None:
            corr += self.makov_payne_correction()
        return raw_energy + corr
