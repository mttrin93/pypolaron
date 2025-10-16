from typing import Dict, Tuple, List, Optional, Union

from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element

# from pyfhiaims import AimsControlIn, AimsGeometryIn
from pyfhiaims.geometry import AimsGeometry
from pyfhiaims.control import AimsControl
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase import Atoms
import numpy as np
import shutil
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="Won't overwrite species")


class PolaronGenerator:
    """
    Class for proposing small polaron localization sites in a crystal structure.
    """

    def __init__(self, structure: Structure, polaron_type: str = "electron"):
        """
        structure : pymatgen Structure
        polaron_type : "electron" (extra electron) or "hole" (missing electron)
        """
        self.structure = structure.copy()
        self.polaron_type = polaron_type
        self.oxidation_assigned = False
        self.crystal_near_neighbors = CrystalNN()

    def assign_oxidation_states(self):
        """Assign oxidation states using Bond Valence Analyzer."""
        bond_valence_analyzer = BVAnalyzer()
        try:
            self.structure = bond_valence_analyzer.get_oxi_state_decorated_structure(
                self.structure
            )
            self.oxidation_assigned = True
        except Exception as e:
            print(
                f"Warning: BVAnalyzer failed to assign oxidation states. Please ensure structure is valid. Error {e}"
            )
            self.oxidation_assigned = False

    def _get_symmetrically_distinct_sites(self) -> List[int]:
        """
        Filters the structure sites to return a list of indices
        for only the symmetrically distinct sites.
        """
        if not self.structure.is_ordered:
            return list(range(len(self.structure)))

        sga = SpacegroupAnalyzer(self.structure)

        try:
            equivalent_indices = sga.get_symmetry_dataset().equivalent_atoms
        except AttributeError:
            equivalent_indices = sga.get_symmetry_dataset()["equivalent_atoms"]

        seen_representatives_indices = set()
        distinct_site_indices = []

        for i, rep_index in enumerate(equivalent_indices):
            if rep_index not in seen_representatives_indices:
                distinct_site_indices.append(i)
                seen_representatives_indices.add(rep_index)

        return distinct_site_indices

    def propose_sites(
        self, max_sites: int = 5
    ) -> List[Tuple[int, str, Optional[float], float, float]]:
        """
        Propose candidate sites for polaron localization.
        Returns list of tuples: (site_index, element, oxidation_state, coordination_number, score)
        """
        if not self.oxidation_assigned:
            self.assign_oxidation_states()

        if not self.oxidation_assigned:
            return []

        distinct_indices = self._get_symmetrically_distinct_sites()
        candidates = []

        for index in distinct_indices:
            site = self.structure.sites[index]
            element = site.specie.symbol
            oxidation_state = (
                site.specie.oxi_state if hasattr(site.specie, "oxi_state") else 0.0
            )
            coordination_number = self.crystal_near_neighbors.get_cn(
                self.structure, index
            )

            score = self._score_site(element, oxidation_state, coordination_number)
            if score > 0:  # only keep plausible candidates
                candidates.append(
                    (index, element, oxidation_state, coordination_number, score)
                )

        # sort by score (higher = more likely polaron site)
        candidates.sort(key=lambda x: -x[4])
        return candidates[:max_sites]

    def _score_site(
        self, element: str, oxidation_state: float, coordination_number: float
    ) -> float:
        """
        Simple heuristic scoring that incorporates both structural and electronic information:
        - oxidation states (high/low for electron/hole polarons, i.e. Ti4+, Fe3+, Mn4+ .../O2-)
        - low coordination numbers
        - Pauli electronegativity differences
        - Madelung potential (high positive/low negative for electron/hole polarons)
        - ionization potentials/electron affinities
        """
        elem = Element(element)
        base_charge_score = abs(oxidation_state) * 2.0

        # Structural factor: Penalty for ideal coordination, bonus for 'strain' (undercoordination)
        # Use simple ideal CNs for common coordination geometries (e.g., 6 for octahedral, 4 for tetrahedral)
        # This is a bit rough, but better than a fixed constant.
        ideal_cn = 6.0
        structural_factor = max(0.0, ideal_cn - coordination_number) * 0.5
        score = base_charge_score + structural_factor

        if self.polaron_type == "electron":
            if oxidation_state < 2:
                return 0.0
            electronic_factor = (
                4.0 - elem.X
            ) * 0.5  # lower electronegativity = more reducible
            score += electronic_factor
        elif self.polaron_type == "hole":
            if oxidation_state > -1:
                return 0.0
            electronic_factor = (
                elem.X * 0.5
            )  # higher electronegativity = better hole host
            score += electronic_factor
        return score

    def _find_central_site_general_supercell(
        self, scell: Structure, site_index: Union[int, List[int]]
    ) -> List[int]:
        """
        Find the index of the atom in the supercell corresponding to
        a primitive cell site, located closest to the geometric center.
        Works for both cubic and non-cubic supercells.
        """
        if isinstance(site_index, int):
            site_index = [site_index]

        scell_center = np.mean([site.coords for site in scell.sites], axis=0)
        used_indices = set()
        mapped_indices = []
        tolerance = 1e-4

        for index in site_index:
            target_specie = self.structure[index].specie
            candidates = [
                (i, site)
                for i, site in enumerate(scell.sites)
                if site.specie == target_specie and i not in used_indices
            ]

            distances = np.linalg.norm(
                [site.coords - scell_center for _, site in candidates], axis=1
            )
            index_min_distance_center = candidates[np.argmin(distances)][0]
            used_indices.add(index_min_distance_center)
            mapped_indices.append(index_min_distance_center)

        return mapped_indices

    def _get_total_charge(
        self,
        scell: Structure,
        polaron_site_indices: Union[int, List[int]]
    ) -> int:
        """
        Calculates the total charge for the supercell.
        """
        if isinstance(polaron_site_indices, int):
            polaron_site_indices = [polaron_site_indices]

        num_polarons = len(polaron_site_indices)

        # Calculate Charge
        total_charge = getattr(scell, "charge", 0)
        if self.polaron_type == "electron":
            total_charge = -num_polarons
        elif self.polaron_type == "hole":
            total_charge = num_polarons

        return int(total_charge)

    def _create_magmoms(
        self, scell: Structure, site_index: Union[int, List[int]], spin_moment: float
    ) -> Tuple[List[int], np.ndarray]:
        """
        Initialize the initial magmoms
        """
        site_index_supercell = self._find_central_site_general_supercell(
            scell, site_index
        )

        magmoms = np.array([0.0] * len(scell))
        magmoms[site_index_supercell] = spin_moment
        return site_index_supercell, magmoms

    def prepare_for_ase(
        self,
        site_index: Union[int, List[int]],
        supercell: Tuple[int, int, int] = (2, 2, 2),
        spin_moment: float = 1.0,
        set_site_magmoms: bool = True,
    ) -> Atoms:
        """
        Build ASE Atoms supercell and attach initial magnetic moments and metadata.
        Returns ASE Atoms instance ready to pass to ASE calculators or to be written out.
        """
        # build pymatgen supercell
        scell = self.structure * supercell

        # convert to ASE
        ase_atoms = AseAtomsAdaptor.get_atoms(scell)

        # assign initial magmom
        if set_site_magmoms:
            site_index_supercell, magmoms = self._create_magmoms(
                scell, site_index, spin_moment
            )
            ase_atoms.set_initial_magnetic_moments(magmoms)
        else:
            site_index_supercell = self._find_central_site_general_supercell(scell, site_index)

        total_charge = self._get_total_charge(scell, site_index_supercell)

        ase_atoms.info["polaron_site_index_in_supercell"] = site_index_supercell
        # attach total charge as metadata
        ase_atoms.info["polaron_total_charge"] = total_charge  # user-defined tag

        return ase_atoms

    def write_vasp_input_files(
        self,
        site_index: Union[int, List[int]],
        supercell: Tuple[int, int, int] = (2, 2, 2),
        spin_moment: float = 1.0,
        set_site_magmoms: bool = True,
        outdir: str = "polaron_calc",
    ):
        """
        Generate a supercell with a seeded polaron (localized spin on chosen site).

        site_index : int
            Index of candidate site in primitive cell
        supercell : tuple
            Replication factors (a,b,c)
        spin_moment : float
            Initial magnetic moment to assign to the polaron site
        vasp : bool
            If True, prepare VASP input files
        outdir : str
            Output directory
        """
        # make output dir
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)

        # build supercell
        scell = self.structure * supercell

        # assign initial magmom
        if set_site_magmoms:
            site_index_supercell, magmoms = self._create_magmoms(
                scell, site_index, spin_moment
            )
            scell.add_site_property("magmom", magmoms)
        else:
            site_index_supercell = self._find_central_site_general_supercell(scell, site_index)

        total_charge = self._get_total_charge(scell, site_index_supercell)
        nelect = scell.get_nelectrons() - total_charge

        user_incar = {
            # Total number of electrons: adjust for polaron
            "NELECT": nelect,
            # Spin polarization
            "ISPIN": 2,
            # Turn off DFT+U
            "LDAU": False,
            # Hybrid functional settings (HSE06)
            "LHFCALC": True,  # Enable hybrid functional
            "HFSCREEN": 0.2,  # Screening parameter for HSE06
            "AEXX": 0.25,  # Fraction of exact exchange
            "ALGO": "All",  # Recommended for hybrids
            "PREC": "Accurate",  # High precision
            "NELM": 200,  # Maximum electronic steps
            "EDIFF": 1e-5,  # Convergence for electronic loop
        }

        vis = MPRelaxSet(scell, user_incar_settings=user_incar)
        vis.write_input(outdir, potcar_spec=True)

    def write_fhi_aims_input_files(
        self,
        site_index: Union[int, List[int]],
        supercell: Tuple[int, int, int] = (2, 2, 2),
        spin_moment: float = 1.0,
        xc: str = "hse06",
        tier: str = "tight",
        species_dir: str = "./",
        set_site_magmoms: bool = True,
        outdir: str = "./fhi_aims_files",
    ):
        """
        Write a simple FHI-aims 'geometry.in' and 'control.in' that:
        - builds the supercell geometry
        - sets total charge (if add_charge != 0) into the control.in
        - sets spin collinear and a simple initial moment configuration (spin init block)
        NOTE: this writes a minimal control.in; you should tune numerical settings for production.
        """
        # TODO: add functionality to optimize the cell size as done in doped
        #  alternatively pass ase_atoms and use that instead of pymatgen structure
        #  add possibility to create electron polarons by introducing an oxygen vacancy
        #  now the default is hse06, add functionality to choose between 3 different options:
        #  1) perform calculations with a single hybrid functional (i.e. hse06 or pbe0)
        #  2) perform calculations with only DFT+U, add functionality to write occupation matrix control
        #  3) firstly perform DFT+U with occ matrix control, then hybrid
        #  4) firstly perform hybrid calculation with atom with one extra electron placed on the electron polaron position, then a second hybrid with the original config
        #  add possibility to choose between relaxation or just scf run
        #  add here the possibility to have electron and hole polarons at the same time (maybe useful?)

        outdir = Path(outdir)
        if outdir.exists():
            shutil.rmtree(outdir)  # remove previous run
        outdir.mkdir(parents=True, exist_ok=True)

        # supercell structure (pymatgen)
        scell = self.structure * supercell

        # assign initial magmom
        if set_site_magmoms:
            site_index_supercell, magmoms = self._create_magmoms(
                scell, site_index, spin_moment
            )
            scell.add_site_property("magmom", magmoms)
        else:
            site_index_supercell = self._find_central_site_general_supercell(scell, site_index)

        total_charge = self._get_total_charge(scell, site_index_supercell)
        scell.remove_oxidation_states()
        # Build geometry.in via pymatgen helper
        geom = AimsGeometry.from_structure(scell)
        geom_file = outdir / "geometry.in"
        geom.write_file(geom_file)

        species_defaults = {el: tier for el in scell.symbol_set}

        # Build basic control.in using AimsControlIn
        params = {
            "xc": xc,
            "species_defaults": species_defaults,
            "species_dir": species_dir,
            "k_grid": "1 1 1",
            "default_initial_moment": 0.0,
            "spin": "collinear",
            "charge": total_charge,
        }

        control = AimsControl(params)
        control.write_file(geom, outdir)
