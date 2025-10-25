from typing import Dict, Tuple, List, Optional, Union

from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element

from pypolaron.utils import parse_aims_plus_u_params, DftSettings, \
    generate_occupation_matrix_content, create_attractor_structure
# from pyfhiaims import AimsControlIn, AimsGeometryIn
from pyfhiaims.geometry import AimsGeometry
from pyfhiaims.control import AimsControl
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase import Atoms
from collections import Counter
import numpy as np
import shutil
import os
import copy
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
        self, scell: Structure, site_index: List[int]
    ) -> List[int]:
        """
        Find the index of the atom in the supercell corresponding to
        a primitive cell site, located closest to the geometric center.
        Works for both cubic and non-cubic supercells.
        """
        scell_center = np.mean([site.coords for site in scell.sites], axis=0)
        used_indices = set()
        mapped_indices = []

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

    def propose_vacancy_sites(self, max_sites: int = 5) -> List[Tuple[int, str, float]]:
        """
        Propose candidate sites for Oxygen Vacancy (Vo) creation, which leads
        to two electron polarons.
        Returns list of tuples: (site_index, element, coordination_number)
        """
        if not self.oxidation_assigned:
            self.assign_oxidation_states()

        distinct_indices = self._get_symmetrically_distinct_sites()
        candidates = []

        for index in distinct_indices:
            site = self.structure.sites[index]
            element = site.specie.symbol
            coordination_number = self.crystal_near_neighbors.get_cn(self.structure, index)

            # --- Primary Filter: Only Anions (O is the most common) ---
            # if not site.specie.is_anion:
            #     continue
            if element != 'O':
                continue

            # Simple vacancy score: prioritize undercoordinated, less electronegative anions
            score = 0.5 * (max(0., 6. - coordination_number) + (4. - Element(element).X))

            # We only care about the vacancy location index for now.
            if score > 0:
                candidates.append(
                    (index, element, score)
                )

        # sort by score (higher = more likely vacancy site)
        candidates.sort(key=lambda x: -x[2])

        # Output format: (site_index, element, score)
        return candidates[:max_sites]

    def _get_total_charge(
        self,
        scell: Structure,
        polaron_site_indices: List[int]
    ) -> int:
        """
        Calculates the total charge for the supercell.
        """
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

    # TODO: we should write functions to estimate the HFSCREEN (=alpha) parameter of hse06 via pasquarello
    # TODO: we should write functions to estimate the DFT+U parameters via pasquarello
    def write_vasp_input_files(
        self,
        site_index: Union[int, List[int]],
        vacancy_site_index: Union[int, List[int]],
        settings: DftSettings,
        outdir: str = "polaron_vasp_calc",
        is_charged_polaron_run: bool = True,
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
        scell = self.structure * settings.supercell

        if isinstance(site_index, int):
            site_index = [site_index]
        if isinstance(vacancy_site_index, int):
            vacancy_site_index = [vacancy_site_index]

        if is_charged_polaron_run:
            total_charge = self._get_total_charge(scell, site_index)
        else:
            total_charge = 0

        if vacancy_site_index is not None:
            vacancy_site_index_supercell = self._find_central_site_general_supercell(scell, vacancy_site_index)
            if vacancy_site_index_supercell:
                scell.remove_sites(vacancy_site_index_supercell)
                total_charge += 2 * len(vacancy_site_index_supercell)

        # TODO: add possibility to assign initial magnetic moment even though the calculation is for pristine
        # assign initial magmom
        if settings.set_site_magmoms and is_charged_polaron_run:
            site_index_supercell, magmoms = self._create_magmoms(
                scell, site_index, settings.spin_moment
            )
            scell.add_site_property("magmom", magmoms)
        # else:
        #     site_index_supercell = self._find_central_site_general_supercell(scell, site_index)

        nelect = sum(site.specie.Z for site in scell) - total_charge

        user_incar = {
            "NELECT": nelect,
            "ISPIN": 2,
            "MAGMOM": Counter(scell.site_properties.get("magmom", None)),
            "EDIFF": 1e-5,
            "NELM": 200,
            "PREC": "Accurate",
        }

        # add calculation type settings
        calc_type_lower = settings.calc_type.lower()
        if calc_type_lower == 'scf':
            user_incar["NSW"] = 0
            user_incar["IBRION"] = -1
        # relax-all: both atoms and cell, relax-atoms: atoms only
        elif calc_type_lower in ["relax-atoms", "relax-all"]:
            user_incar["NSW"] = 99
            user_incar["IBRION"] = 2
            if calc_type_lower == "relax-all":
                user_incar["ISIF"] = 3

        # add functional-specific settings
        if settings.functional.lower() == "hse06":
            user_incar.update({
                "LHFCALC": True,  # Enable hybrid functional
                "HFSCREEN": 0.2,  # Screening parameter for HSE06
                "AEXX": 0.25,  # Fraction of exact exchange
                "ALGO": "All",  # Recommended for hybrids
            })
            user_incar["LDAU"] = False # turn off DFT+U with HSE
        elif settings.functional.lower() == "pbeu":
            user_incar.update({
                "LDAU": True,
                "LDAUTYPE": 2,  # Common DFT+U setting
                "LDAUL": [0, -1, 2, 3],  # Example L values
                "LDAUU": [0, 0, 5.0, 0],  # Example U values (Requires user to set these carefully!)
                "LMAXMIX": 4,
            })
        else: # plain PBE
            user_incar["LDAU"] = False
            user_incar["LHFCALC"] = False

        vis = MPRelaxSet(scell, user_incar_settings=user_incar)
        # vis.write_input(outdir, potcar_spec=True)

        # TODO: test this pseudopotential assignmnets
        original_psp_dir = os.environ.get("VASP_PSP_DIR")

        try:
            if settings.species_dir:
                os.environ["VASP_PSP_DIR"] = settings.species_dir
            vis.write_input(outdir, potcar_spec=True)
        finally:
            if original_psp_dir is not None:
                os.environ["VASP_PSP_DIR"] = original_psp_dir
            elif settings.species_dir:
                del os.environ["VASP_PSP_DIR"]

    def write_fhi_aims_input_files(
        self,
        site_index: Union[int, List[int]],
        vacancy_site_index: Union[int, List[int]],
        settings: DftSettings,
        outdir: str = "./fhi_aims_files",
        is_charged_polaron_run: bool = True,
    ):
        """
        Write a simple FHI-aims 'geometry.in' and 'control.in' that:
        - builds the supercell geometry
        - sets total charge (if add_charge != 0) into the control.in
        - sets spin collinear and a simple initial moment configuration (spin init block)
        NOTE: this writes a minimal control.in; you should tune numerical settings for production.
        """

        # TODO: add functionality to optimize the cell size as done in doped
        #  now the default is hse06, add functionality to choose between 3 different options:
        #  1) perform calculations with a single hybrid functional (i.e. hse06 or pbe0)
        #  2) perform calculations with only DFT+U
        #  3) firstly perform DFT+U with occ matrix control, then hybrid
        #  4) firstly perform hybrid calculation with atom with one extra electron placed on the electron polaron position,
        #  then a second hybrid with the original config

        outdir = Path(outdir)
        if outdir.exists():
            shutil.rmtree(outdir)  # remove previous run
        outdir.mkdir(parents=True, exist_ok=True)

        # build supercell
        scell = self.structure * settings.supercell

        if isinstance(site_index, int):
            site_index = [site_index]
        if isinstance(vacancy_site_index, int):
            vacancy_site_index = [vacancy_site_index]

        if is_charged_polaron_run:
            total_charge = self._get_total_charge(scell, site_index)
        else:
            total_charge = 0

        if vacancy_site_index is not None:
            vacancy_site_index_supercell = self._find_central_site_general_supercell(scell, vacancy_site_index)
            if vacancy_site_index_supercell:
                scell.remove_sites(vacancy_site_index_supercell)
                total_charge += 2 * len(vacancy_site_index_supercell)

        site_index_supercell = []
        # assign initial magmom
        if is_charged_polaron_run:
            site_index_supercell, magmoms = self._create_magmoms(
                scell, site_index, settings.spin_moment
            )
            if settings.set_site_magmoms:
                scell.add_site_property("magmom", magmoms)
        elif settings.attractor_elements is not None:
            site_index_supercell, _ = self._create_magmoms(
                scell, site_index, settings.spin_moment
            )

        if settings.attractor_elements:
            scell = create_attractor_structure(
                scell,
                site_index_supercell,
                settings.attractor_elements
            )

        scell.remove_oxidation_states()
        # Build geometry.in via pymatgen helper
        geom = AimsGeometry.from_structure(scell)
        geom_file = outdir / "geometry.in"
        geom.write_file(geom_file)

        # species_defaults = {el: tier for el in scell.symbol_set}

        # Build basic control.in using AimsControlIn
        params = {
            "relativistic": "atomic_zora scalar",
            "collect_eigenvectors": ".false.",
            "k_grid": "1 1 1",
            "occupation_type": "gaussian 0.001",
            "default_initial_moment": 0.0,
            "spin": "collinear",
            "charge_mix_param": 0.3,
            "mixer": "pulay",
            "charge": total_charge,
            "species_dir": settings.species_dir,
            "compute_forces": ".true." if settings.calc_type.lower() in ["relax-atoms", "relax-all"] else ".false."
        }

        if settings.fix_spin_moment is not None:
            params["fix_spin_moment"] = settings.fix_spin_moment

        if not settings.disable_elsi_restart:
            params.update({
                "elsi_restart": "read_and_write 100",
                "elsi_restart_use_overlap": ".true.",
            })

        is_pbeu_run = settings.functional.lower() == "pbeu"
        # Add functional-specific settings
        if settings.functional.lower() == "hse06":
            params.update({
                 "xc": "hse06 0.11",
                 "hse_unit": "bohr-1",
                 "hybrid_xc_coeff": {settings.alpha},
            })
        elif settings.functional.lower() == "pbe0":
            params.update({
                 "xc": "pbe0",
                 "hybrid_xc_coeff": {settings.alpha},
            })
        elif is_pbeu_run:
            plus_u_values = parse_aims_plus_u_params(settings.hubbard_parameters)
            params.update({
                 "xc": "pbe",
                 "plus_u_petukhov_mixing": "1.0",
                 "plus_u_matrix_control": ".true.",
                 "plus_u": plus_u_values,
            })
        elif settings.functional.lower() == "pbe":
            params["xc"] = "pbe"
        else:
            raise ValueError(f"Unsupported functional '{settings.functional}'. "
                             f" Supported options are: 'hse06', 'pbe', or 'pbeu'.")

        # Add relaxation settings
        if settings.calc_type.lower() in ["relax-atoms", "relax-all"]:
            params["relax_geometry"] = "trm 1e-4"
            if settings.calc_type.lower() == "relax-all":
                params["relax_unit_cell"] = "full"

        control = AimsControl(params)
        control.write_file(geom, outdir)

        if is_pbeu_run and is_charged_polaron_run:
            occupation_matrix_content = generate_occupation_matrix_content(scell, site_index_supercell)

            occupation_matrix_file = outdir / "occupation_matrix_control.txt"
            occupation_matrix_file.write_text(occupation_matrix_content)
