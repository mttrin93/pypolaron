from typing import Dict, Tuple, List, Optional

from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
# from pyfhiaims import AimsControlIn, AimsGeometryIn
from pyfhiaims.geometry import AimsGeometry
from pyfhiaims.control import AimsControl
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet
from ase import Atoms
import numpy as np
import shutil
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Won't overwrite species")


class PolaronGenerator:
    def __init__(self, structure: Structure, polaron_type="electron"):
        """
        structure : pymatgen Structure
        polaron_type : "electron" (extra electron) or "hole" (missing electron)
        """
        self.structure = structure.copy()
        self.polaron_type = polaron_type
        self.oxidation_assigned = False
        self.cnn = CrystalNN()

    def assign_oxidation_states(self):
        """Assign oxidation states using Bond Valence Analyzer."""
        bva = BVAnalyzer()
        self.structure = bva.get_oxi_state_decorated_structure(self.structure)
        self.oxidation_assigned = True

    def propose_sites(self, max_sites=5):
        """
        Propose candidate sites for polaron localization.
        Returns list of tuples: (site_index, element, oxidation_state, coordination_number, score)
        """
        if not self.oxidation_assigned:
            self.assign_oxidation_states()

        # cnn = CrystalNN()
        candidates = []

        for i, site in enumerate(self.structure.sites):
            element = site.specie.symbol
            oxidation_state = site.specie.oxi_state
            coordination_number = self.cnn.get_cn(self.structure, i)

            score = self._score_site(element, oxidation_state, coordination_number)
            if score > 0:  # only keep plausible candidates
                candidates.append((i, element, oxidation_state, coordination_number, score))

        # sort by score (higher = more likely polaron site)
        candidates.sort(key=lambda x: -x[4])
        return candidates[:max_sites]

    def _score_site(self, element, oxidation_state, coordination_number):
        """
        Simple heuristic scoring:
        - electron polaron: reducible high-ox cations (Ti4+, Fe3+, Mn4+ ...) with low coordination
        - hole polaron: anions with -2 (like O2-) with low coordination
        - included Pauli electronegativity differences
        """
        # TODO:
        # 1) add scoring contributions based on Madelung potential (high potential more likely to host polarons)
        # 2)
        score = 0.
        elem = Element(element)
        if self.polaron_type == "electron":
            if oxidation_state >= +3:  # cations in high oxidation state
                score += oxidation_state * 2.
                score += max(0., 6. - coordination_number) * 0.5  # undercoordination bonus
                score += (4. - elem.X) * 0.5  # lower electronegativity = more reducible
        elif self.polaron_type == "hole":
            if oxidation_state <= -2:  # anions in low oxidation state
                score += abs(oxidation_state) * 2.
                score += max(0., 4. - coordination_number) * 0.5
                score += elem.X * 0.5  # higher electronegativity = better hole host
        return score

    def _find_central_site_general_supercell(self, scell, site_index):
        """
        Find the index of the atom in the supercell corresponding to
        a primitive cell site, located closest to the geometric center.
        Works for both cubic and non-cubic supercells.
        """
        if isinstance(site_index, int):
            site_index = [site_index]

        center = np.mean([site.coords for site in scell.sites], axis=0)
        used_indices = set()
        mapped_indices = []

        for index in site_index:
            target_specie = self.structure[index].specie
            candidates = [(i, site) for i, site in enumerate(scell.sites)
                          if site.specie == target_specie and i not in used_indices]

            distances = np.linalg.norm([site.coords - center for _, site in candidates], axis=1)
            index_min_distance_center = candidates[np.argmin(distances)][0]
            used_indices.add(index_min_distance_center)
            mapped_indices.append(index_min_distance_center)

        return mapped_indices

    def _create_magmoms(self, scell, site_index, spin_moment):
        """
        Initialize the initial magmoms
        """
        site_index_supercell = self._find_central_site_general_supercell(scell, site_index)

        magmoms = np.array([0.] * len(scell))
        magmoms[site_index_supercell] = spin_moment
        return site_index_supercell, magmoms

    def prepare_for_ase(self, site_index, supercell=(2, 2, 2),
                        spin_moment=1.0, total_charge=0, set_site_magmoms=True):
        """
        Build ASE Atoms supercell and attach initial magnetic moments and metadata.
        Returns ASE Atoms instance ready to pass to ASE calculators or to be written out.
        """
        # build pymatgen supercell
        scell = self.structure * supercell

        # convert to ASE
        ase_atoms = AseAtomsAdaptor.get_atoms(scell)

        # assign initial magmom
        site_index_supercell, magmoms = self._create_magmoms(scell, site_index, spin_moment)
        ase_atoms.set_initial_magnetic_moments(magmoms)

        ase_atoms.info["polaron_site_index_in_supercell"] = site_index_supercell
        # attach total charge as metadata
        ase_atoms.info["polaron_total_charge"] = total_charge  # user-defined tag

        return ase_atoms

    def write_vasp_input_files(self, site_index, supercell=(2, 2, 2),
                              spin_moment=1.0, outdir="polaron_calc"):
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
        site_index_supercell, magmoms = self._create_magmoms(scell, site_index, spin_moment)
        scell.add_site_property("magmom", magmoms)

        nelect = len(scell) * 8
        if self.polaron_type == 'electron':
            nelect += 1. * len(site_index_supercell)
        elif self.polaron_type == 'hole':
            nelect -= 1. * len(site_index_supercell)
        else:
            raise ValueError(f"Unknown polaron type: {self.polaron_type}")

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

    def write_fhi_aims_input_files(self, site_index, supercell=(2, 2, 2),
                              add_charge=None, spin_moment=1.0,
                              xc="hse06", tier="tight", species_dir='./',
                              outdir='./fhi_aims_files'):
        """
        Write a simple FHI-aims 'geometry.in' and 'control.in' that:
        - builds the supercell geometry
        - sets total charge (if add_charge != 0) into the control.in
        - sets spin collinear and a simple initial moment configuration (spin init block)
        NOTE: this writes a minimal control.in; you should tune numerical settings for production.
        """
        #TODO: add functionality to optimize the cell size as done in doped
        # alternatively pass ase_atoms and use that instead of pymatgen structure
        # add possibility to create electron polarons by introducing an oxygen vacancy
        # now the default is hse06, add functionality to choose between 3 different options:
        # 1) perform calculations with a single hybrid functional (i.e. hse06 or pbe0)
        # 2) perform calculations with only DFT+U, add functionality to write occupation matrix control
        # 3) firstly perform DFT+U with occ matrix control, then hybrid
        # 4) firstly perform hybrid calculation with atom with one extra electron placed on the electron polaron position, then a second hybrid with the original config
        # add possibility to choose between relaxation or just scf run
        # add possibility to choose to set magmoms are not, not setting them still works
        # add here the possibility to have electron and hole polarons at the same time (maybe useful?)

        outdir = Path(outdir)
        if outdir.exists():
            shutil.rmtree(outdir)  # remove previous run
        outdir.mkdir(parents=True, exist_ok=True)

        # supercell structure (pymatgen)
        scell = self.structure * supercell

        # assign initial magmom
        site_index_supercell, magmoms = self._create_magmoms(scell, site_index, spin_moment)
        scell.add_site_property("magmom", magmoms)

        if add_charge is None:
            if self.polaron_type == 'electron':
                add_charge = -1. * len(site_index_supercell)
            elif self.polaron_type == 'hole':
                add_charge = 1. * len(site_index_supercell)
            else:
                raise ValueError(f"Unknown polaron type: {self.polaron_type}")

        total_charge = getattr(scell, "charge", 0) + add_charge
        scell.remove_oxidation_states()
        # Build geometry.in via pymatgen helper
        geom = AimsGeometry.from_structure(scell)
        geom_file = outdir / 'geometry.in'
        geom.write_file(geom_file)

        species_defaults = {el: tier for el in scell.symbol_set}

        # Build basic control.in using AimsControlIn
        params = {
            'xc': xc,
            'species_defaults': species_defaults,
            'species_dir': species_dir,
            'k_grid': '1 1 1',
            'default_initial_moment': 0.,
            'spin': 'collinear',
            'charge': int(total_charge),
        }

        control = AimsControl(params)
        control.write_file(geom, outdir)

