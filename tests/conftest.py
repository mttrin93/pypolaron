import pytest
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition
from pyfhiaims.geometry import AimsGeometry
from pathlib import Path


# This fixture provides a Rutile TiO2 structure (Electron Polaron host)
@pytest.fixture(scope="session")
def tio2_rutile():
    """
    Rutile (TiO2) structure, known to host electron polarons on Ti4+ sites.
    Space group P4_2/mnm (No. 136).
    Ti sites (0,0,0) and O sites (0.305, 0.305, 0).
    All Ti sites are equivalent by symmetry.
    """
    lattice = Lattice.from_parameters(
        a=4.5937, b=4.5937, c=2.9587, alpha=90, beta=90, gamma=90
    )
    species = ["Ti", "Ti", "O", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.305, 0.305, 0.0],
        [0.695, 0.695, 0.0],
        [0.195, 0.805, 0.5],
        [0.805, 0.195, 0.5],
    ]
    # IMPORTANT: We use Structure.from_spacegroup to get the standard setting,
    # but manually defining coordinates is sometimes more robust for testing.
    # Let's ensure oxidation states are correctly assigned by the BVA *later*.
    structure = Structure(lattice, species, coords)
    return structure


# This fixture provides a Rock Salt NiO structure (Hole Polaron host)
@pytest.fixture(scope="session")
def nio_rocksalt():
    """
    NiO rock salt structure, known to host hole polarons (often O^- or Ni3+).
    Space group Fm-3m (No. 225).
    """
    lattice = Lattice.cubic(4.17)
    species = ["Ni"] * 4 + ["O"] * 4
    # coords = [
    #     [0.0, 0.0, 0.0], # Ni
    #     [0.5, 0.0, 0.0]  # O
    # ]
    coords = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],  # Ni
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.5, 0.5, 0.5],  # O
    ]
    # Add a second formula unit to ensure better symmetry finding (8 atoms in conventional cell)
    structure = Structure(
        lattice, species, coords, coords_are_cartesian=False, site_properties=None
    )
    return structure


# This fixture provides a simple structure for symmetry testing (many equivalent sites)
@pytest.fixture(scope="session")
def symmetry_test_structure():
    """
    Simple P-1 unit cell with 8 identical atoms to ensure symmetry filtering works.
    Only one site should be returned by the generator.
    """
    lattice = Lattice.cubic(5.0)
    species = ["Mg", "Mg"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ]
    return Structure(lattice, species, coords)


# --- Fixture to read the AIMS file ---
@pytest.fixture(scope="session")
def lto_aims_structure():
    """
    Reads the mock aims_structure.in file using pymatgen's file parser.
    """
    data_dir = Path(__file__).parent / "data"  # points to tests/data/
    aims_filepath = data_dir / "geometry.in"

    # Parse the file using pymatgen
    return AimsGeometry.from_file(aims_filepath).structure
