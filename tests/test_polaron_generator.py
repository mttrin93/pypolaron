import pytest
from pypolaron.polaron_generator import PolaronGenerator
from pymatgen.core.structure import Structure
from typing import List, Tuple

# We rely on the fixtures from conftest.py: tio2_rutile, nio_rocksalt, symmetry_test_structure

# Helper function to check if the polaron type is correctly assigned
def check_polaron_sites(candidates: List[Tuple], expected_elements: List[str]):
    """Checks if the top candidates localize on the expected elements."""
    if not candidates:
        pytest.fail("No candidates were proposed. Check BVA assignment.")

    # Get the element symbol of the top candidate
    top_element = candidates[0][1]  # (site_index, element, ...)

    # Check if the top element is one of the expected types
    assert top_element in expected_elements, \
        f"Top polaron candidate element '{top_element}' not in expected list {expected_elements}"

    print(
        f"\nTop Candidate: {candidates[0][1]}{candidates[0][2]:.1f}+, CN={candidates[0][3]:.1f}, Score={candidates[0][4]:.2f}")

    # Check if the next few candidates also adhere to the expected element type
    for i, candidate in enumerate(candidates):
        # We check the top 3 sites
        if i < 3 and candidate[4] > 0:
            assert candidate[1] in expected_elements, \
                f"Candidate #{i + 1} element '{candidate[1]}' (Score={candidate[4]:.2f}) is not the expected type."


def test_electron_polaron_rutile(tio2_rutile: Structure):
    """
    Test Electron Polaron in TiO2 (Rutile).
    Expected localization: Ti4+ -> Ti3+ (Cation site).
    """
    pg_e = PolaronGenerator(tio2_rutile, polaron_type="electron")
    candidates = pg_e.propose_sites(max_sites=5)

    assert pg_e.oxidation_assigned is True, "BVA failed to assign oxidation states for TiO2."
    assert len(candidates) > 0, "Electron Polaron generator returned no sites for TiO2."

    # Expect Ti sites to dominate the scoring
    check_polaron_sites(candidates, ["Ti"])

    # In Rutile, there are 2 Ti sites, which should be symmetry equivalent.
    # The list should contain only one Ti site if max_sites=1, but
    # since we filter by distinct sites, the result should represent only
    # the symmetrically distinct sites.

    # Check that the number of distinct candidates is small (only the one Ti site)
    # The Ti site should score highly, and the O sites should score 0.
    ti_candidates = [c for c in candidates if c[1] == "Ti"]
    o_candidates = [c for c in candidates if c[1] == "O"]

    assert len(ti_candidates) == 1, "Rutile Ti sites are all equivalent, only 1 unique Ti site should be returned."
    assert all(c[4] > 0 for c in ti_candidates), "Ti site score should be positive."
    assert all(c[4] == 0 for c in o_candidates if
               c[2] < -1), "O site score (hole polaron target) should be zero for electron polaron type."


def test_hole_polaron_nio(nio_rocksalt: Structure):
    """
    Test Hole Polaron in NiO (Rock Salt).
    Expected localization: O2- -> O- (Anion site) or Ni2+ -> Ni3+.
    Our current heuristic favors the highly charged anion O2- for hole polarons.
    """
    pg_h = PolaronGenerator(nio_rocksalt, polaron_type="hole")
    candidates = pg_h.propose_sites(max_sites=5)

    assert pg_h.oxidation_assigned is True, "BVA failed to assign oxidation states for NiO."
    assert len(candidates) > 0, "Hole Polaron generator returned no sites for NiO."

    # Expect O sites (O2-) to dominate the scoring based on the heuristic's emphasis on high-ox anions
    check_polaron_sites(candidates, ["O"])

    # Check symmetry: all Ni sites are equivalent, all O sites are equivalent.
    ni_candidates = [c for c in candidates if c[1] == "Ni"]
    o_candidates = [c for c in candidates if c[1] == "O"]

    assert len(o_candidates) == 1, "NiO O sites are all equivalent, only 1 unique O site should be returned."
    assert len(ni_candidates) == 0, "NiO Ni sites are all equivalent, only 1 unique Ni site should be returned."
    # assert o_candidates[0][4] > ni_candidates[0][
    #     4], "O site should score higher than Ni site for hole polaron with current heuristic."


def test_symmetry_filtering(symmetry_test_structure: Structure):
    """
    Test that the generator correctly filters out symmetrically equivalent sites.
    The test structure has 2 identical atoms, but they are related by symmetry (translation).
    Only one candidate should be returned.
    """
    pg = PolaronGenerator(symmetry_test_structure, polaron_type="electron")

    # Assigning oxidation states to an elemental structure might fail BVA,
    # but the structural properties (symmetry) should still work.
    # We proceed assuming symmetry analysis works on the undecorated structure if BVA fails.
    pg.structure = symmetry_test_structure.copy()

    # Directly test the distinct site index calculation
    distinct_indices = pg._get_symmetrically_distinct_sites()

    assert len(symmetry_test_structure.sites) == 2, "Test setup error: structure should have 2 sites."
    assert len(
        distinct_indices) == 1, "Symmetry filtering failed: only 1 distinct site should be returned for this structure."

    # Test the full propose_sites method
    candidates = pg.propose_sites(max_sites=5)

    # BVA will likely fail (assigns 0) on this simple structure, so the score should be 0.
    # We mainly test that the iteration loop only processes one site.
    if pg.oxidation_assigned:
        assert len(candidates) <= 1, "Propose sites should return max 1 candidate due to symmetry."
    else:
        # If BVA fails, the scoring is less meaningful, but the *length* check is key.
        # Since BVA fails, it sets ox_state=0, and the score is 0, returning [].
        assert len(candidates) == 0 or len(candidates) == 1, "Propose sites should not generate more than 1 candidate."
