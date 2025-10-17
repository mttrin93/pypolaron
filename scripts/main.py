# from pypolaron import PolaronGenerator
#
#
# def main():
#     generator = PolaronGenerator(...)
#     generator.propose_sites()
#     print("Done!")
#
#
# if __name__ == "__main__":
#     main()

import os
import sys
from typing import List, Tuple, Union, Any

# --- Module Setup ---
# Assuming 'pypolaron' is the root directory containing 'polaron_generator.py'
# This block ensures the script can find your PolaronGenerator class
try:
    # Adjust this path based on your exact package structure if needed
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pypolaron.polaron_generator import PolaronGenerator
except ImportError as e:
    print(f"Error importing PolaronGenerator: {e}")
    print("Please ensure your package is correctly installed or run this script from the project root.")
    sys.exit(1)

# Pymatgen Imports
try:
    from pymatgen.core.structure import Structure
    from pymatgen.ext.matproj import MPRester
except ImportError:
    print("Error: Pymatgen must be installed to run this script.")
    sys.exit(1)

# --- Configuration ---
# NOTE: Replace with your actual Materials Project API Key or set as environment variable
MP_API_KEY = os.environ.get("MP_API_KEY", "YOUR_PLACEHOLDER_KEY")


def load_structure_from_source() -> Optional[Structure]:
    """Interactively asks user for structure source and loads it."""
    source_type = input("Load structure from (F)ile path or (M)aterials Project ID? [F/M]: ").strip().upper()

    if source_type == 'M':
        mp_id = input("Enter Materials Project ID (e.g., mp-2657): ").strip()
        if not MP_API_KEY or MP_API_KEY == "YOUR_PLACEHOLDER_KEY":
            print("ERROR: MP_API_KEY is not set. Cannot fetch structure.")
            return None
        try:
            with MPRester(MP_API_KEY) as mpr:
                # We use the conventional cell, as it's often more intuitive for visualization/supercell
                structure = mpr.get_structure_by_material_id(mp_id, final=True, conventional_unit_cell=True)
                print(f"Successfully loaded structure {mp_id}: {structure.formula} ({len(structure)} sites).")
                return structure
        except Exception as e:
            print(f"ERROR fetching MP ID {mp_id}: {e}")
            return None

    elif source_type == 'F':
        file_path = input("Enter local file path (e.g., POSCAR or structure.cif): ").strip()
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at {file_path}")
            return None
        try:
            structure = Structure.from_file(file_path)
            print(f"Successfully loaded structure from file: {structure.formula} ({len(structure)} sites).")
            return structure
        except Exception as e:
            print(f"ERROR reading file {file_path}: {e}")
            return None

    else:
        print("Invalid choice.")
        return None


def display_candidates(candidates: List[Any], title: str):
    """Prints the ranked list of candidates in a readable format."""
    if not candidates:
        print("\n--- No Plausible Candidates Found ---")
        return

    print(f"\n--- Top {len(candidates)} {title} Candidates ---")
    for i, candidate in enumerate(candidates):
        if title == "Vacancy-Induced Di-Polaron":
            # Format: (vacancy_index, (polaron_index_A, polaron_index_B), final_di_polaron_score)
            v_idx, p_pair, score = candidate
            v_site = "V_O"  # Placeholder for vacancy type
            print(
                f"Rank {i + 1}: Vacancy Site {v_idx} ({v_site}) + Di-Polaron on Cations {p_pair} | Score: {score:.3f}")
        else:  # Single or Dimer Polaron
            # Format: ((site_indices), final_score, polaron_type_str)
            indices, score, site_type = candidate

            site_list = ", ".join([f"Idx {idx} ({pg.structure.sites[idx].specie.symbol})" for idx in indices])
            print(f"Rank {i + 1}: Type: {site_type.capitalize()} | Sites: {site_list} | Score: {score:.3f}")


def run_analysis(pg: PolaronGenerator):
    """Guides the user through the polaron site proposal analysis."""
    print(f"\n--- Running Analysis for {pg.polaron_type.upper()} Polaron ---")

    analysis_type = input(
        "Choose Analysis Type: (S)ingle/Dimer Polaron or (V)acancy-Induced Di-Polaron? [S/V]: ").strip().upper()

    max_sites = 5
    try:
        max_sites = int(input(f"How many top candidate sites should be proposed? (Default: 5): ") or 5)
    except ValueError:
        pass

    if analysis_type == 'S':
        # Single and Dimer Analysis
        candidates = pg.propose_sites(max_sites=max_sites)
        display_candidates(candidates, "Single/Dimer Polaron")

    elif analysis_type == 'V':
        # Vacancy-Induced Di-Polaron Analysis (Only valid for electron polaron)
        if pg.polaron_type != "electron":
            print("\nERROR: Vacancy-induced di-polaron analysis is only chemically relevant for electron polarons.")
            return

        print("\n--- Step 1: Proposing Oxygen Vacancy Candidates ---")
        vacancy_candidates = pg.propose_vacancy_sites(max_sites=max_sites)

        if not vacancy_candidates:
            print("No suitable vacancy sites found.")
            return

        # Run Di-Polaron Analysis on the top vacancy candidates
        print("\n--- Step 2: Proposing Di-Polarons on Neighboring Cations ---")
        di_polaron_candidates = pg.propose_di_polaron_sites(
            vacancy_site_candidates=vacancy_candidates,
            max_di_polaron_candidates=max_sites
        )
        display_candidates(di_polaron_candidates, "Vacancy-Induced Di-Polaron")

    else:
        print("Invalid analysis choice.")


def main():
    print("=========================================")
    print(" PyPolaron High-Throughput Generator CLI ")
    print("=========================================")

    # 1. Load Structure
    structure = load_structure_from_source()
    if structure is None:
        return

    # 2. Get Polaron Type
    polaron_type_input = input("Enter Polaron Type (E)lectron or (H)ole: [E/H]: ").strip().upper()
    if polaron_type_input == 'E':
        polaron_type = "electron"
    elif polaron_type_input == 'H':
        polaron_type = "hole"
    else:
        print("Invalid polaron type choice. Exiting.")
        return

    # 3. Initialize Generator and Run Analysis
    pg = PolaronGenerator(structure, polaron_type=polaron_type)
    pg.assign_oxidation_states()  # Ensure oxidation states are set once

    run_analysis(pg)

    print("\n=========================================")
    print(" Analysis Complete. Ready for DFT Input Generation.")
    print("=========================================")


if __name__ == "__main__":
    main()
