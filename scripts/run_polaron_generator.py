import argparse
import logging
import os
import socket
import getpass
import sys
from typing import List, Tuple, Union, Any, Optional
from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pypolaron import __version__
from pypolaron.polaron_generator import PolaronGenerator
from pyfhiaims.geometry import AimsGeometry

hostname = socket.gethostname()
username = getpass.getuser()


LOG_FMT = '%(asctime)s %(levelname).1s - %(message)s'.format(hostname)
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()

DEFAULT_SEED = 42


def display_candidates(candidates: List[Any], title: str):
    """Prints the ranked list of candidates in a readable format."""
    if not candidates:
        log.info("No Plausible Candidates Found. Exiting.")
        return

    for i, candidate in enumerate(candidates):
        rank = i + 1
        if title == "oxygen vacancy":

            index, element, score = candidate
            log.info(f"{[rank]} Oxygen vacancy at atomic site index {index} | Score: {score:.3f}")

        elif title in ["electron", "hole"]:

            index, element, oxidation_state, coordination_number, score = candidate
            log.info(f"{[rank]} {title} polaron at atom site index {index} | Element : {element}"
                     f"  | Oxidation State: {oxidation_state} | "
                     f"Coordination Number: {coordination_number} | Score: {score:.3f}")

def select_and_generate(
        pg: PolaronGenerator,
        candidates: List[Any],
        analysis_title: str,
        args_parse: argparse.Namespace
):
    """Allows the user to select candidates and generates DFT input files."""

    if not candidates:
        return

    # Interactive selection of sites
    selection_input = input(
        f"\nEnter ranks to generate DFT inputs for (e.g., '1' or '1,3,5'). Type 'N' to skip: "
    ).strip().upper()

    if selection_input == 'N' or not selection_input:
        log.info("Skipping DFT input file generation.")
        return

    try:
        selected_ranks = [int(r.strip()) for r in selection_input.split(',')]
        if not all(1 <= r <= len(candidates) for r in selected_ranks):
            raise ValueError("Invalid rank selected.")
    except ValueError as e:
        log.warning(f"Invalid input: {e}. Please enter comma-separated numbers within the range.")
        return

    # Generate files for each selected candidate
    log.info(f"Starting DFT input generation for {len(selected_ranks)} candidate(s)...")

    for rank in selected_ranks:
        candidate = candidates[rank - 1]
        outdir_suffix = f"rank_{rank}"

        # Determine output directory
        base_outdir = f"{args_parse.dft_tool.lower()}_{args_parse.functional.lower()}_{args_parse.calc_type.lower()}"
        outdir = os.path.join(base_outdir, outdir_suffix)

        log.info(f"--- Generating inputs for Rank {rank} in directory: {outdir} ---")

        # --- Common Parameters ---
        common_kwargs = {
            "supercell": args_parse.supercell,
            "spin_moment": args_parse.spin_moment,
            "functional": args_parse.functional,
            "calc_type": args_parse.calc_type,
            "outdir": outdir,
            "set_site_magmoms": True,
        }

        # --- Case 1: Single/Dimer Polaron (Standard) ---
        if analysis_title in ["electron polaron", "hole polaron"]:
            # candidate is: ((site_indices), score, type_str)
            site_indices, _, site_type = candidate

            if args_parse.dft_tool.lower() == 'vasp':
                pg.write_vasp_input_files(
                    site_index=site_indices,
                    **common_kwargs
                )
            elif args_parse.dft_tool.lower() == 'aims':
                pg.write_fhi_aims_input_files(
                    site_index=site_indices,
                    **common_kwargs
                )
            log.info(f"Generated {args_parse.dft_tool.upper()} files for {site_type} polaron(s).")


        # --- Case 2: Vacancy-Induced Di-Polaron ---
        elif analysis_title == "vacancy-induced di-polaron":
            # candidate is: (v_idx, p_pair, score)
            v_idx, p_pair, _ = candidate

            # NOTE: For V_O, the charge state is fixed at +2 relative to the pristine cell,
            # meaning we need to pass the vacancy site to be removed AND the two polaron sites for magmoms.

            # The file writer must be updated in your PolaronGenerator to handle the `vacancy_index_primitive`
            # and the `polaron_site_indices` (p_pair) simultaneously.

            # Assuming the PolaronGenerator is extended to accept a dedicated 'vacancy_index_primitive' argument:
            vacancy_kwargs = {
                **common_kwargs,
                "polaron_site_indices": p_pair,  # Sites for magmoms
                "vacancy_index_primitive": v_idx,  # Site to remove
                "set_site_magmoms": True,
            }

            if args_parse.dft_tool.lower() == 'vasp':
                # The polaron sites (p_pair) are used for magmoms, vacancy_index is used for structure modification.
                # Since VASP writer currently takes site_index, we pass the sites for magmoms.
                # *** You must modify pg.write_vasp_input_files to accept and use the vacancy_index_primitive! ***
                log.warning(
                    "WARNING: Assuming PolaronGenerator.write_vasp_input_files accepts 'vacancy_index_primitive'.")
                pg.write_vasp_input_files(
                    site_index=p_pair,  # Polaron sites for magmoms
                    vacancy_index_primitive=v_idx,  # New argument your method must accept
                    **common_kwargs
                )

            elif args_parse.dft_tool.lower() == 'aims':
                # *** You must modify pg.write_fhi_aims_input_files to accept and use the vacancy_index_primitive! ***
                log.warning(
                    "WARNING: Assuming PolaronGenerator.write_fhi_aims_input_files accepts 'vacancy_index_primitive'.")
                pg.write_fhi_aims_input_files(
                    site_index=p_pair,  # Polaron sites for magmoms
                    vacancy_index_primitive=v_idx,  # New argument your method must accept
                    **common_kwargs
                )

            log.info(
                f"Generated {args_parse.dft_tool.upper()} files for V_O at index {v_idx} and di-polaron on {p_pair}.")

def main(args):
    parser = argparse.ArgumentParser(prog="pypolaron",
                                     description="Toolkit for automated DFT polaron calculations "
                                                                   "with FHI-AIMS and VASP.\n" +
                                                                   "version: {}".format(__version__),
                                     formatter_class=argparse.RawTextHelpFormatter
                                     )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a structure file (i.e., POSCAR, .cif, geometry.in, structure.xyz)"
    )
    source_group.add_argument(
        "-mq", "--mp-query",
         type=str,
         help="Materials Project ID (i.e., mp=2657) or a chemical formula/composition (i.e. TiO2)"
    )

    parser.add_argument(
        "-mak", "--mp-api-key",
        type=str,
        default=os.environ.get("MP_API_KEY"),
        help="Materials Project API Key. Defaults to the MP_API_KEY environment variable",
    )

    parser.add_argument(
        "-l", "--log",
        type=str,
        default="log.txt",
        help="log filename, default: log.txt",
    )

    parser.add_argument(
        "-pt", "--polaron-type",
        type=str,
        default='electron',
        help="Polaron type: electron or hole polaron"
    )

    parser.add_argument(
        "-pn", "--polaron-number",
        type=int,
        default=1,
        help="Total number of additional polarons (default 1). If set to zero, "
             " specify a oxygen vacancy number larger than zero"
    )

    parser.add_argument(
        "-ovn", "--oxygen-vacancy-number",
        type=int,
        default=0,
        help="Number of added oxygen vacancies. "
             "The creation of oxygen vacancies leads to the formation of two electron polarons"
    )

    parser.add_argument(
        "-s",
        "--supercell",
        type=int,
        nargs=3,
        default=(2, 2, 2),
        help="Supercell dimensions (a, b, c) as three space-separated integers (e.g., 2 2 2)",
    )

    parser.add_argument(
        "-dc",
        "--dft-code",
        type=str,
        choices=['vasp', 'aims'],
        required=True,
        help="DFT code to use: 'vasp' or 'aims'",
    )

    parser.add_argument(
        "-xf",
        "--xc-functional",
        type=str,
        default='hse06',
        help="DFT functional to use (e.g., 'pbe', 'pbeu', 'hse06').",
    )

    parser.add_argument(
        "-ct",
        "--calc-type",
        type=str,
        choices=['scf', 'relax-atoms', 'relax-all'],
        default='relax-atoms',
        help="Calculation type: 'scf' (static), 'relax-atoms' (ions only), or 'relax-all' (ions and cell).",
    )

    parser.add_argument(
        "-sm",
        "--spin-moment",
        type=float,
        default=1.0,
        help="Initial magnetic moment to set on the polaron site(s) for spin seeding.",
    )

    args_parse = parser.parse_args(args)

    if "log" in args_parse:
        log_file_name = args_parse.log
        log.info("Redirecting log into file {}".format(log_file_name))
        file_handler = logging.FileHandler(log_file_name, 'a')
        formatter = logging.Formatter(LOG_FMT)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    log.info("PyPolaron High-Throughput Generator command line interface ")
    log.info(f"Hostname: {hostname}")
    log.info(f"Username: {username}")
    log.info(f"pypolaron version: {__version__}")

    structure: Optional[Structure] = None

    if args_parse.file:
        structure_file_path = args_parse.file
        log.info(f"Try to read initial structure from the filepath: {os.path.abspath(structure_file_path)}")
        if not os.path.exists(structure_file_path):
            log.warning(f"ERROR: File not found at {structure_file_path}")
            return

        path = Path(structure_file_path)
        is_aims_file = path.name.lower() in ["geometry.in"] or path.suffix.lower() in [".in"]

        try:
            if is_aims_file:
                structure = AimsGeometry.from_file(path).structure
                log.info(f"File loaded with AIMS parser: {path.name}")
            else:
                structure = Structure.from_file(structure_file_path)
                log.warning(f"File loaded with universal pymatgen parser: {path.name}")
        except Exception as e:
            log.warning(f"ERROR reading file {structure_file_path}: {e}")
            return

    elif args_parse.mp_query:
        mp_id_or_comp = args_parse.mp_query
        # Determine API Key: CLI Argument > Environment Variable
        api_key = args_parse.mp_api_key

        if not api_key:
            log.warning("ERROR: Materials Project API Key is not set")
            log.warning("Please set the MP_API_KEY environment variable or provide it via --mp-api-key")
            return

        log.info(f"Try to read initial structure from Materials Project with ID/composition "
                 f"{mp_id_or_comp} and api_key {api_key}")
        try:
            with MPRester(api_key) as mpr:
                # Prioritize fetching the conventional cell for polaron supercell generation
                if mp_id_or_comp.startswith("mp-") or mp_id_or_comp.startswith("mvc-"):
                    structure = mpr.get_structure_by_material_id(mp_id_or_comp, conventional_unit_cell=True)
                    log.info(f"Conventional cell for structure with ID/composition {mp_id_or_comp} loaded from "
                             f" Materials Project database ")
                else:
                    # Fetching the most stable entry for a given composition
                    entries = mpr.get_entries(mp_id_or_comp)
                    if entries:
                        best_entry = min(entries, key=lambda e: e.data.get('formation_energy_per_atom', float('inf')))
                        structure = best_entry.structure
                        log.info(f"Most stable structure with ID/composition {mp_id_or_comp} loaded from Materials"
                                 f" Project database ")
                    else:
                        raise ValueError("No stable structure found for this query.")

        except Exception as e:
            log.warning(f"ERROR fetching MP data for '{mp_id_or_comp}': {e}")
            return

    if structure is None:
        log.info("Structure loading failed, Exiting.")
        return

    log.info(f"Successfully loaded structure: {structure.formula} ({len(structure)} sites)")

    # 2. Get Polaron Type (Interactive)
    polaron_type = args_parse.polaron_type.lower()
    if polaron_type not in ["electron", "hole"]:
        log.warning(f"Invalid polaron type choice: {polaron_type}. Exiting.")
        return

    number_of_polarons = args_parse.polaron_number
    number_of_oxygen_vacancies = args_parse.oxygen_vacancy_number

    if number_of_polarons == 0 and number_of_oxygen_vacancies == 0:
        log.warning(f"No polarons or oxygen vacancies will be created. Exiting. ")
        return

    if number_of_oxygen_vacancies > 0:
        log.info(f"{number_of_oxygen_vacancies} oxygen vacancy(ies) will be considered, that"
                 f" correspond(s) to {number_of_oxygen_vacancies*2} electron polarons")

    if polaron_type == 'electron':
        log.info(f"The calculations will run with {number_of_polarons} additional {polaron_type} polaron(s). "
                 f"In total {number_of_polarons + number_of_oxygen_vacancies*2} {polaron_type} polarons ")
    elif polaron_type == 'hole':
        log.info(f"The calculations will run with {number_of_polarons} additional {polaron_type} polaron(s).")

    # TODO: if polaron_type is hole we cannot use occupation matrix control method

    # 3. Initialize Generator and Run Analysis
    polaron_generator = PolaronGenerator(structure, polaron_type=polaron_type)
    polaron_generator.assign_oxidation_states()  # Ensure oxidation states are set once

    # TODO: should we do this analysis of the polaron candidates in the supercell or in the initial smaller cell?

    if polaron_type in ["electron", "hole"] and number_of_polarons > 0:
        polaron_candidates = polaron_generator.propose_sites(max_sites=number_of_polarons)

        log.info(f"Found {len(polaron_candidates)} unique {polaron_type} polaron candidate(s):")
        display_candidates(polaron_candidates, polaron_type)
    else:
        polaron_candidates = []

    if number_of_oxygen_vacancies > 0:
        oxygen_vacancies_candidates = polaron_generator.propose_vacancy_sites(max_sites=number_of_oxygen_vacancies)

        log.info(f"Found {len(oxygen_vacancies_candidates)} unique oxygen vacancy candidate(s):")
        display_candidates(oxygen_vacancies_candidates, 'oxygen vacancy')

        if oxygen_vacancies_candidates:
            electron_polaron_candidates_from_oxygen_vacancies = polaron_generator.propose_sites(
                max_sites=number_of_oxygen_vacancies*2 + number_of_polarons)

            if polaron_candidates:
                new_polaron_candidates = [
                    candidate for candidate in electron_polaron_candidates_from_oxygen_vacancies
                    if candidate not in polaron_candidates
                ]
            else:
                new_polaron_candidates = electron_polaron_candidates_from_oxygen_vacancies[
                                         :number_of_oxygen_vacancies*2 + 1
                                         ]

            log.info(f"Found {len(new_polaron_candidates)} unique "
                     f"electron polaron candidate(s) from oxygen vacancy generation:")
            display_candidates(new_polaron_candidates,'electron')

    log.info("Analysis Complete. Ready for DFT Input Generation")

    # TODO: start by specifing workdir where the DFT calculations will be performed


if __name__ == "__main__":
    main(sys.argv[1:])
