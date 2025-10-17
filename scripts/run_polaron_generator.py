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
    print(candidates)
    if not candidates:
        log.info("No Plausible Candidates Found. Exiting.")
        return

    for i, candidate in enumerate(candidates):
        if title == "oxygen vacancy":

            log.info(f"Top {len(candidates)} {title} candidates:")
            index, element, score = candidate
            log.info(f"Type: {title} | Atom index {index} | Score: {score:.3f}")

        elif title in ["electron", "hole"]:

            log.info(f"Top {len(candidates)} {title} polaron candidates:")
            index, element, oxidation_state, coordination_number, score = candidate
            log.info(f"Type: {title} polaron | Atom index {index} | Element : {element}"
                     f"  | Oxidation State: {oxidation_state} | "
                     f"Coordination Number: {coordination_number} | Score: {score:.3f}")


def main(args):
    parser = argparse.ArgumentParser(prog="pypolaron", description="Toolkit for automated DFT polaron calculations "
                                                                   "with FHI-AIMS and VASP.\n" +
                                                                   "version: {}".format(__version__),
                                     formatter_class=argparse.RawTextHelpFormatter)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a structure file (i.e., POSCAR, .cif, geometry.in, structure.xyz)"
    )
    source_group.add_argument(
        "-m", "--mp-query",
         type=str,
         help="Materials Project ID (i.e., mp=2657) or a chemical formula/composition (i.e. TiO2)"
    )

    parser.add_argument(
        "-mp-key", "--mp-api-key",
        type=str,
        default=os.environ.get("MP_API_KEY"),
        help="Materials Project API Key. Defaults to the MP_API_KEY environment variable",
    )

    parser.add_argument(
        "-polaron-type", "--polaron-type",
        type=str,
        default='electron',
        help="Polaron type: electron or hole polaron"
    )

    parser.add_argument(
        "-polaron-number", "--polaron-number",
        type=int,
        default=1,
        help="Total number of additional polarons. If set to zero, specify a oxygen vacancy number larger than zero"
    )

    parser.add_argument(
        "-oxygen-vacancy", "--oxygen-vacancy-number",
        type=int,
        default=0,
        help="Number of added oxygen vacancies. "
             "The creation of oxygen vacancies leads to the formation of two electron polarons"
    )

    args_parse = parser.parse_args(args)

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
                    log.info(f"Conventional cell for structure with ID/composition {mp_id_or_comp} loaded from MP")
                else:
                    # Fetching the most stable entry for a given composition
                    entries = mpr.get_entries(mp_id_or_comp)
                    if entries:
                        best_entry = min(entries, key=lambda e: e.data.get('formation_energy_per_atom', float('inf')))
                        structure = best_entry.structure
                        log.info(f"Most stable structure with ID/composition {mp_id_or_comp} loaded from MP")
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
        log.warning(f"No polarons will be created. Exiting. ")
        return

    if number_of_oxygen_vacancies > 0:
        log.info(f"{number_of_oxygen_vacancies} oxygen vacancy(ies) will be considered, that"
                 f" correspond(s) to {number_of_oxygen_vacancies*2} electron polarons")

    log.info(f"The calculations will run with {number_of_polarons} additional {polaron_type} polaron(s)")

    # TODO: if polaron_type is hole we cannot use occupation matrix control method

    # 3. Initialize Generator and Run Analysis
    polaron_generator = PolaronGenerator(structure, polaron_type=polaron_type)

    if polaron_type in ["electron", "hole"] and number_of_polarons > 0:
        polaron_candidates = polaron_generator.propose_sites(max_sites=number_of_polarons)
        display_candidates(polaron_candidates, polaron_type)

    if number_of_oxygen_vacancies > 0:
        if polaron_type != "electron":
            print("Calculations with both hole polaron and oxygen vacancies are not possible")
            return
        oxygen_vacancies_candidates = polaron_generator.propose_vacancy_sites(max_sites=number_of_oxygen_vacancies)
        display_candidates(oxygen_vacancies_candidates, 'oxygen vacancy')

    polaron_generator.assign_oxidation_states()  # Ensure oxidation states are set once
    log.info("Analysis Complete. Ready for DFT Input Generation")


if __name__ == "__main__":
    main(sys.argv[1:])
