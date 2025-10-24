import argparse
import logging
import os
import socket
import getpass
import sys
from typing import List, Tuple, Union, Any, Optional, Dict, TypedDict
from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
# Import your package components:
from pypolaron import __version__
from pypolaron.polaron_generator import PolaronGenerator
from pypolaron.workflow import PolaronWorkflow
from pypolaron.utils import DftSettings, parse_aims_plus_u_params, DftParameters
from pyfhiaims.geometry import AimsGeometry

hostname = socket.gethostname()
username = getpass.getuser()

LOG_FMT = '%(asctime)s %(levelname).1s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


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
            log.info(f"{[rank]} {title} polaron at atomic site index {index} | Element : {element}"
                     f"  | Oxidation State: {oxidation_state} | "
                     f"Coordination Number: {coordination_number} | Score: {score:.3f}")

def main(args=None):
    parser = argparse.ArgumentParser(
        prog="pypolaron-run-attractor",
        description=f"Tool for Electron Attractor polaron localization method "
                    f" (sequential workflow). Version: {__version__}",
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

    # --- Core Attractor Setup ---
    parser.add_argument(
        "-ae", "--attractor-elements",
        type=str,
        required=True,
        help="Element symbol (e.g., Cr, Zr) used to substitute the host atom to create the localized potential well."
             "Provide either a single element symbol or a list of symbols. "
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
        help="Polaron type: electron or hole polaron. Default is electron."
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
             "The creation of oxygen vacancies leads to the formation of two electron polarons. Default is 0."
    )

    parser.add_argument(
        "-s", "--supercell-dims",
        type=int,
        nargs=3,
        default=(2, 2, 2),
        help="Supercell dimensions (a, b, c) as three space-separated integers. Default is (2, 2, 2)",
    )

    parser.add_argument(
        "-dc", "--dft-code",
        type=str,
        choices=['vasp', 'aims'],
        default="aims",
        help="DFT code to use: 'vasp' or 'aims'. Default is aims.",
    )

    parser.add_argument(
        "-xf", "--xc-functional",
        type=str,
        default='hse06',
        help="DFT functional to use (e.g., 'pbe', 'pbeu', 'hse06', 'pbe0'). Default is hse06.",
    )

    parser.add_argument(
        "-hp", "--hubbard-parameters",
        type=str,
        help="Specify Hubbard parameters as a element:orbital:U string (e.g., 'Ti:3d:2.65,Fe:3d:4.0')",
    )

    parser.add_argument(
        "-fsp", "--fix-spin-moment",
        type=float,
        default=None,
        help="Specify fixed_spin_moment for FHI-aims calculation, that allows to enforce fixed overall spin "
             "moment."
    )

    parser.add_argument(
        "-der", "--disable-elsi-restart",
        action="store_true",
        default=False,
        help="If set, elsi_restart and elsi_restart_use_overlap will not be used. Its used is recommended. It"
             "will be disabled when switching to other type of functional "
    )

    parser.add_argument(
        "-a", "--alpha-exchange",
        type=float,
        default=0.25,
        help="Fraction of exact exchange (alpha) for hybrid functionals (HSE06/PBE0). Defaults to 0.25."
    )

    parser.add_argument(
        "-ct", "--calc-type",
        type=str,
        choices=['scf', 'relax-atoms', 'relax-all'],
        default='relax-atoms',
        help="Calculation type: 'scf' (static), 'relax-atoms' (ions only), or 'relax-all' (ions and cell)."
             "Default is relax-atoms. ",
    )

    parser.add_argument(
        "-sm", "--spin-moment",
        type=float,
        default=1.0,
        help="Initial magnetic moment to set on the polaron site(s) for spin seeding. Default is 1.0",
    )

    parser.add_argument(
        "-ssm", "--set-site-magmoms",
        action="store_true",
        default=False,
        help="Set initial magnetic moment on the polaron site(s) for spin seeding. Default is false.",
    )

    parser.add_argument(
        "-ac", "--aims-command",
        type=str,
        default="mpirun -np 8 aims.x",
        help="Full command to execute FHI-aims (used in job bin)."
    )

    parser.add_argument(
        "-sd", "--species-dir",
        type=str,
        help="Directory containing FHI-aims species files."
    )

    parser.add_argument(
        "-rdr", "--run-dir-root",
        type=str,
        default="polaron_runs",
        help="Root directory for output calculations. Default is ./polaron_runs")

    parser.add_argument(
        "-ds", "--do-submit",
        action="store_true",
        default=False,
        help="If set, job scripts are submitted to the cluster immediately (placeholder logic)."
             "Default is false. "
    )

    parser.add_argument(
        "-rp", "--run-pristine",
        action="store_true",
        default=False,
        help="A calculation will run for the pristine structure. This is needed for formation "
             " energy calculations. Default is false.",
    )

    args_parse = parser.parse_args(args)

    log.info("PyPolaron electron attractor CLI Initialized.")

    log.info("PyPolaron run generator CLI Initialized.")

    if "log" in args_parse:
        log_file_name = args_parse.log
        log.info("Redirecting log into file {}".format(log_file_name))
        log_file_name_path = Path(args_parse.run_dir_root) / log_file_name
        file_handler = logging.FileHandler(log_file_name_path, 'a')
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
                 f" correspond(s) to {number_of_oxygen_vacancies * 2} electron polarons")

    if polaron_type == 'electron':
        log.info(f"The calculations will run with {number_of_polarons} additional {polaron_type} polaron(s). "
                 f"In total {number_of_polarons + number_of_oxygen_vacancies * 2} {polaron_type} polarons ")
    elif polaron_type == 'hole':
        log.info(f"The calculations will run with {number_of_polarons} additional {polaron_type} polaron(s).")

    if args_parse.xc_functional.lower() == 'pbeu' and polaron_type == 'hole':
        log.warning("PBE+U not possible with hole polarons. Occupation matrix control works only for electron"
                    "polarons. Please switch to hybrid functionals. Exiting ")
        return

    # 3. Initialize Generator and Run Analysis
    polaron_generator = PolaronGenerator(structure, polaron_type=polaron_type)
    polaron_generator.assign_oxidation_states()  # Ensure oxidation states are set once

    dft_parameters = {
        "dft_code": args_parse.dft_code, "functional": args_parse.xc_functional,
        "calc_type": args_parse.calc_type, "supercell": args_parse.supercell_dims,
        "aims_command": args_parse.aims_command, "species_dir": args_parse.species_dir,
        "run_dir_root": args_parse.run_dir_root, "do_submit": args_parse.do_submit,
        "set_site_magmoms": args_parse.set_site_magmoms, "spin_moment": args_parse.spin_moment,
        "run_pristine": args_parse.run_pristine, "alpha": args_parse.alpha_exchange,
        "hubbard_parameters": args_parse.hubbard_parameters, "fix_spin_moment": args_parse.fix_spin_moment,
        "disable_elsi_restart": args_parse.disable_elsi_restart,
    }

    polaron_candidates = []
    new_polaron_candidates = []
    oxygen_vacancies_candidates = []

    if polaron_type in ["electron", "hole"] and number_of_polarons > 0:
        polaron_candidates = polaron_generator.propose_sites(max_sites=number_of_polarons)

        log.info(f"Found {len(polaron_candidates)} unique {polaron_type} polaron candidate(s):")
        display_candidates(polaron_candidates, polaron_type)

    if number_of_oxygen_vacancies > 0:
        oxygen_vacancies_candidates = polaron_generator.propose_vacancy_sites(max_sites=number_of_oxygen_vacancies)

        log.info(f"Found {len(oxygen_vacancies_candidates)} unique oxygen vacancy candidate(s):")
        display_candidates(oxygen_vacancies_candidates, 'oxygen vacancy')

        if oxygen_vacancies_candidates:
            electron_polaron_candidates_from_oxygen_vacancies = polaron_generator.propose_sites(
                max_sites=number_of_oxygen_vacancies * 2 + number_of_polarons)

            if polaron_candidates:
                new_polaron_candidates = [
                    candidate for candidate in electron_polaron_candidates_from_oxygen_vacancies
                    if candidate not in polaron_candidates
                ]
            else:
                new_polaron_candidates = electron_polaron_candidates_from_oxygen_vacancies[
                                         :number_of_oxygen_vacancies * 2 + 1
                                         ]

            log.info(f"Found {len(new_polaron_candidates)} unique "
                     f"electron polaron candidate(s) from oxygen vacancy generation:")
            display_candidates(new_polaron_candidates,'electron')

    polaron_candidates.extend(new_polaron_candidates)

    log.info("Analysis of possible polaron and oxygen vacancies completed. Ready for DFT Input Generation")

    chosen_polaron_sites = [value[0] for value in polaron_candidates]
    chosen_oxygen_vacancy_sites = [value[0] for value in oxygen_vacancies_candidates]

    # Initialize Workflow
    workflow = PolaronWorkflow(aims_executable_command=dft_parameters["aims_command"])

    dft_parameters_for_workflow = {
        key: value
        for key, value in dft_parameters.items()
        if key not in ["aims_command"]
    }
    settings = DftSettings(**dft_parameters_for_workflow)

    # Run the generation
    workflow.run_attractor_workflow(
        generator=polaron_generator,
        chosen_site_indices=chosen_polaron_sites,
        chosen_vacancy_site_indices=chosen_oxygen_vacancy_sites,
        attractor_elements=args_parse.attractor_elements,
        settings=settings,
    )

    log.info(f"Input files written to folder {dft_parameters['run_dir_root']}")


if __name__ == "__main__":
    main(sys.argv[1:])