import argparse
import logging
import os
from typing import List, Tuple, Union, Any, Optional, Dict, Literal
from pathlib import Path
import socket
import yaml
import getpass
from dataclasses import asdict

from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pypolaron import __version__
from pyfhiaims.geometry import AimsGeometry
from pypolaron.utils import DftSettings, setup_pypolaron_logger, WorkflowPolicy
from pypolaron.workflow import PolaronWorkflow
from pypolaron.polaron_generator import PolaronGenerator

hostname = socket.gethostname()
username = getpass.getuser()
log = setup_pypolaron_logger()

def setup_cli_logging(args_parse: argparse.Namespace):
    """Sets up file logging based on CLI arguments."""
    log_file_name = args_parse.log
    log.info(f"Redirecting log into file {log_file_name}")
    log_file_name_path = Path(args_parse.run_dir_root) / log_file_name

    # Ensure root directory exists before setting up file handler
    Path(args_parse.run_dir_root).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file_name_path, 'a')
    log_formatter = f'%(asctime)s %(levelname).1s - ({hostname}) - %(message)s'
    formatter = logging.Formatter(log_formatter)
    file_handler.setFormatter(formatter)

    # Only add the handler if it's not already there (prevents duplication on multiple runs)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        log.addHandler(file_handler)

    log.info("PyPolaron CLI Initialized.")
    log.info(f"Hostname: {hostname}")
    log.info(f"Username: {username}")
    log.info(f"pypolaron version: {__version__}")

def build_common_parser(prog_name: str, description: str) -> argparse.ArgumentParser:
    """Creates the base ArgumentParser instance with all shared arguments."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=description + f"\nVersion: {__version__}",
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
        "-scf", "--setup-config-file",
        type=str,
        required=True,
        help="Path to the YAML file containing all DFT and defect generation settings.",
    )

    parser.add_argument(
        "-pf", "--policy-file",
        type=str,
        default=None,
        help="Path to a YAML file containing workflow execution policy settings."
    )

    return parser


def load_structure(args_parse: argparse.Namespace) -> Optional[Structure]:
    """Loads structure from file or Materials Project."""
    structure: Optional[Structure] = None
    if args_parse.file:
        structure_file_path = args_parse.file
        log.info(f"Try to read initial structure from the filepath: {os.path.abspath(structure_file_path)}")
        if not os.path.exists(structure_file_path):
            log.warning(f"ERROR: File not found at {structure_file_path}")
            return None

        path = Path(structure_file_path)
        is_aims_file = path.name.lower() in ["geometry.in"] or path.suffix.lower() in [".in"]

        try:
            if is_aims_file:
                structure = AimsGeometry.from_file(path).structure
                log.info(f"File loaded with AIMS parser: {path.name}")
            else:
                structure = Structure.from_file(structure_file_path)
                log.info(f"File loaded with universal pymatgen parser: {path.name}")
        except Exception as e:
            log.warning(f"ERROR reading file {structure_file_path}: {e}")
            return None

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

    return structure

def validate_dft_input(arg_parse : argparse.Namespace, dft_parameters: DftSettings) -> bool:
    """Performs final checks on polaron/vacancy and functional choices."""
    polaron_type = arg_parse.polaron_type.lower()
    number_of_polarons = arg_parse.polaron_number
    number_of_oxygen_vacancies = arg_parse.oxygen_vacancy_number

    if polaron_type not in ["electron", "hole"]:
        log.warning(f"Invalid polaron type choice: {polaron_type}. Exiting.")
        return False

    if number_of_polarons == 0 and number_of_oxygen_vacancies == 0:
        log.warning(f"No polarons or oxygen vacancies will be created. Exiting. ")
        return False

    if dft_parameters.functional.lower() == 'pbeu' and polaron_type == 'hole':
        log.warning("PBE+U not possible with hole polarons. Occupation matrix control works only for electron "
                    "polarons. Please switch to hybrid functionals. Exiting ")
        return False

    return True

def map_args_to_dft_params(args_parse: argparse.Namespace) -> DftSettings:
    """
    Loads DftSettings from YAML and applies final CLI overrides
    for run directory and submission.
    """

    dft_settings_dict = load_dft_settings_from_yaml(args_parse.setup_config_file)

    if args_parse.run_dir_root:
        dft_settings_dict['run_dir_root'] = args_parse.run_dir_root

    if args_parse.do_submit:
        dft_settings_dict['do_submit'] = args_parse.do_submit

    if args_parse.run_pristine:
        dft_settings_dict['run_pristine'] = args_parse.run_pristine

    return DftSettings(**dft_settings_dict)

def load_workflow_policy_from_yaml(
        policy_file: Union[str, Path]
) -> WorkflowPolicy:
    """
    Loads execution policy settings from a YAML file, merging them with defaults.
    """
    policy_dict = asdict(WorkflowPolicy())

    policy_path = Path(policy_file)
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy_file}")

    try:
        with open(policy_path, 'r') as f:
            user_settings = yaml.safe_load(f)

        if user_settings:
            for key, value in user_settings.items():
                if key in policy_dict:
                    policy_dict[key] = value

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML policy file {policy_file}: {e}")

    return WorkflowPolicy(**policy_dict)

def load_dft_settings_from_yaml(
        dft_settings_file_path: Union[str, Path]
) -> Dict[str, Any]:
    """Loads DFT settings from YAML, checking against known DftSettings fields."""
    dft_settings_dict = asdict(DftSettings())

    yaml_path = Path(dft_settings_file_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"DFT config file not found: {dft_settings_file_path}")

    try:
        with open(yaml_path, 'r') as f:
            user_settings = yaml.safe_load(f)

        if user_settings:
            for key, value in user_settings.items():
                if key in dft_settings_dict:
                    # Handle tuple conversion for supercell if necessary
                    if key == 'supercell' and isinstance(value, list):
                        dft_settings_dict[key] = tuple(value)
                    else:
                        dft_settings_dict[key] = value
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML policy file {dft_settings_file_path}: {e}")

    return dft_settings_dict

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

def process_and_generate_candidates(
        structure: Structure,
        polaron_type: str,
        number_of_polarons: int,
        number_of_oxygen_vacancies: int
) -> Tuple[List, List, PolaronGenerator]:
    """
    Initializes the generator, finds polaron and vacancy candidates,
    and handles the logic for vacancy-induced electron polarons.
    Logs all results.

    Returns:
        (polaron_candidates, oxygen_vacancies_candidates, polaron_generator_instance)
    """

    if number_of_oxygen_vacancies > 0:
        log.info(f"{number_of_oxygen_vacancies} oxygen vacancy(ies) will be considered, that"
                 f" correspond(s) to {number_of_oxygen_vacancies * 2} electron polarons")

    if polaron_type == 'electron':
        log.info(f"The calculations will run with {number_of_polarons} additional {polaron_type} polaron(s). "
                 f"In total {number_of_polarons + number_of_oxygen_vacancies * 2} {polaron_type} polarons ")
    elif polaron_type == 'hole':
        log.info(f"The calculations will run with {number_of_polarons} additional {polaron_type} polaron(s).")

    polaron_generator = PolaronGenerator(structure, polaron_type=polaron_type)
    polaron_generator.assign_oxidation_states()

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

    return polaron_candidates, oxygen_vacancies_candidates, polaron_generator

def run_polaron_workflow(
        polaron_generator: PolaronGenerator,
        polaron_candidates: List[Tuple[int, str, Optional[float], float, float]],
        oxygen_vacancy_candidates: List[Tuple[int, str, float]],
        dft_params: DftSettings,
        policy: WorkflowPolicy,
):
    """
    Asks the user to select a candidate index to calculate and triggers file generation.
    """
    log.info("DFT calculation setup initialization and job submission:")
    log.info(
        f"DFT Code: {dft_params.dft_code.upper()} | Functional: {dft_params.functional} "
        f" | Calculation Type: {dft_params.calc_type}")

    # Format: (index, element, oxidation_state, coordination_number, score)
    chosen_polaron_sites = [value[0] for value in polaron_candidates]
    chosen_oxygen_vacancy_sites = [value[0] for value in oxygen_vacancy_candidates]

    # Initialize Workflow
    workflow = PolaronWorkflow(
        generator=polaron_generator,
        dft_code=dft_params.dft_code,
        policy=policy,
        log=log,
    )

    # Run the generation
    report = workflow.run_polaron_workflow(
        chosen_site_indices=chosen_polaron_sites,
        chosen_vacancy_site_indices=chosen_oxygen_vacancy_sites,
        settings=dft_params,
    )

    return report

def run_sequential_relaxations_workflow(
        polaron_generator: PolaronGenerator,
        polaron_candidates: List[Tuple[int, str, Optional[float], float, float]],
        oxygen_vacancy_candidates: List[Tuple[int, str, float]],
        dft_params: DftSettings,
        policy: WorkflowPolicy,
        relaxation_method: Literal["attractor", "pbeu_plus_hybrid"],
):
    """
    Asks the user to select a candidate index to calculate and triggers file generation.
    """
    log.info("DFT calculation setup initialization and job submission:")
    log.info(
        f"DFT Code: {dft_params.dft_code.upper()} | Functional: {dft_params.functional} "
        f" | Calculation Type: {dft_params.calc_type}")

    # Format: (index, element, oxidation_state, coordination_number, score)
    chosen_polaron_sites = [value[0] for value in polaron_candidates]
    chosen_oxygen_vacancy_sites = [value[0] for value in oxygen_vacancy_candidates]

    # Initialize Workflow
    workflow = PolaronWorkflow(
        generator=polaron_generator,
        dft_code=dft_params.dft_code,
        policy=policy,
        log=log,
    )

    # Run the generation
    report = workflow.sequential_relaxations_workflow(
        chosen_site_indices=chosen_polaron_sites,
        chosen_vacancy_site_indices=chosen_oxygen_vacancy_sites,
        settings=dft_params,
        method=relaxation_method,
    )

    return report
