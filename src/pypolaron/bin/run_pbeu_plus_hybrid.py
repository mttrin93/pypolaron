import sys
from pypolaron.cli.common_utils import build_common_parser, setup_cli_logging, load_structure, \
    validate_dft_input, map_args_to_dft_params, log, process_and_generate_candidates, \
    map_args_to_policy, run_sequential_relaxations_workflow


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # 1) build parser using common arguments
    parser = build_common_parser(
        prog_name="pypolaron-run-attractor",
        description="Toolkit for automated DFT polaron calculations."
    )

    args_parse = parser.parse_args(args)

    # 2) setup logging file
    setup_cli_logging(args_parse)

    # 3) load structure from file or MP
    structure = load_structure(args_parse)

    polaron_type = args_parse.polaron_type.lower()
    number_of_polarons = args_parse.polaron_number
    number_of_oxygen_vacancies = args_parse.oxygen_vacancy_number

    if not validate_dft_input(args_parse):
        return

    dft_parameters = map_args_to_dft_params(args_parse)
    workflow_policy = map_args_to_policy(args_parse)

    if dft_parameters.calc_type not in ["relax-atoms", "relax-all"]:
        log.warning("The pbeu_plus_hybrid method works only with either relax-atoms or relax-all. Exiting")

    if dft_parameters.functional not in ["hse06", "pbe0"]:
        log.warning("The selected functional of the pbeu_plus_hybrid run must be hybrid. Exiting")

    # 4) candidates generation
    polaron_candidates, oxygen_vacancies_candidates, polaron_generator = process_and_generate_candidates(
        structure=structure,
        polaron_type=polaron_type,
        number_of_polarons=number_of_polarons,
        number_of_oxygen_vacancies=number_of_oxygen_vacancies,
    )

    # 5) run workflow
    report = run_sequential_relaxations_workflow(
        polaron_generator=polaron_generator,
        polaron_candidates=polaron_candidates,
        oxygen_vacancy_candidates=oxygen_vacancies_candidates,
        dft_params=dft_parameters,
        policy=workflow_policy,
        relaxation_method="pbeu_plus_hybrid",
    )

    # log.info(f"Input files written to folder {dft_parameters.run_dir_root}")
    # if args_parse.do_submit:
    #     log.info("Jobs submitted to cluster")

if __name__ == "__main__":
    main(sys.argv[1:])
