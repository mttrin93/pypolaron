import sys
from pypolaron.cli.common_utils import build_common_parser, setup_cli_logging, load_structure, \
    validate_dft_input, map_args_to_dft_params, log, process_and_generate_candidates, run_polaron_workflow, \
    load_workflow_policy_from_yaml


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # 1) build parser using common arguments
    parser = build_common_parser(
        prog_name="pypolaron-run-generator",
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
    workflow_policy = load_workflow_policy_from_yaml(args_parse.policy_file)

    # 4) candidates generation
    polaron_candidates, oxygen_vacancies_candidates, polaron_generator = process_and_generate_candidates(
        structure=structure,
        polaron_type=polaron_type,
        number_of_polarons=number_of_polarons,
        number_of_oxygen_vacancies=number_of_oxygen_vacancies,
    )

    # 5) run workflow
    report = run_polaron_workflow(
        polaron_generator=polaron_generator,
        polaron_candidates=polaron_candidates,
        oxygen_vacancy_candidates=oxygen_vacancies_candidates,
        dft_params=dft_parameters,
        policy=workflow_policy,
    )

    log.info(f"Input files written to folder {dft_parameters.run_dir_root}")
    if args_parse.do_submit:
        log.info("Jobs submitted to cluster")

if __name__ == "__main__":
    main(sys.argv[1:])
