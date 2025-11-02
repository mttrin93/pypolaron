import sys
from pypolaron.cli.common_utils import build_common_parser, setup_cli_logging, load_structure, \
    validate_dft_input, map_args_to_dft_params, log, process_and_generate_candidates, \
    run_attractor_workflow, map_args_to_policy
from dataclasses import replace


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # 1) build parser using common arguments
    parser = build_common_parser(
        prog_name="pypolaron-run-attractor",
        description="Toolkit for automated DFT polaron calculations."
    )

    parser.add_argument(
        "-ae", "--attractor-elements",
        type=str,
        required=True,
        help="Element symbol used to substitute the host atom to create the localized potential well."
             "Provide either a single element symbol or a list of symbols. "
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

    new_attractor_value = args_parse.attractor_elements
    dft_params_attractor = replace(
        dft_parameters,
        attractor_elements=new_attractor_value
    )

    # 4) candidates generation
    polaron_candidates, oxygen_vacancies_candidates, polaron_generator = process_and_generate_candidates(
        structure=structure,
        polaron_type=polaron_type,
        number_of_polarons=number_of_polarons,
        number_of_oxygen_vacancies=number_of_oxygen_vacancies,
    )

    # 5) run workflow
    report = run_attractor_workflow(
        polaron_generator=polaron_generator,
        polaron_candidates=polaron_candidates,
        oxygen_vacancy_candidates=oxygen_vacancies_candidates,
        dft_params=dft_params_attractor,
        policy=workflow_policy,
    )

    # log.info(f"Input files written to folder {dft_parameters.run_dir_root}")
    # if args_parse.do_submit:
    #     log.info("Jobs submitted to cluster")

if __name__ == "__main__":
    main(sys.argv[1:])
