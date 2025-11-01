# PYPOLARON

`pypolaron` is a Python software for generation, submission, pre- and post-processing and analysis of **polaron calculations**.
This repository contains scripts for automated **polaron calculations** with FHI-AIMS and VASP.
 
## Overview 

`pypolaron` is a high-throughput framework to automate density-functional-theory (DFT) calculations on SLURM clusters, starting from 
an input structure or composition. The framework focuses on workflows for polaron localisation and migration: it can prepare and 
submit relaxation jobs that encourage polarons to localize on chosen atomic sites, perform post-processing to extract polaron 
formation energies, charges and magnetic moments, and generate nudged-elastic-band (NEB) calculations to estimate migration barriers. 
The produced data can be assembled into datasets suitable for training polaron-aware machine-learning interatomic potentials.

## Key features

All features and functionality are fully-customisable:

* Prepare and submit DFT calculations to SLURM clusters from a single structure or composition input.

* Automated relaxation workflows that help localize polarons on specific atomic sites.

* Post-processing tools to extract:

  * polaron formation energies

  * atomic charges (Bader / Hirshfeld / Mulliken)

  * local magnetic moments 

* Automated NEB generation for polaron migration pathways and barrier estimation.

* Utilities to collect and format results into datasets for ML potential fitting.

* Flexible: works with common DFT packages through configurable templates (VASP, FHI-AIMS).

## Installation

1. Create a `conda` environment with the command:

`conda create -n pypolaron`

2. Download the `pypolaron` repository:

```bash
git clone https://github.com/mttrin93/pypolaron.git
cd pypolaron
```

3. Run installation script:

`pip install .`

to install the package via `pip`. Alternatively, run the command:

`pip install .[dev]`

to install the package and tests used for development. 
Now, the `run_generator` and `run_attractor` scripts can be used to run the polaron workflows.

## Literature

The following literature contains useful discussions of various aspects of polaron calculations:

[1] [Polarons free from many-body self-interaction in density functional theory](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.125119)

[2] [Finite-size corrections of defect energy levels involving ionic polarization](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.041115)

[3] [Efficient Method for Modeling Polarons Using Electronic Structure Methods](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00374)

[4] [Polarons in materials](https://www.nature.com/articles/s41578-021-00289-w)

## Running workflow (CLI executables)

The `run_generator` executable can be run with the following options:

* `-f, --file`: path to a structure file (POSCAR, CIF, geometry.in, structure.xyz). Use 
this to run workflows starting from a local structure.

* `-mq, --mp-query`: query the Materials Project by ID (e.g. `mp-2657`) or by composition 
(e.g. TiO2).

* `-mak, --mp-api-key`: Materials Project API Key. Defaults to the MP_API_KEY environment variable

* `-dc, --dft-code`: choose the DFT backend (`vasp` or `aims`).

* `-xf, --xc-functional`: DFT functional to use (e.g., 'pbe', 'pbeu', 'hse06', 'pbe0'). Default 
is hse06.

* `-hp, --hubbard-parameters`: Specify Hubbard parameters as a element:orbital:U string 
(e.g., 'Ti:3d:2.65,Fe:3d:4.0')

* `-fsp, --fix-spin-moment`: Specify fixed_spin_moment for FHI-aims calculation, that allows to 
enforce fixed overall spin moment.

*  `-der, --disable-elsi-restart`: If set, elsi_restart and elsi_restart_use_overlap will not 
be used. Its used is recommended.  
  
* `-a, --alpha ALPHA`: Fraction of exact exchange (alpha) for hybrid functionals (HSE06/PBE0). 
Defaults to 0.25.

* `-ct, --calc-type`: calculation type: `scf`, `relax-atoms`, or `relax-all`.

* `-pt, --polaron-type`: electron or hole. Default is electron.

* `-pn, --polaron-number`: number of extra polarons to add (default: 1).

* `-ovn, --oxygen-vacancy-number`: number of oxygen vacancies to create.  Default is 0.

* `-s, --supercell-dims`: three integers for supercell replication, e.g. `-s 2 2 2`. 
Default is (2, 2, 2)

* `-sm, --spin-moment` and `-ssm, --set-site-magmoms`: seed initial magnetic moments on 
the target site(s) to favour localisation.

* `-rdr, --run-dir-root`: root path where workflow run directories are created.

* `-ds, --do-submit`: enable submission of generated job scripts to the scheduler immediately.

* `-ac, --aims-command` and `-sd, --species-dir`: AIMS-specific settings required when 
using `--dft-code aims`.

* `-l, --log`: set a log filename (default `log.txt`).

* `-rp, --run-pristine`: run the pristine (undefected) structure, useful for formation 
energy references.

A more detailed description can be obtained running `run_generator -h`. 
A common example command for relaxing an electron polaron in MgO using `aims`, with the 
structure fetched from Materials Project, would be:

`run_generator -mq MgO -mak ID -ac "mpirun -np 28 /path/to/aims.x" -sd /path/to/species -rdr ./workdir -ds`

where the calculation is automatically submitted to the SLURM cluster.

The `run_attractor` executable can be run with the following additional options:

* `-ae, --attractor-elements`: Element symbol used to substitute the host atom to create the 
localized potential well. 

## Development & testing

Run unit tests with pytest:

`pytest tests/`

Use the included linters and formatters (e.g. `black` and `flake8`) to maintain code quality.

## Contributing

Contributions, suggestions and bug reports are very welcome! Please open issues or pull requests. Consider adding tests and 
updating documentation for non-trivial changes.

## Citation

If you use `pypolaron` in published work, please cite the repository.

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Contact

Maintainer: Matteo Rinaldi — mrinaldi@fhi-berlin.mpg.de