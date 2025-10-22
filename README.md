# PYPOLARON

`pypolaron` is a Python software for generation, submission, pre- and post-processing and analysis of **polaron calculations**.
This repository contains scripts for automated **polaron calculations** with FHI-AIMS and VASP.
 
## Overview 

The high-throughput `pypolaron` framework allows to submit DFT calculations to slurm clusters starting from an initial structure
or composition. 

## Installation

Run:

`pip install -e .`

to install the package via `pip` in editable mode. 

Run the command:

`pip install -e .[dev]`

to install the package and tests used for development.

## Installation

1. Create a `conda` environment with the command:

`conda create -n ace python=3.9`

2. Download the `pyace` repository:

```bash
git clone https://github.com/ICAMS/python-ace.git
cd python-ace
```

3. Run installation script:

`pip install --upgrade .`

4. Install compatible version of TensorFlow:

`pip install tensorflow==2.8.0` 

5. Download the `ace_plus_q` repository and run the installation script:

`pip install --upgrade .`

Now, `ace_plus_q` should be available from the terminal, if corresponding conda environment is loaded.


## Running workflow

To run fit, it is required to provide:

1. Training dataset. How to create a dataset in a suitable format can be found here:
   https://pacemaker.readthedocs.io/en/latest/pacemaker/quickstart/#optional_manual_fitting_dataset_preparation . 
The only additional information required is the total charge. It should be added in the `total_charge` column of the DataFrame.
Optionally, one can include  the set of atomic charges and/or the total dipole moment adding the `atomic_charges` 
and `total_dipole` columns to the DataFrame. A non-periodic or periodic fit will be performed based on the periodicity
of the structures contained in dataset.

2. Input file in the `.yaml` format. Example of the file could be found in the `examples` folder.
Parameters found in this file are described below.

To start a fit simply run `ace_plus_q input.yaml [-p potential.yaml] [-o output]`.

`python scripts/run_polaron_generator.py -mq MgO -mak ID -ov 1 -pn 1 -pt electron -dc aims -rdr examples/run_polaron_gen/`