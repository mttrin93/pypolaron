from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Any, Callable
import textwrap
import numpy as np
import subprocess
import json
import shutil
from dataclasses import replace

from pymatgen.core import Structure

import logging
import socket
import getpass

from pypolaron.polaron_generator import PolaronGenerator
from pypolaron.polaron_analyzer import PolaronAnalyzer
from pypolaron.utils import DftSettings, run_job_and_wait, read_final_geometry, is_job_completed


# --- Global Setup ---
hostname = socket.gethostname()
username = getpass.getuser()

LOG_FMT = '%(asctime)s %(levelname).1s - %(message)s'.format(hostname)
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger('pypolaron')


# TODO: When seeding multiple polarons, be careful about total charge and spin multiplicity:
#  two electrons can pair up (singlet) or remain as separate spins — seed spin moments with the sign you want (e.g., both +1.0 for triplet-like initial guess). Check the final spin in aims.out.
# TODO: Always check supercell convergence (formation energy vs cell size). The FNV/Makov–Payne corrections are approximations — don't rely just on a single cell.
# TODO: Use actual Bader (Henkelman) for accurate charges. The sphere-integration fallback is handy for quick checks but not production-quality.

# TODO: next implementations:
#  Goal: Extract physical insights: polaron formation energies, interaction, and thermodynamic properties.
#  Algorithmic steps:
#  Polaron formation energy tables for single/multiple polarons.
#  Interaction energies for multi-polaron systems:
#  Thermodynamic modeling:
#  Boltzmann statistics for occupancy at finite T.
#  Free energies for hopping pathways.
#  Visualizations:
#  Charge density isosurfaces, polaron–polaron distance maps, formation energy histograms.

# TODO: have a look at vibes and high-throughput ams-tools for examples how to deal with submission queues in AIMS


class PolaronWorkflow:
    def __init__(
        self,
        aims_executable_command: str,
        epsilon: Optional[float] = None,
        fermi_energy: float = 0.0,
        volume_ang3: Optional[float] = None,
    ):
        self.aims_executable_command = aims_executable_command
        self.epsilon = epsilon
        self.fermi_energy = fermi_energy
        self.volume_ang3 = volume_ang3

    def write_simple_job_script(
        self,
        workdir: Path,
        ntasks: int = 72,
        walltime: str = "02:00:00",
        scheduler: str = "slurm",
    ) -> Path:
        """
        Write a simple shell script to run FHI-aims in 'workdir'.
        aims_command: e.g. 'mpirun -np 64 /path/to/aims > aims.out'
        scheduler: optional, if 'slurm' use sbatch header; otherwise plain shell script.
        Returns path to script.
        """
        # workdir_p = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        script_path = workdir / "run_aims.sh"
        if scheduler == "slurm":
            content = textwrap.dedent(
                f"""\
                #!/bin/bash -l
                #SBATCH -o ./job.out.%j
                #SBATCH -e ./job.err.%j
                #SBATCH -D ./
                #SBATCH -J pypolaron
                #SBATCH --nodes=1
                #SBATCH --ntasks-per-node={ntasks}
                #SBATCH --time={walltime}\n
                \n
                \n
                module purge
                module load intel/19.1.2  mkl/2020.2 impi/2019.8  gcc/10  cmake/3.26 elpa/mpi/openmp/2021.05.001 hdf5-serial/1.8.22\n
                \n
                export OMP_NUM_THREADS=1
                export MKL_DYNAMIC=FALSE
                export MKL_NUM_THREADS=1
                \n
                cd {workdir} 
                export LD_LIBRARY_PATH=/mpcdf/soft/SLE_15/packages/x86_64/intel_parallel_studio/2020.2/compilers_and_libraries/linux/lib/intel64_lin:$LD_LIBRARY_PATH
                \n
                {self.aims_executable_command} > aims.out
            """
            )
        else:
            content = textwrap.dedent(
                f"""\
                 #!/bin/bash
                 cd {workdir}
                 {self.aims_executable_command} > aims.out 2>&1  
            """
            )
        script_path.write_text(content)
        script_path.chmod(0o755)
        return script_path

    def run_polaron_workflow(
        self,
        generator: PolaronGenerator,
        chosen_site_indices: Union[int, List[int]],
        chosen_vacancy_site_indices: Union[int, List[int]],
        settings: DftSettings,
    ) -> Dict[str, Union[str, float, Dict[str, Any]]]:
        """
        High-level orchestrator for polaron calculations + optional analysis.
        auto_analyze: if True, calls PolaronAnalyzer if outputs exist

        A first run should be performed with do_submit=True to produce aims.out and cube files
        followed by do_submit=False to analyze the outputs
        """
        root = Path(settings.run_dir_root)
        root.mkdir(parents=True, exist_ok=True)

        # Directories
        pristine_dir = root / "pristine" if settings.run_pristine else None
        charged_dir = root / "charged"

        script_pristine = None

        if settings.dft_code == "aims":
            write_func = generator.write_fhi_aims_input_files
        elif settings.dft_code == "vasp":
            write_func = generator.write_vasp_input_files
        else:
            raise ValueError(f"Unknown DFT tool: {settings.dft_code}. Must be 'aims' or 'vasp'.")

        if settings.run_pristine:
            write_func(
                site_index=[],
                vacancy_site_index=chosen_vacancy_site_indices,
                settings=settings,
                outdir=str(pristine_dir),
                is_charged_polaron_run=False,
            )
            script_pristine = self.write_simple_job_script(pristine_dir)

        write_func(
            site_index=chosen_site_indices,
            vacancy_site_index=chosen_vacancy_site_indices,
            settings=settings,
            outdir=str(charged_dir),
            is_charged_polaron_run=True,
        )

        # 2) Write job scripts
        pristine_dir_str = str(script_pristine) if settings.run_pristine else None
        script_charged = self.write_simple_job_script(charged_dir)

        planned = {
            "pristine_dir": pristine_dir_str,
            "charged_dir": str(charged_dir),
            "pristine_script": script_pristine,
            "charged_script": script_charged,
            "instructions": "Run the scripts in the respective directories to perform calculations.",
        }

        # 3) Optionally submit
        if settings.do_submit:
            if settings.run_pristine:
                subprocess.run(["bash", script_pristine], check=True)
            subprocess.run(["bash", script_charged], check=True)
            planned["submitted"] = True
        else:
            planned["submitted"] = False

        # Initialize report
        report = {"planned": planned}

        # 4) Optional post-processing
        if settings.auto_analyze:
            try:
                analyzer = PolaronAnalyzer(
                    fermi_energy=self.fermi_energy,
                    epsilon=self.epsilon,
                    volume_ang3=self.volume_ang3,
                )

                # Automatically find outputs
                # TODO: add here reading of spin cube files
                pristine_out = pristine_dir / "aims.out"
                charged_out = charged_dir / "aims.out"
                potential_cube_pristine = pristine_dir / "potential.cube"
                potential_cube_charged = charged_dir / "potential.cube"

                # TODO: here we calculate polaron formation energies and corrections
                if pristine_out.exists() and charged_out.exists():
                    results = analyzer.analyze_polaron_run(
                        pristine_out=str(pristine_out),
                        charged_out=str(charged_out),
                        atom_coords=np.array(
                            [0, 0, 0]
                        ),  # placeholder: pass real defect coords
                        site_index_supercell=chosen_site_indices[0],
                        total_charge=settings.add_charge,
                        pot_cube_pristine=(
                            str(potential_cube_pristine)
                            if potential_cube_pristine.exists()
                            else None
                        ),
                        pot_cube_charged=(
                            str(potential_cube_charged)
                            if potential_cube_charged.exists()
                            else None
                        ),
                    )
                    report["analysis"] = results
                else:
                    report["analysis_status"] = (
                        "Aims outputs not found; run calculations first."
                    )
            except Exception as e:
                report["analysis_error"] = str(e)

        # 5) Save report
        report_path = root / "polaron_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        return report

    def _execute_job_with_rerun(
            self,
            planned: Dict[str, Union[str, float, bool]],
            input_script: Path,
            input_dir: Path,
            geometry_filename: str,
            geometry_next_step_filename: str,
    ):
        MAX_RETRIES = 3
        planned["submitted"] = True
        current_retries = 0
        job_status = "PENDING"

        while current_retries < MAX_RETRIES:
            if current_retries > 0:
                log.info(f"Retrying job after TIMEOUT (Attempt {current_retries + 1}/{MAX_RETRIES}).")

            try:
                job_status = run_job_and_wait(input_script)
            except RuntimeError as e:
                planned["status"] = f"Job failed on attempt {current_retries + 1}. Fatal error: {e}"
                return planned
                # return {"planned": planned}

            if job_status == "COMPLETED":
                log.info(f"Job completed successfully after {current_retries} retries.")
                break
            elif job_status == "TIMEOUT":
                next_step_path = input_dir / geometry_next_step_filename
                geometry_path = input_dir / geometry_filename

                if next_step_path.exists():
                    log.warning(
                        f"TIMEOUT detected. Copying {geometry_next_step_filename} to "
                        f" {geometry_filename} for restart.")
                    shutil.copy2(next_step_path, geometry_path)
                    current_retries += 1
                else:
                    planned["status"] = (f"Job timed out but could not find "
                                         f" {geometry_next_step_filename} to restart.")
                    return planned
            else:
                planned["status"] = f"Job failed with unexpected status: {job_status}"
                return planned

        if job_status != "COMPLETED":
            planned["status"] = f"Job failed after {MAX_RETRIES} retries. Halting workflow."
            return planned

        return planned

    def remove_attractor_elements(
            self,
            generator: PolaronGenerator,
            chosen_site_indices: Union[int, List[int]],
            settings: DftSettings,
            relaxed_attractor_structure: Structure,
    ) -> Structure:
        """
        Move original atoms back to previous attractor locations
        """
        if isinstance(settings.attractor_elements, str):
            elements_list = [e.strip().capitalize() for e in settings.attractor_elements.split(',') if e.strip()]
        else:
            elements_list = [e.strip().capitalize() for e in settings.attractor_elements]

        if len(elements_list) == 1 and len(chosen_site_indices) > 1:
            elements_list = elements_list * len(chosen_site_indices)

        indices_attractor = [index for index, site in enumerate(relaxed_attractor_structure) if
                             site.specie.symbol in elements_list]

        chosen_sites_species = [generator.structure[index].specie for index in chosen_site_indices]
        indices_original_elements = [chg_spec.element for chg_spec in chosen_sites_species]
        final_polaron_structure = relaxed_attractor_structure.copy()

        for index_attractor, index_original in zip(indices_attractor, indices_original_elements):
            final_polaron_structure.replace(index_attractor, index_original)

        return final_polaron_structure

    def _run_and_check_job(
            self,
            job_name: str,
            job_dir: Path,
            write_func: Callable,
            settings: DftSettings,
            chosen_site_indices: Union[int, List[int]],
            chosen_vacancy_site_indices: Union[int, List[int]],
            is_charged_polaron_run: bool,
            geometry_filenames: Tuple[str, str],  # (geometry_filename, geometry_next_step_filename)
            base_structure: Optional[Structure] = None,
    ) -> Dict[str, Union[str, float, bool]]:
        """Runs a single job, handles file writing, execution, and status check."""

        # job_dir.mkdir(parents=True, exist_ok=True)
        geometry_filename, geometry_next_step_filename = geometry_filenames

        # 1. Write input files
        write_func(
            site_index=chosen_site_indices,
            vacancy_site_index=chosen_vacancy_site_indices,
            settings=settings,
            outdir=str(job_dir),
            is_charged_polaron_run=is_charged_polaron_run,
            base_structure=base_structure,
        )
        script_path = self.write_simple_job_script(job_dir)

        planned = {
            f"{job_name.lower()}_dir": str(job_dir),
            f"{job_name.lower()}_script": str(script_path),
        }

        # 2. Run and check status
        if settings.do_submit:
            planned = self._execute_job_with_rerun(
                planned=planned,
                input_script=script_path,
                input_dir=job_dir,
                geometry_filename=geometry_filename,
                geometry_next_step_filename=geometry_next_step_filename,
            )
        else:
            planned["submitted"] = False
            planned["status"] = f"{job_name} files written, ready to run."

        return planned

    def run_attractor_workflow(
            self,
            generator: PolaronGenerator,
            chosen_site_indices: Union[int, List[int]],
            chosen_vacancy_site_indices: Union[int, List[int]],
            settings: DftSettings,
    ) -> Dict[str, Union[str, float, bool, Path, Dict[str, Any]]]:
        """
        Manages the sequential file generation and optional execution for the Electron Attractor method.

        This method requires three jobs:
        1. Attractor Run -> Relaxed Geometry (must run first)
        2. Polaron Run (original element substituted back), starting from Job 1 geometry
        3. Pristine Reference (original structure with no polaron, fully relaxed)
        """

        root = Path(settings.run_dir_root)
        root.mkdir(parents=True, exist_ok=True)

        if settings.dft_code == "aims":
            write_func = generator.write_fhi_aims_input_files
            geometry_filenames = ("geometry.in", "geometry.in.next_step")
        elif settings.dft_code == "vasp":
            write_func = generator.write_vasp_input_files
            geometry_filenames = ("POSCAR", "CONTCAR")
        else:
            raise ValueError(f"Unknown DFT tool: {settings.dft_code}")

        planned = {}

        if settings.run_pristine:
            pristine_dir = root / "Pristine_Ref"

            result_pristine = self._run_and_check_job(
                job_name="pristine",
                job_dir=pristine_dir,
                write_func=write_func,
                settings=settings,
                chosen_site_indices=[],
                chosen_vacancy_site_indices=chosen_vacancy_site_indices,
                is_charged_polaron_run=False,
                geometry_filenames=geometry_filenames,
            )

            if "status" in result_pristine and result_pristine["submitted"] == True:
                return {"planned": result_pristine}

            planned.update(result_pristine)

        attractor_dir = root / "01_Attractor_Run"

        if not is_job_completed(settings.dft_code, attractor_dir):
            result_job1 = self._run_and_check_job(
                job_name="attractor",
                job_dir=attractor_dir,
                write_func=write_func,
                settings=settings,
                chosen_site_indices=chosen_site_indices,
                chosen_vacancy_site_indices=chosen_vacancy_site_indices,
                is_charged_polaron_run=False,
                geometry_filenames=geometry_filenames,
            )

            if "status" in result_job1 and result_job1["submitted"] == True:
                return {"planned": result_job1}

            planned.update(result_job1)

        relaxed_attractor_structure = read_final_geometry(settings.dft_code, attractor_dir)
        if relaxed_attractor_structure is None:
            planned["status"] = "Could not read relaxed geometry for Job 1"
            return {"planned": planned}

        final_polaron_structure = self.remove_attractor_elements(
            generator=generator,
            chosen_site_indices=chosen_site_indices,
            settings=settings,
            relaxed_attractor_structure=relaxed_attractor_structure
        )

        polaron_dir = root / "02_Final_Polaron_Run"
        settings_job2 = replace(settings, attractor_elements=None)

        result_job2 = self._run_and_check_job(
            job_name="polaron",
            job_dir=polaron_dir,
            write_func=write_func,
            settings=settings_job2,
            chosen_site_indices=chosen_site_indices,
            chosen_vacancy_site_indices=chosen_vacancy_site_indices,
            is_charged_polaron_run=True,
            geometry_filenames=geometry_filenames,
            base_structure=final_polaron_structure,
        )

        if "status" in result_job2:
            return {"planned": result_job2}

        planned.update(result_job2)

        return {"planned": planned}
