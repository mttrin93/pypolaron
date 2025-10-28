from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Any
import textwrap
import numpy as np
import subprocess
import json

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from pypolaron.polaron_generator import PolaronGenerator
from pypolaron.polaron_analyzer import PolaronAnalyzer
from pypolaron.utils import DftSettings, run_job_and_wait, read_final_geometry


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

    def run_attractor_workflow(
            self,
            generator: PolaronGenerator,
            chosen_site_indices: Union[int, List[int]],
            chosen_vacancy_site_indices: Union[int, List[int]],
            settings: DftSettings,
    ) -> Dict[str, Union[str, float, Dict[str, Any]]]:
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
        elif settings.dft_code == "vasp":
            write_func = generator.write_vasp_input_files
        else:
            raise ValueError(f"Unknown DFT tool: {settings.dft_code}")

        # --- JOB 1: ATTRACTOR RELAXATION (M^0 Substitution) ---
        attractor_dir = root / "01_Attractor_Run"

        write_func(
            site_index=chosen_site_indices,
            vacancy_site_index=chosen_vacancy_site_indices,
            settings=settings,
            outdir=str(attractor_dir),
            is_charged_polaron_run=False,
        )
        script_attractor = self.write_simple_job_script(attractor_dir)

        planned = {
            "attractor_dir": str(attractor_dir),
            "attractor_script": script_attractor,
            "instructions": "Run Job 1 first. Then run Job 2 and 3 sequentially using the geometry from Job 1."
        }

        relaxed_attractor_structure = None

        if settings.do_submit:
            planned["submitted"] = True

            try:
                run_job_and_wait(script_attractor)
            except Exception as e:
                planned["status"] = "Job 1 failed. Error: {e}"
                return {"planned": planned}

            relaxed_attractor_structure = read_final_geometry(settings.dft_code, attractor_dir)
            if relaxed_attractor_structure is None:
                planned["status"] = "Could not read relaxed geometry for Job 1"
                return {"planned": planned}

            # --- JOB 2: FINAL POLARON RUN (Original M^n+ with Spin Seed) ---
            polaron_dir = root / "02_Final_Polaron_Run"

            write_func(
                site_index=chosen_site_indices,  # Seed the spin on the target site
                vacancy_site_index=chosen_vacancy_site_indices,
                settings=settings,
                outdir=str(polaron_dir),
                is_charged_polaron_run=True,  # Charged/Magnetic calculation
            )
            script_polaron = self.write_simple_job_script(polaron_dir)

            planned["polaron_dir"] = str(polaron_dir)
            planned["polaron_script"] = script_polaron

            run_job_and_wait(script_polaron)
            #
            # script_pristine = None
            # pristine_dir = None
            # if settings.run_pristine:
            #     pristine_dir = root / "03_Pristine_Ref"
            #
            #     # This job should ideally also start from the relaxed host lattice.
            #     write_func(
            #         site_index=[],
            #         vacancy_site_index=chosen_vacancy_site_indices,
            #         settings=settings,
            #         outdir=str(pristine_dir),
            #         is_charged_polaron_run=False,
            #     )
            #     script_pristine = self.write_simple_job_script(pristine_dir)
            #
            #     planned["pristine_dir"] = str(pristine_dir)
            #     planned["pristine_script"] = script_pristine
            #
            #     # 3a. Submit Job 3 and wait (Conceptual)
            #     run_job_and_wait(script_pristine)

        #
        # # --- ORCHESTRATION REPORT ---
        #
        # planned = {
        #     "attractor_dir": str(attractor_dir),
        #     "polaron_dir": str(polaron_dir),
        #     "pristine_dir": str(pristine_dir) if pristine_dir else None,
        #     "attractor_script": script_attractor,
        #     "polaron_script": script_polaron,
        #     "pristine_script": script_pristine,
        #     "instructions": "Job 1 (Attractor) must run first. Job 2 (Polaron) must use Job 1's final geometry as input."
        # }

        return {"planned": planned}
