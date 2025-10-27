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
from pypolaron.utils import DftSettings


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
        nthreads: int = 8,
        walltime: str = "02:00:00",
        scheduler: Optional[str] = None,
    ) -> str:
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
                #!/bin/bash
                #SBATCH --job-name=aims_job
                #SBATCH --time={walltime}
                #SBATCH --cpus-per-task={nthreads}
                #SBATCH --output=aims.out\n
                cd {workdir}
                {self.aims_executable_command}
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
        return str(script_path)

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

        # TODO: start debug from here
        # --- JOB 2: FINAL POLARON RUN (Original M^n+ with Spin Seed) ---
        polaron_dir = root / "02_Final_Polaron_Run"

        # This uses the original polaron site and seeds the spin moment.
        # This job MUST be started from the relaxed geometry of Job 1.

        write_func(
            site_index=chosen_site_indices,  # Seed the spin on the target site
            vacancy_site_index=chosen_vacancy_site_indices,
            settings=settings,
            outdir=str(polaron_dir),
            is_charged_polaron_run=True,  # Charged/Magnetic calculation
        )
        script_polaron = self.write_simple_job_script(polaron_dir)

        # --- JOB 3: PRISTINE REFERENCE (Optional) ---

        script_pristine = None
        pristine_dir = None
        if settings.run_pristine:
            pristine_dir = root / "03_Pristine_Ref"

            # This is the standard neutral reference run
            write_func(
                site_index=[],
                vacancy_site_index=chosen_vacancy_site_indices,
                settings=settings,
                outdir=str(pristine_dir),
                is_charged_polaron_run=False,
            )
            script_pristine = self.write_simple_job_script(pristine_dir)

        # --- ORCHESTRATION REPORT ---

        planned = {
            "attractor_dir": str(attractor_dir),
            "polaron_dir": str(polaron_dir),
            "pristine_dir": str(pristine_dir) if pristine_dir else None,
            "attractor_script": script_attractor,
            "polaron_script": script_polaron,
            "pristine_script": script_pristine,
            "instructions": "Job 1 (Attractor) must run first. Job 2 (Polaron) must use Job 1's final geometry as input."
        }

        return {"planned": planned}

#
# def run_attractor_workflow(
#         self,
#         generator: PolaronGenerator,
#         chosen_site_indices: Union[int, List[int]],
#         chosen_vacancy_site_indices: Union[int, List[int]],
#         settings: DftSettings,
# ) -> Dict[str, Union[str, float, Dict[str, Any]]]:
#     """
#     Manages the sequential file generation and optional execution for the Electron Attractor method.
#
#     Requires three jobs in sequence:
#     1. Attractor Run (Substitution) -> Relaxed Geometry
#     2. Polaron Run (Spin Seed) starting from Job 1 geometry
#     3. Pristine Reference (Optional)
#     """
#     root = Path(settings.run_dir_root)
#     root.mkdir(parents=True, exist_ok=True)
#
#     if settings.dft_code == "aims":
#         write_func = generator.write_fhi_aims_input_files
#     elif settings.dft_code == "vasp":
#         write_func = generator.write_vasp_input_files
#     else:
#         raise ValueError(f"Unknown DFT tool: {settings.dft_code}")
#
#     # --- JOB 1: ATTRACTOR RELAXATION (M^0 Substitution) ---
#     attractor_dir = root / "01_Attractor_Run"
#
#     # 1. GENERATE INPUT FILES FOR JOB 1 (Static Generation)
#     write_func(
#         site_index=chosen_site_indices,
#         vacancy_site_index=chosen_vacancy_site_indices,
#         settings=settings,
#         outdir=str(attractor_dir),
#         is_charged_polaron_run=False,
#     )
#     script_attractor = self.write_simple_job_script(attractor_dir)
#
#     # --- ORCHESTRATION / DYNAMIC EXECUTION ---
#
#     planned = {
#         "attractor_dir": str(attractor_dir),
#         "attractor_script": script_attractor,
#         "instructions": "Run Job 1 first. Then run Job 2 and 3 sequentially using the geometry from Job 1."
#     }
#
#     relaxed_attractor_structure = None
#
#     if settings.do_submit:
#         # If submitting immediately, we must perform the execution loop
#         planned["submitted"] = True
#
#         # 1a. Submit Job 1 and wait for convergence (Conceptual)
#         log.info(f"Submitting Job 1: Attractor Run in {attractor_dir}")
#         # Assuming _run_job_and_wait exists and handles the execution/waiting logic
#         # _run_job_and_wait(script_attractor)
#
#         # 1b. Read the relaxed structure from Job 1 output (Conceptual)
#         # relaxed_attractor_structure = _read_final_geometry(settings.dft_code, attractor_dir)
#         log.info(f"Job 1 complete. Reading relaxed geometry for Job 2 input.")
#
#         if relaxed_attractor_structure is None:
#             log.error("Could not read relaxed geometry from Job 1. Stopping workflow.")
#             return planned
#
#         # --- JOB 2: FINAL POLARON RUN (Spin Seed, Dynamic Generation) ---
#         polaron_dir = root / "02_Final_Polaron_Run"
#
#         # 2. GENERATE INPUT FILES FOR JOB 2, STARTING FROM JOB 1's RELAXED STRUCTURE
#         write_func(
#             site_index=chosen_site_indices,  # Seed the spin on the target site
#             vacancy_site_index=chosen_vacancy_site_indices,
#             settings=settings,
#             outdir=str(polaron_dir),
#             is_charged_polaron_run=True,  # Charged/Magnetic calculation
#             base_structure=relaxed_attractor_structure  # <--- CRITICAL CHANGE: Use relaxed geometry
#         )
#         script_polaron = self.write_simple_job_script(polaron_dir)
#         planned["polaron_dir"] = str(polaron_dir)
#         planned["polaron_script"] = script_polaron
#
#         # 2a. Submit Job 2 and wait (Conceptual)
#         # _run_job_and_wait(script_polaron)
#         log.info(f"Submitted Job 2: Final Polaron Run in {polaron_dir}")
#
#         # --- JOB 3: PRISTINE REFERENCE (Optional, Dynamic Generation) ---
#         if settings.run_pristine:
#             pristine_dir = root / "03_Pristine_Ref"
#
#             # This job should ideally also start from the relaxed host lattice.
#             write_func(
#                 site_index=[],
#                 vacancy_site_index=chosen_vacancy_site_indices,
#                 settings=settings,
#                 outdir=str(pristine_dir),
#                 is_charged_polaron_run=False,
#                 base_structure=relaxed_attractor_structure  # Use relaxed host lattice
#             )
#             script_pristine = self.write_simple_job_script(pristine_dir)
#
#             planned["pristine_dir"] = str(pristine_dir)
#             planned["pristine_script"] = script_pristine
#
#             # 3a. Submit Job 3 and wait (Conceptual)
#             # _run_job_and_wait(script_pristine)
#             log.info(f"Submitted Job 3: Pristine Reference in {pristine_dir}")
#
#     else:
#         # If not submitting, we instruct the user how to generate the remaining files
#         planned["submitted"] = False
#         planned["instructions"] = (
#             "Job 1 inputs generated in 01_Attractor_Run. "
#             "After Job 1 converges, run this script again with the '--base-geometry' flag "
#             "pointing to the CONTCAR/geometry.in.next_step to generate Job 2 and 3 inputs."
#         )
#
#     # ... (Final analysis/report generation follows) ...
#     return {"planned": planned}

# import subprocess
# import logging
# from pathlib import Path
# from typing import Optional
#
# from pymatgen.core.structure import Structure
# from pyfhiaims.geometry import AimsGeometry  # Assuming this helper class is available
#
# log = logging.getLogger(__name__)
#
#

#
#
# def _read_final_geometry(dft_code: str, job_directory: Path) -> Optional[Structure]:
#     """
#     Reads the final relaxed structure from a completed DFT job directory.
#     """
#     dft_code = dft_code.lower()
#
#     if dft_code == "aims":
#         # FHI-AIMS output for relaxed geometry is typically geometry.in.next_step
#         final_geom_path = job_directory / "geometry.in.next_step"
#         if final_geom_path.exists():
#             log.info(f"Reading AIMS geometry from {final_geom_path.name}")
#             try:
#                 # Need to use the Aims-specific parser if the file isn't POSCAR-like
#                 return AimsGeometry.from_file(final_geom_path).structure
#             except Exception as e:
#                 log.error(f"Error parsing AIMS geometry: {e}")
#                 return None
#
#     elif dft_code == "vasp":
#         # VASP output for relaxed geometry
#         final_geom_path = job_directory / "CONTCAR"
#         if final_geom_path.exists():
#             log.info(f"Reading VASP CONTCAR from {final_geom_path.name}")
#             try:
#                 # Use standard pymatgen parser
#                 return Structure.from_file(str(final_geom_path))
#             except Exception as e:
#                 log.error(f"Error parsing VASP CONTCAR: {e}")
#                 return None
#
#     log.error(f"Final geometry file not found for {dft_code.upper()} in {job_directory.name}.")
#     return None


# from typing import Dict, Tuple, Optional, Union, List, Any
# import textwrap
# import logging
# from pathlib import Path
# import subprocess
# import json
# import numpy as np
#
# # Assuming these are available from your refactored project structure:
# from pypolaron.polaron_generator import PolaronGenerator
# from pypolaron.polaron_analyzer import PolaronAnalyzer
# from pypolaron.utils.dft_settings import DftSettings  # Assuming DftSettings lives here
# from pypolaron.utils.workflow_helpers import _run_job_and_wait, _read_final_geometry
#
# log = logging.getLogger(__name__)
#
#
# class PolaronWorkflow:
#     def __init__(
#             self,
#             aims_executable_command: str,
#             epsilon: Optional[float] = None,
#             fermi_energy: float = 0.0,
#             volume_ang3: Optional[float] = None,
#     ):
#         self.aims_executable_command = aims_executable_command
#         self.epsilon = epsilon
#         self.fermi_energy = fermi_energy
#         self.volume_ang3 = volume_ang3
#
#     def write_simple_job_script(
#             self,
#             workdir: Path,
#             nthreads: int = 8,
#             walltime: str = "02:00:00",
#             scheduler: Optional[str] = None,
#     ) -> str:
#         """
#         Write a simple shell script to run FHI-aims in 'workdir'.
#         """
#         workdir.mkdir(parents=True, exist_ok=True)
#         script_path = workdir / "run_aims.sh"
#         # --- (SLURM/SHELL SCRIPT CONTENT REMAINS THE SAME) ---
#         if scheduler == "slurm":
#             content = textwrap.dedent(
#                 f"""\
#                 #!/bin/bash
#                 #SBATCH --job-name=aims_job
#                 #SBATCH --time={walltime}
#                 #SBATCH --cpus-per-task={nthreads}
#                 #SBATCH --output=aims.out\n
#                 cd {workdir}
#                 {self.aims_executable_command}
#             """
#             )
#         else:
#             content = textwrap.dedent(
#                 f"""\
#                  #!/bin/bash
#                  cd {workdir}
#                  {self.aims_executable_command} > aims.out 2>&1
#             """
#             )
#         # --- (END SCRIPT CONTENT) ---
#         script_path.write_text(content)
#         script_path.chmod(0o755)
#         return str(script_path)
#
#     def run_polaron_workflow(
#             self,
#             generator: PolaronGenerator,
#             chosen_site_indices: Union[int, List[int]],
#             chosen_vacancy_site_indices: Union[int, List[int]],
#             settings: DftSettings,
#     ) -> Dict[str, Union[str, float, Dict[str, Any]]]:
#         """ Standard polaron workflow (non-sequential, not shown fully here) """
#         # ... (Implementation of standard run_polaron_workflow) ...
#         return {}  # Placeholder
#
#     def run_attractor_workflow(
#             self,
#             generator: PolaronGenerator,
#             chosen_site_indices: Union[int, List[int]],
#             chosen_vacancy_site_indices: Union[int, List[int]],
#             settings: DftSettings,
#     ) -> Dict[str, Union[str, float, Dict[str, Any]]]:
#         """
#         Manages the sequential file generation and execution for the Electron Attractor method.
#         """
#         root = Path(settings.run_dir_root)
#         root.mkdir(parents=True, exist_ok=True)
#
#         if settings.dft_code == "aims":
#             write_func = generator.write_fhi_aims_input_files
#         elif settings.dft_code == "vasp":
#             write_func = generator.write_vasp_input_files
#         else:
#             raise ValueError(f"Unknown DFT tool: {settings.dft_code}")
#
#         # --- JOB 1: ATTRACTOR RELAXATION (M^0 Substitution) ---
#         attractor_dir = root / "01_Attractor_Run"
#
#         # 1. Generate Input Files for Job 1 (Starts from initial structure)
#         log.info(f"Preparing Job 1 inputs in {attractor_dir.name}")
#         write_func(
#             site_index=chosen_site_indices,
#             vacancy_site_index=chosen_vacancy_site_indices,
#             settings=settings,
#             outdir=str(attractor_dir),
#             is_charged_polaron_run=False,
#         )
#         script_attractor = self.write_simple_job_script(attractor_dir)
#
#         # --- ORCHESTRATION / DYNAMIC EXECUTION ---
#
#         planned = {
#             "attractor_dir": str(attractor_dir),
#             "attractor_script": script_attractor,
#             "instructions": "Run Job 1 first. Then use the output geometry to generate files for Job 2 and 3."
#         }
#
#         # This will store the relaxed structure from Job 1
#         relaxed_attractor_structure = None
#
#         if settings.do_submit:
#             log.info("Starting sequential execution (do_submit=True)")
#             planned["submitted"] = True
#
#             # 1a. EXECUTE Job 1 and wait for convergence
#             try:
#                 _run_job_and_wait(Path(script_attractor))
#                 log.info(f"Job 1 completed successfully in {attractor_dir.name}.")
#             except Exception as e:
#                 log.error(f"Job 1 (Attractor Run) failed. Stopping workflow. Error: {e}")
#                 planned["status"] = "Job 1 Failed"
#                 return {"planned": planned}
#
#             # 1b. Read the relaxed structure from Job 1 output
#             relaxed_attractor_structure = _read_final_geometry(settings.dft_code, attractor_dir)
#
#             if relaxed_attractor_structure is None:
#                 log.error("Could not read relaxed geometry from Job 1. Stopping workflow.")
#                 planned["status"] = "Failed to Read Job 1 Output"
#                 return {"planned": planned}
#
#             log.info("Successfully read relaxed geometry for subsequent jobs.")
#
#             # --- JOB 2: FINAL POLARON RUN (Dynamic Generation/Execution) ---
#             polaron_dir = root / "02_Final_Polaron_Run"
#             log.info(f"Preparing Job 2 inputs in {polaron_dir.name}")
#
#             # 2. GENERATE INPUT FILES FOR JOB 2
#             write_func(
#                 site_index=chosen_site_indices,
#                 vacancy_site_index=chosen_vacancy_site_indices,
#                 settings=settings,
#                 outdir=str(polaron_dir),
#                 is_charged_polaron_run=True,
#                 base_structure=relaxed_attractor_structure  # <--- CRITICAL: Uses relaxed geometry
#             )
#             script_polaron = self.write_simple_job_script(polaron_dir)
#             planned["polaron_dir"] = str(polaron_dir)
#             planned["polaron_script"] = script_polaron
#
#             # 2a. Execute Job 2 and wait
#             try:
#                 _run_job_and_wait(Path(script_polaron))
#                 log.info(f"Job 2 completed successfully in {polaron_dir.name}.")
#             except Exception as e:
#                 log.error(f"Job 2 (Polaron Run) failed. Stopping workflow. Error: {e}")
#                 planned["status"] = "Job 2 Failed"
#                 return {"planned": planned}
#
#             # --- JOB 3: PRISTINE REFERENCE (Optional, Dynamic Generation/Execution) ---
#             if settings.run_pristine:
#                 pristine_dir = root / "03_Pristine_Ref"
#                 log.info(f"Preparing Job 3 inputs in {pristine_dir.name}")
#
#                 # 3. GENERATE INPUT FILES FOR JOB 3 (Starts from relaxed host lattice)
#                 write_func(
#                     site_index=[],
#                     vacancy_site_index=chosen_vacancy_site_indices,
#                     settings=settings,
#                     outdir=str(pristine_dir),
#                     is_charged_polaron_run=False,
#                     base_structure=relaxed_attractor_structure  # Use relaxed host lattice
#                 )
#                 script_pristine = self.write_simple_job_script(pristine_dir)
#                 planned["pristine_dir"] = str(pristine_dir)
#                 planned["pristine_script"] = script_pristine
#
#                 # 3a. Execute Job 3 and wait
#                 try:
#                     _run_job_and_wait(Path(script_pristine))
#                     log.info(f"Job 3 completed successfully in {pristine_dir.name}.")
#                 except Exception as e:
#                     log.error(f"Job 3 (Pristine Run) failed. Continuing analysis if possible. Error: {e}")
#                     planned["status"] = "Job 3 Failed"
#         else:
#             # If not submitting, we instruct the user how to generate the remaining files
#             planned["submitted"] = False
#             planned["instructions"] = (
#                 "Job 1 inputs generated in 01_Attractor_Run. "
#                 "To continue the workflow manually, run Job 1, then use the resulting CONTCAR/geometry.in.next_step "
#                 "as the base structure for Job 2 and 3 inputs."
#             )
#
#         # --- (Final analysis/report generation follows) ---
#         return {"planned": planned}
