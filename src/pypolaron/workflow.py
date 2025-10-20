from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import textwrap
import numpy as np
import subprocess
import json

from pypolaron.polaron_generator import PolaronGenerator
from pypolaron.polaron_analyzer import PolaronAnalyzer

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
        generator: Optional[PolaronGenerator],
        chosen_site_indices: Union[int, List[int]],
        chosen_vacancy_site_indices: Union[int, List[int]],
        supercell: Tuple[int, int, int] = (2, 2, 2),
        add_charge: int = -1,   # TODO: obsolete argument
        spin_moment: float = 1.0,
        set_site_magmoms: bool = False,
        run_dir_root: str = "polaron_runs",
        species_dir: str = None,
        do_submit: bool = False,
        calc_type: str = "relax-atoms",
        functional: str = "hse06",
        dft_code: str = "aims",
        run_pristine: bool = False,
        do_bader: bool = True,
        potential_axis: int = 2,
        dielectric_eps: float = 10.0,
        auto_analyze: bool = True,
    ) -> Dict[str, Union[str, float]]:
        """
        High-level orchestrator for polaron calculations + optional analysis.
        auto_analyze: if True, calls PolaronAnalyzer if outputs exist

        A first run should be performed with do_submit=True to produce aims.out and cube files
        followed by do_submit=False to analyze the outputs
        """
        root = Path(run_dir_root)
        root.mkdir(parents=True, exist_ok=True)

        # Directories
        pristine_dir = root / "pristine" if run_pristine else None
        charged_dir = root / "charged"

        script_pr = None

        common_args = {
            "site_index": chosen_site_indices,
            "vacancy_site_index": chosen_vacancy_site_indices,
            "supercell": supercell,
            "spin_moment": spin_moment,
            "set_site_magmoms": set_site_magmoms,
            "calc_type": calc_type,
            "functional": functional,
        }

        if dft_code == "aims":
            write_func = generator.write_fhi_aims_input_files
            common_args['species_dir'] = species_dir
        elif dft_code == "vasp":
            write_func = generator.write_vasp_input_files
            common_args['potcar_dir'] = species_dir
        else:
            raise ValueError(f"Unknown DFT tool: {dft_code}. Must be 'aims' or 'vasp'.")

        if run_pristine:
            write_func(
                **common_args,
                outdir=str(pristine_dir),
                is_charged_polaron_run=False,
            )
            script_pr = self.write_simple_job_script(pristine_dir)

        write_func(
            **common_args,
            outdir=str(charged_dir),
            is_charged_polaron_run=True,
        )

        # 2) Write job scripts
        pristine_dir_str = str(script_pr) if run_pristine else None
        script_ch = self.write_simple_job_script(charged_dir)

        planned = {
            "pristine_dir": pristine_dir_str,
            "charged_dir": str(charged_dir),
            "pristine_script": script_pr,
            "charged_script": script_ch,
            "instructions": "Run the scripts in the respective directories to perform calculations.",
        }

        # 3) Optionally submit
        if do_submit:
            if run_pristine:
                subprocess.run(["bash", script_pr], check=True)
            subprocess.run(["bash", script_ch], check=True)
            planned["submitted"] = True
        else:
            planned["submitted"] = False

        # Initialize report
        report = {"planned": planned}

        # 4) Optional post-processing
        if auto_analyze:
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
                        total_charge=add_charge,
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
