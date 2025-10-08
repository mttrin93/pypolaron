from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import textwrap
import numpy as np
import subprocess
import json

from pypolaron.polaron_generator import PolaronGenerator
from pypolaron.polaron_analyzer import PolaronAnalyzer

class PolaronWorkflow:
    def __init__(self, aims_executable_command: str, epsilon: Optional[float] = None,
                 fermi_energy: float = 0.0, volume_ang3: Optional[float] = None):
        self.aims_executable_command = aims_executable_command
        self.epsilon = epsilon
        self.fermi_energy = fermi_energy
        self.volume_ang3 = volume_ang3

    def write_simple_job_script(self, workdir: Path, nthreads: int = 8, walltime: str = "02:00:00",
                                scheduler: Optional[str] = None) -> str:
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
            content = textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name=aims_job
                #SBATCH --time={walltime}
                #SBATCH --cpus-per-task={nthreads}
                #SBATCH --output=aims.out\n
                cd {workdir}
                {self.aims_executable_command}
            """)
        else:
            content = textwrap.dedent(f"""\
                 #!/bin/bash
                 cd {workdir}
                 {self.aims_executable_command} > aims.out 2>&1  
            """)
        script_path.write_text(content)
        script_path.chmod(0o755)
        return str(script_path)

    def run_polaron_workflow(self,
                             polgen: Optional[PolaronGenerator],
                             chosen_site_indices: List[int],
                             supercell=(2, 2, 2),
                             add_charge: int = -1,
                             spin_moments: float = None,
                             run_dir_root: str = "polaron_runs",
                             species_dir: str = None,
                             do_submit: bool = False,
                             do_bader: bool = True,
                             potential_axis: int = 2,
                             dielectric_eps: float = 10.0,
                             auto_analyze: bool = True) -> Dict:
        """
        High-level orchestrator for polaron calculations + optional analysis.
        auto_analyze: if True, calls PolaronAnalyzer if outputs exist

        A first run should be performed with do_submit=True to produce aims.out and cube files
        followed by do_submit=False to analyze the outputs
        """
        root = Path(run_dir_root)
        root.mkdir(parents=True, exist_ok=True)

        # Directories
        pristine_dir = root / "pristine"
        charged_dir = root / "charged"

        # Write pristine inputs (no added charge)
        # TODO: add functionality where the write_fhi_aims method recongnize if there are polarons or not, here
        # is adding unecessary magnetic moment to the chose_site_indices atoms
        polgen.write_fhi_aims_input_files(site_index=chosen_site_indices, supercell=supercell, add_charge=0.,
                                          species_dir=species_dir, outdir=pristine_dir)
        # Write charged inputs
        polgen.write_fhi_aims_input_files(site_index=chosen_site_indices, supercell=supercell, add_charge=add_charge,
                                     species_dir=species_dir, outdir=charged_dir)

        # 2) Write job scripts
        script_pr = self.write_simple_job_script(pristine_dir)
        script_ch = self.write_simple_job_script(charged_dir)

        planned = {
            "pristine_dir": str(pristine_dir),
            "charged_dir": str(charged_dir),
            "pristine_script": script_pr,
            "charged_script": script_ch,
            "instructions": "Run the scripts in the respective directories to perform calculations."
        }

        # 3) Optionally submit
        if do_submit:
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
                analyzer = PolaronAnalyzer(fermi_energy=self.fermi_energy,
                                           epsilon=self.epsilon,
                                           volume_ang3=self.volume_ang3)

                # Automatically find outputs
                pristine_out = pristine_dir / "aims.out"
                charged_out = charged_dir / "aims.out"
                potential_cube_pristine = pristine_dir / "potential.cube"
                potential_cube_charged = charged_dir / "potential.cube"

                if pristine_out.exists() and charged_out.exists():
                    results = analyzer.analyze_polaron_run(
                        pristine_out=str(pristine_out),
                        charged_out=str(charged_out),
                        atom_coords=np.array([0, 0, 0]),  # placeholder: pass real defect coords
                        site_index_supercell=chosen_site_indices[0],
                        total_charge=add_charge,
                        pot_cube_pristine=str(potential_cube_pristine) if potential_cube_pristine.exists() else None,
                        pot_cube_charged=str(potential_cube_charged) if potential_cube_charged.exists() else None
                    )
                    report["analysis"] = results
                else:
                    report["analysis_status"] = "Aims outputs not found; run calculations first."
            except Exception as e:
                report["analysis_error"] = str(e)

        # 5) Save report
        report_path = root / "polaron_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        return report
