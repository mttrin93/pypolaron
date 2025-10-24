from pymatgen.core import Structure
from pypolaron import PolaronGenerator, PolaronWorkflow
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path
from ase.io import read

# API_key = "Lau0fNd1JaP2nupVt1U0zriqEuyL4TdT"
#
# # Instantiate MPRester (reads API key from environment)
# with MPRester(API_key) as mpr:
#     # Get the first matching structure for TiO2
#     structure = mpr.get_structure_by_material_id("mp-2657")  # mp-2657 = rutile TiO2

path = '/home/rinaldi/2_polaron/geometry_0.xyz'

structure_ase = read(path)
structure = AseAtomsAdaptor.get_structure(structure_ase)

# Electron polaron search (extra electron → likely Ti site)
pg_e = PolaronGenerator(structure, polaron_type="electron")
candidates_e = pg_e.propose_sites(max_sites=2)
# best_sites = candidates_e[0][0]
best_sites = [candidates_e[0][0]] + [candidates_e[1][0]]
print(best_sites)

workflow = PolaronWorkflow(aims_executable_command="mpirun -np 24 /path/to/aims")

attract_struc = workflow.create_attractor_structure(pg_e, best_sites, 'V')
print(attract_struc)

# for site in attract_struc:
#     print(site.specie.oxi_state)

# workdir = Path("./")
# workflow.write_simple_job_script(workdir)
#
# report = workflow.run_polaron_workflow(
#     generator=pg_e,
#     chosen_site_indices=best_sites,
#     species_dir="/home/rinaldi/Documents/programs/FHIaims/species_defaults/defaults_2020/tight",
# )

# print(report)
