from pymatgen.core import Structure
from pypolaron import PolaronGenerator, PolaronWorkflow
from pymatgen.ext.matproj import MPRester
# from mp_api.client import MPRester

API_key = 'Lau0fNd1JaP2nupVt1U0zriqEuyL4TdT'

# Instantiate MPRester (reads API key from environment)
with MPRester(API_key) as mpr:
    # Get the first matching structure for TiO2
    structure = mpr.get_structure_by_material_id("mp-2657")  # mp-2657 = rutile TiO2

# Electron polaron search (extra electron → likely Ti site)
pg_e = PolaronGenerator(structure, polaron_type="electron")
candidates_e = pg_e.propose_sites()
best_sites = candidates_e[0][0]
print(best_sites)

workflow = PolaronWorkflow(aims_executable_command ='mpirun -np 24 /path/to/aims', )
workflow.run_polaron_workflow(polgen=pg_e, chosen_site_indices=best_sites,
                              species_dir='/home/rinaldi/Documents/programs/FHIaims/species_defaults/defaults_2020/tight')

