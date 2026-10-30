"""A phantom's GATE materials (Schneider 2000, density tolerance 0.05 g/cm3), their label image, and each material's
relative stopping power for protons from Geant4's own stopping powers (EmCalculatorActor, QGSP_BIC_EMY).

RSP is the material's total stopping power over water's, averaged over 100, 150 and 200 MeV (it changes by well under
1% across that range). Writes OUT_DIR/NAME_labels.mhd, OUT_DIR/dedx/*.txt and OUT_DIR/NAME_rsp.npz (the RSP of every
voxel on the phantom grid, and the table per material).

Usage: python -I pct_materials.py PHANTOM_DIR OUT_DIR                 (the head: PHANTOM_DIR/head_hu.mhd, NAME head)
       python -I pct_materials.py PHANTOM_MHD OUT_DIR NAME
"""
import pathlib
import sys

import numpy as np
import opengate as gate

if len(sys.argv) > 3:
    phantom_mhd, out_dir, name = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
else:
    phantom_mhd, out_dir, name = pathlib.Path(sys.argv[1]) / 'head_hu.mhd', pathlib.Path(sys.argv[2]), 'head'
(out_dir / 'dedx').mkdir(parents=True, exist_ok=True)
mm, MeV, gcm3 = gate.g4_units.mm, gate.g4_units.MeV, gate.g4_units.g_cm3
ENERGIES = [100.0, 150.0, 200.0]

sim = gate.Simulation()
sim.g4_verbose = False
sim.visu = False
sim.output_dir = out_dir
sim.random_seed = 1
sim.world.size = [1000 * mm, 1000 * mm, 1000 * mm]
sim.world.material = 'G4_AIR'
head = sim.add_volume('Image', 'head')
head.image = str(phantom_mhd)
head.material = 'G4_AIR'
data = gate.utility.get_data_folder()
head.voxel_materials, materials = gate.geometry.materials.HounsfieldUnit_to_material(
    sim, 0.05 * gcm3, str(data / 'Schneider2000MaterialsTable.txt'), str(data / 'Schneider2000DensitiesTable.txt'))
head.dump_label_image = out_dir / f'{name}_labels.mhd'
print(f'{len(materials)} materials')
sim.physics_manager.physics_list_name = 'QGSP_BIC_EMY'

names = list(materials) + ['G4_WATER']
for k, m in enumerate(names):
    a = sim.add_actor('EmCalculatorActor', f'dedx_{k}')
    a.is_ion = False
    a.particle_name = 'proton'
    a.material = m
    a.nominal_energies = [e * MeV for e in ENERGIES]
    a.savefile_path = str(out_dir / 'dedx' / f'{k:03d}_{m}.txt')

src = sim.add_source('GenericSource', 'one')
src.particle = 'proton'
src.energy.mono = 150 * MeV
src.position.translation = [0, 0, -400 * mm]
src.direction.type = 'momentum'
src.direction.momentum = [0, 0, 1]
src.n = 1
sim.run()

# --- RSP per material and per voxel
def dedx(path):
    vals = []
    for ln in pathlib.Path(path).read_text().splitlines():
        p = ln.replace(',', ' ').split()
        try:
            vals.append([float(v) for v in p])
        except ValueError:
            continue
    return np.array([v for v in vals if len(v) >= 2])

tables = [dedx(out_dir / 'dedx' / f'{k:03d}_{m}.txt') for k, m in enumerate(names)]
print('example table (first material, MeV cm2/g):\n', tables[0][:5])
water = tables[-1]
db = sim.volume_manager.material_database
density = np.array([db.FindOrBuildMaterial(m).GetDensity() / gcm3 for m in names])   # g/cm3
rsp = []
for k, m in enumerate(materials):
    # the tables are mass stopping powers: RSP = (S/rho)_m rho_m / ((S/rho)_w rho_w)
    rsp.append(float(np.mean(tables[k][:, -1] * density[k] / (water[:, -1] * density[-1]))))
rsp = np.array(rsp)
import itk  # noqa: E402
lab = itk.GetArrayFromImage(itk.imread(str(out_dir / f'{name}_labels.mhd'))).transpose(2, 1, 0)   # back to [x, y, z]
# the label image keeps 0 for the default material (G4_AIR here) and k + 1 for voxel_materials[k]
lab = lab.astype(np.int16) - 1
lab[lab < 0] = 0
voxel_rsp = rsp[lab].astype(np.float32)
used = np.unique(lab)
for k, m in enumerate(materials):
    if k in used:
        print(f'  {k:3d} {m:22s} HU {head.voxel_materials[k][0]:7.0f}..{head.voxel_materials[k][1]:7.0f}  '
              f'{density[k]:.4f} g/cm3  RSP {rsp[k]:.4f}  voxels {np.count_nonzero(lab == k)}')
np.savez_compressed(out_dir / f'{name}_rsp.npz', rsp=voxel_rsp, labels=lab.astype(np.int16), material_rsp=rsp,
                    material_density=density[:-1],
                    material_names=np.array(materials), hu_intervals=np.array([v[:2] for v in head.voxel_materials],
                                                                                   dtype=float))
print('wrote', out_dir / f'{name}_rsp.npz', 'RSP range', voxel_rsp.min(), voxel_rsp.max())
