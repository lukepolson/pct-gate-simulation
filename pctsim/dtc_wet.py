"""Stopping powers of the DTC's materials from Geant4 (EmCalculatorActor, QGSP_BIC_EMY), each material's RSP, the
water-equivalent thickness (WET) of one tracker and one calorimeter layer, and the CSDA range of protons in the
calorimeter stack, with a Bragg-Kleeman fit R = alpha E^p (depth in mm from the front face of calorimeter layer 0,
air gaps included, as Pettersen et al. 2019 fit it to their range-energy data).

Writes OUT_DIR/dtc_materials.npz and .json.

Usage: python -I dtc_wet.py OUT_DIR
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import curve_fit

import opengate as gate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtc_geometry as dg  # noqa: E402

out = pathlib.Path(sys.argv[1])
(out / 'dedx').mkdir(parents=True, exist_ok=True)
mm, MeV, gcm3 = gate.g4_units.mm, gate.g4_units.MeV, gate.g4_units.g_cm3
ENERGIES = np.geomspace(1.0, 360.0, 251)
MATERIALS = ['G4_WATER', 'G4_Al', 'G4_Si', 'G4_KAPTON', 'DTC_Epoxy', 'DTC_CarbonEpoxy', 'G4_AIR']

sim = gate.Simulation()
sim.g4_verbose = False
sim.visu = False
sim.output_dir = out
sim.random_seed = 1
sim.world.size = [1000 * mm, 1000 * mm, 1000 * mm]
sim.world.material = 'G4_AIR'
dg.add_materials(sim)
holder = sim.add_volume('Box', 'holder')            # the new materials must be used by a volume to be built
holder.size = [10 * mm, 10 * mm, 10 * mm]
holder.material = 'DTC_CarbonEpoxy'
holder2 = sim.add_volume('Box', 'holder2')
holder2.size = [10 * mm, 10 * mm, 10 * mm]
holder2.translation = [0, 0, 100 * mm]
holder2.material = 'DTC_Epoxy'
sim.physics_manager.physics_list_name = 'QGSP_BIC_EMY'
for k, m in enumerate(MATERIALS):
    a = sim.add_actor('EmCalculatorActor', f'dedx_{k}')
    a.is_ion = False
    a.particle_name = 'proton'
    a.material = m
    a.nominal_energies = [float(e) * MeV for e in ENERGIES]
    a.savefile_path = str(out / 'dedx' / f'{m}.txt')
src = sim.add_source('GenericSource', 'one')
src.particle = 'proton'
src.energy.mono = 150 * MeV
src.position.translation = [0, 0, -400 * mm]
src.direction.type = 'momentum'
src.direction.momentum = [0, 0, 1]
src.n = 1
sim.run()


def table(path):
    vals = []
    for ln in pathlib.Path(path).read_text().splitlines():
        try:
            vals.append([float(v) for v in ln.replace(',', ' ').split()])
        except ValueError:
            continue
    return np.array([v for v in vals if len(v) >= 2])


db = sim.volume_manager.material_database
density = {m: db.FindOrBuildMaterial(m).GetDensity() / gcm3 for m in MATERIALS}
S = {m: table(out / 'dedx' / f'{m}.txt')[:, 1] * density[m] / 10.0 for m in MATERIALS}     # MeV/mm
E = ENERGIES
avg = (E >= 100) & (E <= 200)
rsp = {m: float(np.mean(S[m][avg] / S['G4_WATER'][avg])) for m in MATERIALS}
print('RSP (mean over 100-200 MeV):', {m: round(v, 4) for m, v in rsp.items()})

wet = {}
for kind in ('trk', 'cal'):
    slabs = dg.layer_slabs(kind)
    w = sum((y1 - y0) * rsp[mat] for _, mat, y0, y1 in slabs)
    wet[kind] = {'wet_mm': w, 'physical_mm': slabs[-1][3] - slabs[0][2],
                 'slabs': [(n, mat, round(y1 - y0, 4), round((y1 - y0) * rsp[mat], 4)) for n, mat, y0, y1 in slabs]}
    print(f'{kind}: WET {w:.3f} mm over {wet[kind]["physical_mm"]:.3f} mm')
    for s in wet[kind]['slabs']:
        print('   ', s)


def stopping(mat, e):
    return np.exp(np.interp(np.log(np.maximum(e, E[0])), np.log(E), np.log(S[mat])))


# CSDA range through the calorimeter stack, from the front face of layer 0's carrier A
slabs = dg.layer_slabs('cal')
period = []                                          # (material, thickness) over one 5.5 mm pitch, air included
y = slabs[0][2]
for _, mat, y0, y1 in slabs:
    period.append((mat, y1 - y0))
pitch_check = sum(t for _, t in period)
assert abs(pitch_check - dg.PITCH) < 1e-9, pitch_check


def csda_depth(e0, n_layers=120, dy=0.005):
    e, depth = e0, 0.0
    for _ in range(n_layers):
        for mat, t in period:
            n = max(1, int(np.ceil(t / dy)))
            h = t / n
            for _ in range(n):
                s = stopping(mat, e)
                if s * h >= e:
                    return depth + e / s
                e -= s * h
                depth += h
    return np.nan


e_fit = np.arange(30.0, 340.1, 10.0)
r_det = np.array([csda_depth(e) for e in e_fit])
r_water = np.array([np.trapezoid(1.0 / stopping('G4_WATER', np.linspace(0.5, e, 4000)),
                                 np.linspace(0.5, e, 4000)) for e in e_fit])


def bk(e, alpha, p):
    return alpha * e ** p


fit_sel = (e_fit >= 50) & (e_fit <= 260)
(alpha, p), _ = curve_fit(bk, e_fit[fit_sel], r_det[fit_sel], p0=(0.02, 1.7))
resid = r_det - bk(e_fit, alpha, p)
print(f'Bragg-Kleeman fit on the calorimeter stack (50-260 MeV): alpha {alpha:.5f} mm/MeV^p, p {p:.4f}; '
      f'residual rms {np.sqrt(np.mean(resid[fit_sel] ** 2)):.3f} mm, max {np.max(np.abs(resid[fit_sel])):.3f} mm')
sel340 = e_fit >= 50
(alpha340, p340), _ = curve_fit(bk, e_fit[sel340], r_det[sel340], p0=(alpha, p))
resid340 = r_det - bk(e_fit, alpha340, p340)
print(f'Bragg-Kleeman fit over 50-340 MeV: alpha {alpha340:.5f}, p {p340:.4f}; residual rms '
      f'{np.sqrt(np.mean(resid340[sel340] ** 2)):.3f} mm, max {np.max(np.abs(resid340[sel340])):.3f} mm')
layers_needed = {int(e): float(r) / dg.PITCH for e, r in zip(e_fit, r_det)}
print('layers to stop (CSDA) at 230, 250, 300, 330 MeV:',
      [round(layers_needed[e], 2) for e in (230, 250, 300, 330)])

np.savez(out / 'dtc_materials.npz', energies=E, materials=np.array(MATERIALS),
         stopping_mev_per_mm=np.array([S[m] for m in MATERIALS]), e_fit=e_fit, csda_depth_mm=r_det,
         csda_water_mm=r_water, bk_alpha=alpha, bk_p=p)
res = {'rsp_100_200MeV': rsp, 'density_g_cm3': density, 'layer_wet': wet,
       'bragg_kleeman': {'alpha_mm_per_MeV_p': alpha, 'p': p, 'fit_MeV': [50, 260],
                         'residual_rms_mm': float(np.sqrt(np.mean(resid[fit_sel] ** 2)))},
       'bragg_kleeman_340': {'alpha_mm_per_MeV_p': alpha340, 'p': p340, 'fit_MeV': [50, 340],
                             'residual_rms_mm': float(np.sqrt(np.mean(resid340[sel340] ** 2)))},
       'csda': {str(int(e)): {'depth_mm': float(r), 'water_mm': float(w), 'layers': float(r / dg.PITCH)}
                for e, r, w in zip(e_fit, r_det, r_water)}}
(out / 'dtc_materials.json').write_text(json.dumps(res, indent=1))
print('wrote', out / 'dtc_materials.json')
