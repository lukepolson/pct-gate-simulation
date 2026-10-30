"""Figures of the DTC model: one calorimeter and one tracker layer to scale (with the two half-layers' chips
interleaving in z), the range-energy relation of the calorimeter stack with its Bragg-Kleeman fit, and how deep the
protons of real scan views reach (the depth decision: 41 or 47 layers).

Inputs: gate/dtc/materials/dtc_materials.json, gate/dtc/beam_test_330/dtc_beam_check.json, and the converted views of
the timing runs in gate/dtc/timing_parallel/<run>/view_XXX.npz (those present are used).

Usage: python -I dtc_geometry_plots.py DATA_DIR
"""
import json
import pathlib
import sys

import numpy as np
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dtc_geometry as dg  # noqa: E402
import style  # noqa: E402
from style import OURS, REAL, REF, plt  # noqa: E402

data = pathlib.Path(sys.argv[1])
out = data / 'realism' / 'plots'
out.mkdir(parents=True, exist_ok=True)
mats = json.loads((data / 'gate' / 'dtc' / 'materials' / 'dtc_materials.json').read_text())
COL = {'G4_Al': '#9aa3ad', 'G4_Si': '#3b6fb6', 'G4_KAPTON': '#d9a441', 'DTC_Epoxy': '#7c5a3a',
       'DTC_CarbonEpoxy': '#2f2f2f', 'G4_AIR': '#ffffff'}
NAME = {'G4_Al': 'aluminium', 'G4_Si': 'silicon (chip)', 'G4_KAPTON': 'polyimide (flex)', 'DTC_Epoxy': 'epoxy glue',
        'DTC_CarbonEpoxy': 'carbon–epoxy'}

# ---- 1. one layer of each kind, to scale
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw={'width_ratios': [1.25, 0.75, 1.1]})
for a, kind, title in ((axes[0], 'cal', 'Calorimeter layer (×41 Bergen, ×77 for 330 MeV), 5.5 mm pitch'),
                       (axes[1], 'trk', 'Tracker layer (×2)')):
    p = dg.CAL if kind == 'cal' else dg.TRK
    cm, ct = p['carrier']
    chip_t = p['chip']
    # plates and flex sheets span z; chips sit in bands
    for name, mat, y0, y1 in dg.layer_slabs(kind):
        if name.startswith('chip') or mat == 'G4_AIR':
            continue
        a.add_patch(Rectangle((y0, -30), y1 - y0, 60, fc=COL[mat], ec='none'))
    for half, sign in (('A', 1), ('B', -1)):
        base = -sign * dg.GAP / 2 + sign * (dg.GLUE + dg.POLYIMIDE + dg.FLEX_AL)
        y0 = min(base, base + sign * chip_t)
        for b in range(dg.N_BANDS):
            if b % 2 != (0 if half == 'A' else 1):
                continue
            z0 = dg.Z_LO + b * dg.PIXELS_Z
            if z0 > 32 or z0 + dg.CHIP_Z < -32:
                continue
            a.add_patch(Rectangle((y0, z0), chip_t, dg.CHIP_Z, fc=COL['G4_Si'], ec='none'))
            epi0 = y0 + (chip_t - dg.EPI if sign > 0 else 0)
            a.add_patch(Rectangle((epi0, z0), dg.EPI, dg.PIXELS_Z, fc='#e8492e', ec='none'))
    lo = dg.layer_slabs(kind)[0][2]
    hi = dg.layer_slabs(kind)[-1][3]
    a.set_xlim(lo - 0.1, hi + 0.1)
    a.set_ylim(-30, 30)
    a.grid(False)
    a.set_xlabel('y along the beam, mm (0 = centre of the gap)')
    a.set_ylabel('z, mm' if kind == 'cal' else '')
    w = mats['layer_wet'][kind]['wet_mm']
    a.set_title(f'{title}\n{hi - lo:.2f} mm thick, {w:.2f} mm WET (outside the overlaps)', fontsize=10)
    a.annotate('half A', (lo + ct + 0.05, 27), fontsize=8.5)
    a.annotate('half B', (dg.GAP / 2 - 0.55, 27), fontsize=8.5)
handles = [Rectangle((0, 0), 1, 1, fc=c, ec='0.6' if m == 'G4_AIR' else 'none') for m, c in COL.items() if m != 'G4_AIR']
labels = [NAME[m] for m in COL if m != 'G4_AIR']
handles.append(Rectangle((0, 0), 1, 1, fc='#e8492e'))
labels.append('25 µm sensitive layer')
axes[1].legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
a = axes[2]
a.axis('off')
lines = ['WET per slab, calorimeter layer (mm):']
for n, mat, t, wet in mats['layer_wet']['cal']['slabs']:
    lines.append(f'  {n:12s} {t * 1000:6.0f} µm  {wet:6.3f}')
lines.append(f'  {"total":12s} {mats["layer_wet"]["cal"]["physical_mm"] * 1000:6.0f} µm  '
             f'{mats["layer_wet"]["cal"]["wet_mm"]:6.3f}')
lines.append('')
lines.append('Alme 2020: "7.5 mm WET" for the 3.5 mm of Al alone;')
lines.append('the flex, glue and chip add about 0.7 mm.')
a.text(0.12, 0.98, '\n'.join(lines), family='monospace', fontsize=8.2, va='top', transform=a.transAxes)
style.save(fig, out / 'dtc_layer_stack.png')

# ---- 2. range-energy relation and the Bragg-Kleeman fit
csda = mats['csda']
E = np.array([float(k) for k in csda])
r = np.array([csda[k]['depth_mm'] for k in csda])
fits = [(mats['bragg_kleeman'], '-', 260), (mats['bragg_kleeman_340'], '--', 340)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.4, 1]})
a = axes[0]
a.plot(E, r, 'o', ms=4, color=OURS, label='CSDA range in our stack (Geant4 stopping powers)')
for f, ls, top in fits:
    ee = np.linspace(30, top, 300)
    a.plot(ee, f['alpha_mm_per_MeV_p'] * ee ** f['p'], ls, color=REAL,
           label=f'Bragg–Kleeman fit, 50–{top} MeV: α = {f["alpha_mm_per_MeV_p"]:.4f}, p = {f["p"]:.3f}')
ee = np.linspace(30, 260, 300)
a.plot(ee, 0.0211 * ee ** 1.639, ':', color=REF, label='Pettersen 2019, 4 mm Al slabs: α = 0.0211, p = 1.639')
chk = data / 'gate' / 'dtc' / 'beam_test_330' / 'dtc_beam_check.json'
if chk.exists():
    bc = json.loads(chk.read_text())
    Em = np.array([float(k) for k in bc['energies']])
    rmed = np.array([v['range_median_mm'] for v in bc['energies'].values()])
    a.plot(Em, rmed, 's', ms=5, mfc='white', color='0.15', label='GATE: median stopping depth (pencil-beam test)')
for n in (41, 77):
    a.axhline(n * dg.PITCH, color='0.7', lw=0.8)
    a.text(32, n * dg.PITCH + 4, f'{n} layers', fontsize=8.5, color='0.35')
a.set_xlabel('energy entering the calorimeter, MeV')
a.set_ylabel('range, mm of calorimeter depth')
a.set_title('Range–energy relation of the calorimeter')
a.legend(fontsize=8.2, loc='lower right')
a = axes[1]
for f, ls, top in fits:
    sel = E <= top
    lab = f'CSDA − fit 50–{top}'
    a.plot(E[sel], (r - f['alpha_mm_per_MeV_p'] * E ** f['p'])[sel], 'o' + ls, ms=3.5, color=REAL, label=lab)
if chk.exists():
    f = mats['bragg_kleeman_340']
    a.plot(Em, rmed - f['alpha_mm_per_MeV_p'] * Em ** f['p'], 's', ms=5, mfc='white', color='0.15',
           label='GATE median − fit 50–340')
a.axhline(0, color='0.6', lw=0.8)
a.axvspan(30, 50, color='0.93', lw=0)
a.set_xlabel('energy, MeV')
a.set_ylabel('mm of depth')
a.set_title('Residuals: a power law is good to about 1 mm up to 260 MeV, 3 mm to 340')
a.legend(fontsize=8.5)
style.save(fig, out / 'dtc_range_energy.png')

# ---- 3. how deep the protons of real views reach
runs = [('dtc41_head', "head, 230 MeV, Bergen's 41 layers", OURS),
        ('dtc41_abdomen', 'abdomen, 250 MeV, 41 layers (dropped)', '0.65'),
        ('dtc77w16_abdomen330', 'abdomen, 330 MeV, 77 layers, 480 mm wide', REAL)]
fig, a = plt.subplots(figsize=(7.6, 4.0))
summary = {}
for run, lab, col in runs:
    files = sorted((data / 'gate' / 'dtc' / 'timing_parallel' / run).glob('view_*.npz'))
    if not files:
        continue
    counts = None
    n_last = n_tot = 0
    for f in files:
        d = np.load(f)
        pm = d['hit_primary'] == 1
        ev = d['hit_event'][pm].astype(np.int64)
        lay = d['hit_plane'][pm].astype(np.int64) // 2
        deep = np.full(int(d['event'].max()) + 1, -1)
        np.maximum.at(deep, ev, lay)
        deep = deep[np.isin(np.arange(len(deep)), d['event'])]
        deep = deep[deep >= 0]
        n_layers = int(deep.max())
        hst = np.bincount(deep, minlength=80)
        counts = hst if counts is None else counts + hst[:len(counts)]
        n_last += int((deep == n_layers).sum())
        n_tot += len(deep)
    x = np.arange(len(counts)) - 1                                    # calorimeter layer index (-1, 0 = trackers)
    a.step(x, counts / counts.sum(), where='mid', color=col, label=f'{lab} ({len(files)} views)')
    summary[run] = {'views': len(files), 'fraction_in_last_layer': n_last / max(n_tot, 1)}
for n in (41, 77):
    a.axvline(n - 1, color='0.7', lw=0.8)
a.set_xlabel('deepest calorimeter layer reached by a primary proton (−1, 0: trackers)')
a.set_ylabel('fraction of protons')
a.set_title('Where the protons stop: a peak at the last layer means they pass through')
a.legend(fontsize=8.5, loc='upper left')
style.save(fig, out / 'dtc_depth_coverage.png')
(data / 'gate' / 'dtc' / 'depth_coverage.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
