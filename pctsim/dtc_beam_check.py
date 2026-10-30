"""The DTC's response to pencil beams (dtc_beam_test.py output): per-layer energy deposits (Bragg curves), the MC truth
range against the CSDA range of dtc_wet.py, the Bragg-Kleeman range estimate per proton (Pettersen et al. 2019: a
least-squares fit of each track's energy loss per layer), its error against the truth, and the energy-deposit spectra
in the 25 um epitaxial layers. Only primaries that stop by ionisation (not in a nuclear interaction) are scored for
range.

Fit per proton: edep_l = A (R - z_l)^(1/p - 1) over its calorimeter layers, p from the stack's range-energy fit
(dtc_materials.json), A in closed form for each R on a 0.01 mm grid between its deepest layer and the next.

Writes TEST_DIR/dtc_beam_check.json and the figures to OUT_DIR.

Usage: python -I dtc_beam_check.py TEST_DIR MATERIALS_JSON OUT_DIR [THEME]
  THEME: light (default) or dark (black background, for the results pages)
"""
import json
import pathlib
import sys

import numpy as np
import uproot

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dtc_geometry as dg  # noqa: E402
from dtc_convert import COLS, clusters  # noqa: E402
import style  # noqa: E402
from style import OURS, REAL, REF, plt  # noqa: E402

test, mat_path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
INK, SEQ = '0.3', style.SEQ
if len(sys.argv) > 4 and sys.argv[4] == 'dark':
    OURS, REAL, REF, INK, SEQ = style.dark()
out.mkdir(parents=True, exist_ok=True)
info = json.loads((test / 'info.json').read_text())
mats = json.loads(mat_path.read_text())
plane_y = np.array(info['sensor_y_mm'])
n_layers_total = len(plane_y) // 2
z0 = dg.CAL_Y0 - dg.GAP / 2 - dg.CAL['carrier'][1]                  # front face of calorimeter layer 0
wet_per_mm = mats['layer_wet']['cal']['wet_mm'] / dg.PITCH
energies = info['energies_MeV']
fit_key = 'bragg_kleeman_340' if max(energies) > 260 else 'bragg_kleeman'      # the fit that covers these energies
p_bk = mats[fit_key]['p']
EX = 250.0 if 250.0 in energies else energies[len(energies) // 2]              # the energy of the examples
q = 1.0 / p_bk - 1.0
layer_z = (np.array(info['layer_y_mm']) - z0)                          # gap centre of each layer from z0
hits = uproot.open(test / 'hits.root')['hits'].arrays(COLS + ['RunID'], library='np')
stops = uproot.open(test / 'stops.root')['stops'].arrays(library='np')

def rsd(x):
    """A robust sd (1.4826 x the median absolute deviation): a few protons that stop early after a large-angle
    nuclear elastic scatter would otherwise dominate the plain sd."""
    x = np.asarray(x)[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


res = {'z0_mm': z0, 'wet_per_mm': wet_per_mm, 'p': p_bk, 'sd': 'robust: 1.4826 x MAD; bias: median',
       'energies': {}}
curves, examples, spectra = {}, {}, {}
for r, E in enumerate(energies):
    n = info['n_per_energy']
    sel = hits['RunID'] == r
    c = clusters({k: hits[k][sel] for k in COLS}, plane_y, info.get('n_chips', dg.N_CHIPS),
                 info.get('n_bands', dg.N_BANDS))
    prim = c['hit_primary'] == 1
    layer = c['hit_plane'].astype(int) // 2
    ed = np.zeros((n, n_layers_total))
    np.add.at(ed, (c['hit_event'][prim].astype(int), layer[prim]), c['hit_edep'][prim])
    # tracker 0, half A: the primary's own steps, and its cluster (its own steps plus its delta rays)
    raw0 = sel & (np.abs(hits['PostPosition_Y'] - plane_y[0]) < dg.EPI) & (hits['TrackID'] == 1)
    t0_own = np.zeros(n)
    np.add.at(t0_own, hits['EventID'][raw0].astype(int), hits['TotalEnergyDeposit'][raw0] * 1000.0)
    t0 = np.zeros(n)
    m0 = (c['hit_plane'] == 0) & prim
    np.add.at(t0, c['hit_event'][m0].astype(int), c['hit_edep'][m0])
    spectra[E] = (t0_own[t0_own > 0], t0[t0 > 0])
    s = (stops['RunID'] == r) & (stops['TrackID'] == 1)
    ev_s = stops['EventID'][s].astype(int)
    y_stop = np.full(n, np.nan)
    ion = np.zeros(n, bool)
    y_stop[ev_s] = stops['PostPosition_Y'][s]
    ion[ev_s] = stops['ProcessDefinedStep'][s] == 'hIoni'
    r_true = y_stop - z0
    cal = ed[:, 2:]
    zl = layer_z[2:]
    deepest = np.where(cal.any(axis=1), cal.shape[1] - 1 - np.argmax(cal[:, ::-1] > 0, axis=1), -1)
    ok = ion & (deepest >= 1) & (r_true > 0) & (deepest < cal.shape[1] - 1)
    r_hat = np.full(n, np.nan)
    for i in np.flatnonzero(ok):
        e = cal[i, :deepest[i] + 1]
        z = zl[:deepest[i] + 1]
        use = e > 0
        grid = z[-1] + np.arange(0.01, dg.PITCH + 1e-9, 0.01)
        f = (grid[:, None] - z[None, use]) ** q
        fe = f @ e[use]
        ff = (f * f).sum(axis=1)
        r_hat[i] = grid[np.argmin((e[use] ** 2).sum() - fe ** 2 / ff)]
    r_layer = zl[np.clip(deepest, 0, None)] + dg.PITCH / 2              # deepest layer + half a pitch
    err_bk = (r_hat - r_true)[ok] * wet_per_mm
    err_layer = (r_layer - r_true)[ok] * wet_per_mm
    strag = rsd(r_true[ion]) * wet_per_mm
    res['energies'][str(E)] = {
        'stopped_by_ionisation': float(ion.mean()), 'scored': int(ok.sum()),
        'range_mean_mm': float(np.nanmean(r_true[ion])), 'range_median_mm': float(np.nanmedian(r_true[ion])),
        'range_sd_wet_mm': float(strag),
        'csda_mm': mats['csda'].get(str(int(E)), {}).get('depth_mm'),
        'bk_error_mean_wet_mm': float(np.median(err_bk)), 'bk_error_sd_wet_mm': rsd(err_bk),
        'layer_error_mean_wet_mm': float(np.median(err_layer)), 'layer_error_sd_wet_mm': rsd(err_layer),
        'total_sd_wet_mm': rsd(r_hat[ok] * wet_per_mm),
        't0_primary_edep_keV_median': float(np.median(t0[t0 > 0])), 'deepest_layer_mean': float(deepest[ok].mean())}
    keep = ion & (deepest >= 0)
    curves[E] = (zl, cal[keep].mean(axis=0), cal[keep].std(axis=0))
    if E == EX:
        pick = np.flatnonzero(ok)[:3]
        examples = {'zl': zl, 'rows': [(cal[i, :deepest[i] + 1], r_hat[i], r_true[i]) for i in pick]}
        res['err_hist'] = {'energy': EX, 'bk': err_bk.tolist(), 'layer': err_layer.tolist()}
    v = res['energies'][str(E)]
    print(f'{E:5.0f} MeV: ionisation stops {ion.mean():.3f}, range median {v["range_median_mm"]:7.2f} mm '
          f'(CSDA {v["csda_mm"]:.2f}), straggling {strag:.2f} WET mm, BK error {v["bk_error_mean_wet_mm"]:+.2f} +- '
          f'{v["bk_error_sd_wet_mm"]:.2f}, deepest-layer error {v["layer_error_mean_wet_mm"]:+.2f} +- '
          f'{v["layer_error_sd_wet_mm"]:.2f}, total {v["total_sd_wet_mm"]:.2f}')
(test / 'dtc_beam_check.json').write_text(json.dumps(res, indent=1))

# ---- Bragg curves in the calorimeter
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.35, 1]})
a = axes[0]
for k, E in enumerate(energies):
    zl, mean, sd = curves[E]
    last = np.flatnonzero(mean > 0)
    last = last[-1] + 1 if len(last) else 0
    a.plot(zl[:last] * wet_per_mm, mean[:last], '.-', ms=3.5, lw=1.0, color=SEQ(k / max(1, len(energies) - 1)),
           label=f'{E:g} MeV')
a.set_xlabel('depth in the calorimeter, mm WET (from the front of layer 0)')
a.set_ylabel('mean primary energy per layer, keV (25 µm Si)')
a.set_title('Energy deposited per layer: the Bragg curve as the DTC samples it')
a.legend(ncol=2)
a = axes[1]
zl = examples['zl']
for k, (e, rh, rt) in enumerate(examples['rows']):
    col = [OURS, REAL, INK][k]
    z = zl[:len(e)]
    a.plot(z * wet_per_mm, e, 'o', ms=4, color=col)
    zz = np.linspace(0, rh - 0.02, 300)
    use = e > 0
    f = (rh - z[use]) ** q
    A = (f @ e[use]) / (f @ f)
    a.plot(zz * wet_per_mm, A * (rh - zz) ** q, '-', lw=1.0, color=col)
    a.axvline(rt * wet_per_mm, color=col, lw=0.8, ls=':')
rh_all = [rh for _, rh, _ in examples['rows']]
a.set_xlim(min(rh_all) * wet_per_mm - 120, max(rh_all) * wet_per_mm + 12)
e_max = max(float(e.max()) for e, _, _ in examples['rows'])
a.set_ylim(0, 1.35 * e_max)
a.set_xlabel('depth, mm WET (the last 120 mm before the stop)')
a.set_ylabel('energy in the layer, keV')
a.set_title(f'Three {EX:g} MeV protons: Bragg–Kleeman fits (lines),\ntrue stopping depth (dotted); p = {p_bk:.3f}')
style.save(fig, out / 'dtc_bragg_curves.png')

# ---- range accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
a = axes[0]
h = res['err_hist']
bins = np.linspace(-12, 12, 97)
a.hist(h['layer'], bins=bins, histtype='step', color=REF, lw=1.3, label='deepest layer + half a pitch')
a.hist(h['bk'], bins=bins, histtype='step', color=REAL, lw=1.6, label='Bragg–Kleeman fit')
a.set_xlabel('range estimate − true range, mm WET')
a.set_ylabel('protons')
ex = res['energies'][str(EX)]
a.set_title(f'{EX:g} MeV: fit error {ex["bk_error_mean_wet_mm"]:+.2f} ± {ex["bk_error_sd_wet_mm"]:.2f} mm WET, '
            f'layer error ± {ex["layer_error_sd_wet_mm"]:.2f}')
a.legend()
a = axes[1]
Es = np.array(energies)
g = [res['energies'][str(E)] for E in energies]
a.plot(Es, [v['range_sd_wet_mm'] for v in g], 'o-', color=OURS, label='range straggling (MC truth)')
a.plot(Es, [v['bk_error_sd_wet_mm'] for v in g], 's-', color=REAL, label='detector: Bragg–Kleeman fit')
a.plot(Es, [v['layer_error_sd_wet_mm'] for v in g], '^-', color=REF, label='detector: deepest layer')
a.axhline(2.0, color=REAL, ls='--', lw=0.9)
a.text(Es[-1], 1.82, 'Pettersen 2019: about 2 mm WET intrinsic', color=REAL, fontsize=9, ha='right', va='top')
a.set_xlabel('beam energy at the detector, MeV')
a.set_ylabel('sd, mm WET')
a.set_ylim(0, None)
a.set_title('Range error per proton (robust sd): the detector adds to straggling')
a.legend(loc='upper left')
style.save(fig, out / 'dtc_range_accuracy.png')

# ---- energy deposit spectra in a 25 um layer
fig, a = plt.subplots(figsize=(6.4, 3.8))
bins = np.linspace(0, 60, 121)
for E, col in ((EX, REAL), (100.0, OURS)):
    if E in spectra:
        prim, tot = spectra[E]
        a.hist(prim, bins=bins, histtype='step', color=col, lw=1.6, label=f'{E:g} MeV: the proton’s own steps')
        a.hist(tot, bins=bins, histtype='step', color=col, lw=0.9, ls='--', label=f'{E:g} MeV: its cluster, delta rays added')
a.set_xlabel('energy deposited in the first tracker’s 25 µm epitaxial layer, keV')
a.set_ylabel('protons')
a.set_title('Landau spectra in one layer: the signal the cluster size will encode')
a.legend()
style.save(fig, out / 'dtc_edep_spectra.png')
