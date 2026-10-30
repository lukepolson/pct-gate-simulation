"""Run a proton CT simulation from one config file, stage by stage: from a CT (or a 511 keV attenuation map) to
PyTomography's list mode file.

  phantom    the CT's voxels in HU, cropped and couch-free (pctsim/phantom_from_ct.py), or a mu-map turned into HU
             (pctsim/phantom_from_mumap.py)
  materials  HU to GATE's Schneider tissue materials, their label image and each voxel's true RSP (pct_materials.py)
  detector   the DTC's materials: stopping powers, each layer's water-equivalent thickness (dtc_wet.py)
  calibrate  WEPL calibrations with water slabs: the ideal exit plane (pct_calib.py) and the DTC (dtc_calib.py)
  scan       the scan, one GATE process per view, several at once, each view converted to a table (dtc_run.py)
  measure    what the DTC measures of each proton (dtc_planes.py), and the pencil beam's spot as its entry (dtc_spots.py)
  listmode   cuts, entries and the list mode file (gate_to_listmode.py)

Each stage is skipped when its output exists, and the scan and calibrations carry on where they stopped, so a run can
be stopped and started again. The planes and the detector's place follow from the phantom as in the scan scripts: the
entry plane 15 mm outside the circle the phantom's box sweeps as it turns, the exit plane 3 mm before the first
tracker, the DTC moved back from Bergen's place when the phantom would reach it.

Usage: python run.py CONFIG.json [STAGE ...]      (all stages by default)
"""
import json
import math
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE / 'pctsim'
STAGES = ('phantom', 'materials', 'detector', 'calibrate', 'scan', 'measure', 'listmode')
REAR_Y, PLANE_GAP = 225.35, 15.0        # Bergen's first tracker (mm from the isocentre); the planes' clearance


def swept_radius(mhd):
    """Half-diagonal of the phantom's box in x and y (mm), as pct_scan.swept_radius."""
    h = dict(ln.split(' = ', 1) for ln in pathlib.Path(mhd).read_text().splitlines() if ' = ' in ln)
    n = [float(v) for v in h['DimSize'].split()]
    d = [float(v) for v in h['ElementSpacing'].split()]
    return math.hypot(n[0] * d[0] / 2, n[1] * d[1] / 2)


def run(python, script, *args, log=None):
    cmd = [python, str(SIM / script), *[str(a) for a in args]]
    print('$', ' '.join(cmd), flush=True)
    t0 = time.time()
    with open(log if log else os.devnull, 'a') as out:
        r = subprocess.run(cmd, stdout=out if log else None, stderr=subprocess.STDOUT if log else None)
    if r.returncode:
        sys.exit(f'{script} failed (exit {r.returncode})' + (f'; see {log}' if log else ''))
    print(f'  done in {time.time() - t0:.0f} s', flush=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = pathlib.Path(sys.argv[1])
    c = json.loads(path.read_text())
    stages = sys.argv[2:] or STAGES
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f'unknown stages {bad}; the stages are {STAGES}')
    root = (path.parent / c['out']).resolve() if not pathlib.Path(c['out']).is_absolute() else pathlib.Path(c['out'])
    root.mkdir(parents=True, exist_ok=True)
    gate_py, analysis_py = c['python']['gate'], c['python']['analysis']
    name, beam, det, scan, cal = c['name'], c['beam'], c['detector'], c['scan'], c['calibration']
    workers = int(c.get('workers', 4))

    def where(p):                       # paths in the config are relative to the config file
        return (path.parent / p).resolve() if not pathlib.Path(p).is_absolute() else pathlib.Path(p)

    d_phantom, d_det, d_scan = root / 'phantom', root / 'detector', root / f'scan_{name}'
    d_cal_ideal, d_cal_dtc = root / 'calibration_ideal', root / 'calibration_dtc'
    d_truth, d_measured, d_spot = root / f'{name}_truth', root / f'{name}_measured', root / f'{name}_spot'
    mhd = d_phantom / f'{name}_hu.mhd'
    materials_json = d_det / 'dtc_materials.json'
    listmode = root / f'{name}.h5'

    if 'phantom' in stages and not mhd.exists():
        p = c['phantom']
        if p['from'] == 'ct':
            args = [where(p['ct']), d_phantom, name]
            args += ['--slices', *p['slices']] if 'slices' in p else ['--around', *[where(m) for m in p['around']],
                                                                     '--margin', p.get('margin', 25.0)]
            if p.get('masks'):
                args += ['--masks', *[f'{k}={where(v)}' for k, v in p['masks'].items()]]
            run(analysis_py, 'phantom_from_ct.py', *args)
        else:
            data = subprocess.run([gate_py, '-c', 'import opengate.utility as u; print(u.get_data_folder())'],
                                  capture_output=True, text=True).stdout.strip().splitlines()[-1]
            run(analysis_py, 'phantom_from_mumap.py', where(p['mumap']),
                pathlib.Path(data) / 'Schneider2000MaterialsTable.txt',
                pathlib.Path(data) / 'Schneider2000DensitiesTable.txt', d_phantom, p.get('voxel', 1.0))
            for suffix in ('hu.mhd', 'hu.raw'):                           # the builder names its files head_*
                src = d_phantom / f'head_{suffix}'
                if src.exists() and name != 'head':
                    src.replace(d_phantom / f'{name}_{suffix}')
            if name != 'head':
                text = mhd.read_text().replace('head_hu.raw', f'{name}_hu.raw')
                mhd.write_text(text)
    if not mhd.exists():
        sys.exit(f'no phantom yet ({mhd}): run the phantom stage first')
    rear_y = max(REAR_Y, swept_radius(mhd) + PLANE_GAP)
    shift, exit_y = rear_y - REAR_Y, rear_y - 3.0
    print(f'{name}: entry plane at y = {-(swept_radius(mhd) + PLANE_GAP):.2f} mm, exit plane at {exit_y:.2f} mm, '
          f'DTC moved back {shift:.2f} mm', flush=True)

    if 'materials' in stages and not (d_phantom / f'{name}_rsp.npz').exists():
        run(gate_py, 'pct_materials.py', mhd, d_phantom, name, log=d_phantom / 'materials.log')
    if 'detector' in stages and not materials_json.exists():
        run(gate_py, 'dtc_wet.py', d_det, log=root / 'detector.log')
    if 'calibrate' in stages:
        if not (d_cal_ideal / 'calib.npz').exists():
            run(gate_py, 'pct_calib.py', d_cal_ideal, cal['protons_per_thickness'], beam['energy'], exit_y,
                cal['max_thickness'], workers, log=root / 'calibration_ideal.log')
        if not (d_cal_dtc / 'calib.npz').exists():
            run(gate_py, 'dtc_calib.py', d_cal_dtc, cal['protons_per_thickness'], beam['energy'], shift,
                cal['max_thickness'], workers, det['layers'], det['chips'], det['bands'], materials_json,
                log=root / 'calibration_dtc.log')
    if 'scan' in stages:
        n_views = int(scan['views'])
        if len(list(d_scan.glob('view_*.npz'))) < n_views:
            run(gate_py, 'dtc_run.py', name, d_scan, root / f'work_{name}', scan['protons_per_view'],
                beam['z_centre'], beam['z_half'], scan['seed'], workers, mhd, n_views, beam['energy'],
                beam['x_half'], det['layers'], det['chips'], det['bands'], 0, n_views)
    if 'measure' in stages:
        if len(list(d_measured.glob('view_*.npz'))) < int(scan['views']):
            run(gate_py, 'dtc_planes.py', d_scan, d_truth, d_measured, materials_json, workers)
        if len(list(d_spot.glob('view_*.npz'))) < int(scan['views']):
            run(gate_py, 'dtc_spots.py', d_scan, d_measured, d_spot, workers)
    if 'listmode' in stages and not listmode.exists():
        e = c.get('entries', {'mode': 'tracker'})
        if e['mode'] == 'tracker':                  # the true entry, as a front tracker would measure it
            args = [d_measured, d_cal_dtc / 'calib.npz', listmode]
        elif e['mode'] in ('spot', 'likely'):       # no front tracker: the spot's ray, or the most likely entry
            args = [d_spot, d_cal_dtc / 'calib.npz', listmode, '--spot-sd', e['spot_sd'],
                    '--divergence', e['divergence']]
            if e['mode'] == 'likely':
                args += ['--likely', d_cal_ideal / 'calib.npz', '--exit-angle-sd', e['exit_angle_sd'],
                         '--scale', e['scale']]
        else:
            sys.exit(f"entries.mode is 'tracker', 'spot' or 'likely', not {e['mode']!r}")
        run(analysis_py, 'gate_to_listmode.py', *args)
    print('finished:', ', '.join(stages), flush=True)


if __name__ == '__main__':
    main()
