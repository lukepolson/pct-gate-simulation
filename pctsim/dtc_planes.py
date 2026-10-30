"""Turn a DTC scan (dtc_run.py) into two scans in the ideal-plane form that io.PCT.read_gate reads (view_XXX.npz with
'event', 'front', 'rear', 'angle'; info_XXX.json with 'front_y' and 'rear_y'), both with the rear plane at the scan's
exit plane, 3 mm before the first tracker:

  TRUTH_DIR:    rear = the ideal exit plane (position, direction, kinetic energy in MeV): the same protons with no
                detector, for a reconstruction with the 330 MeV water calibration (pct_calib.py).
  MEASURED_DIR: rear = what the DTC measures (dtc_range.py): position and direction from the two tracker layers, and
                in place of the energy the range in the calorimeter (mm WET), NaN for protons the detector loses or
                filters; read with the DTC's own calibration (dtc_calib.py).

Each view is processed in blocks of protons, WORKERS views at once (one BLAS thread, below normal priority). Writes
loss counts per view to MEASURED_DIR/measure_counts.json.

Usage: python -I dtc_planes.py SCAN_DIR TRUTH_DIR MEASURED_DIR MATERIALS_JSON WORKERS
"""
import os

for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtc_range as dr  # noqa: E402

BLOCK = 200_000                  # protons per block
HIT_KEYS = ('hit_event', 'hit_plane', 'hit_col', 'hit_row', 'hit_edep', 'hit_primary')


def _low_priority():
    from common import lower_this_process
    lower_this_process()


def one_view(path, truth_dir, measured_dir, materials):
    path = pathlib.Path(path)
    v = path.stem.split('_')[1]
    info = json.loads((path.parent / f'info_{v}.json').read_text())
    stack = dr.Stack(info['dtc'], materials)
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    events = d['event']
    rear = np.full((len(events), 6), np.nan, np.float32)
    counts = {}
    for s in range(0, len(events), BLOCK):
        ev = events[s:s + BLOCK]
        a = np.searchsorted(d['hit_event'], ev[0], side='left')       # hits are sorted by event
        b = np.searchsorted(d['hit_event'], ev[-1], side='right')
        block = {k: d[k][a:b] for k in HIT_KEYS}
        rear[s:s + BLOCK], c = dr.measure(block, ev, stack, info['exit_y'])
        for k, n in c.items():
            counts[k] = counts.get(k, 0) + n
    for out, r, source in ((truth_dir, d['exit'], 'truth: the ideal exit plane, kinetic energy in MeV'),
                           (measured_dir, rear, 'measured by the DTC (dtc_range.py); column 5 is the range, mm WET')):
        np.savez(out / path.name, event=events, front=d['front'], rear=r, angle=d['angle'])
        (out / f'info_{v}.json').write_text(json.dumps({**info, 'rear_y': info['exit_y'], 'rear_source': source},
                                                       indent=1))
    counts['exit_truth'] = int(np.isfinite(d['exit'][:, 0]).sum())
    return int(v), counts


if __name__ == '__main__':
    scan, truth_dir, measured_dir = (pathlib.Path(p) for p in sys.argv[1:4])
    materials = json.loads(pathlib.Path(sys.argv[4]).read_text())
    workers = int(sys.argv[5])
    for p in (truth_dir, measured_dir):
        p.mkdir(parents=True, exist_ok=True)
    files = sorted(scan.glob('view_*.npz'))
    todo = [f for f in files if not ((measured_dir / f.name).exists() and (truth_dir / f.name).exists())]
    counts_path = measured_dir / 'measure_counts.json'
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_low_priority) as pool:
        futures = [pool.submit(one_view, str(f), truth_dir, measured_dir, materials) for f in todo]
        for k, fu in enumerate(futures, 1):
            v, c = fu.result()
            counts[f'{v:03d}'] = c
            if k % 20 == 0 or k == len(futures):
                counts_path.write_text(json.dumps(dict(sorted(counts.items())), indent=1))
                print(f'{k}/{len(futures)} views, {time.time() - t0:.0f} s; view {v}: {c}', flush=True)
    tot = {k: sum(c[k] for c in counts.values()) for k in next(iter(counts.values()))}
    print('all views:', tot, {k: round(tot[k] / tot['protons'], 4) for k in tot})
