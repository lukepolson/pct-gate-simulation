"""Give a DTC scan the entry a real Bergen-type scanner has: no front tracker, so each proton enters along the nominal
central ray of the pencil-beam spot it was delivered in (position and direction from the spot's place in the plan
and the steering magnets), not where it truly entered.

Which spot each proton came from is what the beam delivery log gives a real scanner. GATE fires the spots in plan
order (sorted_spot_generation) but draws each spot's number of protons at random, so the log is rebuilt from the
event order: the boundary between spot k and k + 1 is the change point, within 700-1700 events of the last one, that
best splits the protons' true entry points between the two spots' rays (Gaussian, SIGMA_MM). Consecutive spots are
7 mm apart and the spot is 3.6 mm wide at the entry plane, so a boundary is found to within an event or two. The
true entry points are used for nothing else.

Writes OUT_DIR/view_XXX.npz (event, front = the spot's ray at the entry plane: x, z, direction, the nominal energy;
rear = the DTC measurement from MEASURED_DIR; angle; spot, the spot of each proton), info_XXX.json, and
spot_entry.json: per view, the protons per spot and the entry error (true minus the spot's ray) the scanner cannot see.

Usage: python -I dtc_spots.py SCAN_DIR MEASURED_DIR OUT_DIR WORKERS
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

SIGMA_MM = 3.6                         # the spot's sd at the entry plane (measured on view 0: 3.6-3.7 mm)
D_MAGNET_X, D_MAGNET_Z = 6780.0, 7430.0    # steering magnets to isocentre, mm (the scan's beam model)


def _low_priority():
    from common import lower_this_process
    lower_this_process()


def spot_rays(info):
    """Each spot's central ray at the entry plane, in plan order (z outer, x inner): position (x, z) and the unit
    direction, from the spot's place at the isocentre and the steering magnets' distances."""
    xs, zs = np.array(info['spots_x']), np.array(info['spots_z'])
    X, Z = np.meshgrid(xs, zs)                         # rows: z, columns: x
    X, Z = X.ravel(), Z.ravel()
    fy = info['front_y']
    pos = np.stack([X * (D_MAGNET_X + fy) / D_MAGNET_X, Z * (D_MAGNET_Z + fy) / D_MAGNET_Z], axis=1)
    d = np.stack([X / D_MAGNET_X, np.ones_like(X), Z / D_MAGNET_Z], axis=1)
    return pos, d / np.linalg.norm(d, axis=1, keepdims=True)


def boundaries(p, centres, sigma=SIGMA_MM, lo=700, hi=1700):
    """Start index of every spot in the event-ordered entry points p [N, 2], spot by spot (the change point)."""
    n, k_spots = len(p), len(centres)
    b = np.zeros(k_spots + 1, np.int64)
    b[-1] = n
    for k in range(k_spots - 1):
        w0, w1 = min(b[k] + lo, n), min(b[k] + hi, n)
        q = p[w0:w1]
        llr = (((q - centres[k]) ** 2).sum(1) - ((q - centres[k + 1]) ** 2).sum(1)) / (2 * sigma ** 2)
        llr = np.clip(llr, -20, 20)
        tail = np.cumsum(llr[::-1])[::-1]               # sum of llr from each b to the window's end
        b[k + 1] = w0 + int(np.argmax(np.append(tail, 0.0)))
    return b


def one_view(scan_path, measured_dir, out_dir):
    scan_path = pathlib.Path(scan_path)
    v = scan_path.stem.split('_')[1]
    info = json.loads((pathlib.Path(measured_dir) / f'info_{v}.json').read_text())
    with np.load(scan_path) as z:
        event, true_front = z['event'], z['front']
    with np.load(pathlib.Path(measured_dir) / scan_path.name) as z:
        assert np.array_equal(z['event'], event)
        rear, angle = z['rear'], z['angle']
    pos, d = spot_rays(info)
    b = boundaries(true_front[:, :2].astype(np.float64), pos)
    spot = np.repeat(np.arange(len(pos)), np.diff(b)).astype(np.uint16)
    front = np.empty_like(true_front)
    front[:, :2], front[:, 2:5], front[:, 5] = pos[spot], d[spot], info['energy_MeV']
    np.savez(pathlib.Path(out_dir) / scan_path.name, event=event, front=front, rear=rear, angle=angle, spot=spot)
    (pathlib.Path(out_dir) / f'info_{v}.json').write_text(json.dumps(
        {**info, 'front_source': "the spot's nominal central ray at the entry plane (spot from the event order, as a "
                                 "beam delivery log gives it); energy the nominal beam energy"}, indent=1))
    counts = np.diff(b)
    err = true_front[:, :2] - front[:, :2]
    derr = 1000 * (true_front[:, 2:5:2] / true_front[:, 3:4] - front[:, 2:5:2] / front[:, 3:4])    # mrad
    per_spot_mean = np.array([err[a:c].mean(0) for a, c in zip(b[:-1], b[1:])])
    return int(v), {'protons_per_spot': [int(counts.min()), float(counts.mean()), int(counts.max()), float(counts.std())],
                    'entry_error_sd_mm': err.std(0).round(3).tolist(), 'entry_error_mean_mm': err.mean(0).round(4).tolist(),
                    'direction_error_sd_mrad': derr.std(0).round(3).tolist(),
                    'per_spot_mean_error_max_mm': float(np.abs(per_spot_mean).max()),
                    'entry_error_hist_mm': np.histogram(err[:, 0], np.arange(-15, 15.01, 0.25))[0].tolist(),
                    'direction_error_hist_mrad': np.histogram(derr[:, 0], np.arange(-12, 12.01, 0.1))[0].tolist()}


if __name__ == '__main__':
    scan, measured, out = (pathlib.Path(p) for p in sys.argv[1:4])
    workers = int(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(scan.glob('view_*.npz'))
    todo = [f for f in files if not (out / f.name).exists()]
    path = out / 'spot_entry.json'
    res = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_low_priority) as pool:
        futures = [pool.submit(one_view, str(f), str(measured), str(out)) for f in todo]
        for k, fu in enumerate(futures, 1):
            v, r = fu.result()
            res[f'{v:03d}'] = r
            if k % 30 == 0 or k == len(futures):
                path.write_text(json.dumps(dict(sorted(res.items()))))
                print(f'{k}/{len(futures)} views, {time.time() - t0:.0f} s; view {v}: per spot {r["protons_per_spot"]}, '
                      f'entry error sd {r["entry_error_sd_mm"]} mm, direction {r["direction_error_sd_mrad"]} mrad, '
                      f'worst spot mean {r["per_spot_mean_error_max_mm"]:.3f} mm', flush=True)
