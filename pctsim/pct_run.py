"""Run a GATE pCT scan as one GATE process per view (GATE 10.0.3 crashes on a second run with the treatment-plan
source), several at once, and convert each view to a compact table as soon as it finishes.

A view is started only while the free commit stays above FLOOR_GB plus one process's commit (PROCESS_GB), so the
machine keeps its 10 GB floor; processes run below normal priority. Each GATE process runs with one BLAS and OpenMP
thread: otherwise numpy's and scipy's OpenBLAS each reserve a buffer per core at import (1.6 GB of commit on 24 cores,
for a single-threaded process). Views already converted are skipped, so a stopped run
resumes. Per view, OUT_DIR/view_XXX.npz holds every primary proton that crossed the front plane:
    event   uint32 [N]      EventID (spots are fired in EventID order)
    front   float32 [N, 6]  x, z (mm), direction x, y, z, kinetic energy (MeV) entering the front plane (its y is in
                            info_XXX.json: -165 mm for the head box)
    rear    float32 [N, 6]  the same entering the rear plane (y in info_XXX.json: +225.35 mm for the head); NaN where
                            the proton never got there
    angle   float           the phantom's rotation about +z for this view, degrees
Progress goes to OUT_DIR/progress.log. ENERGY (MeV, default 230) and X_HALF (the spot grid's half-width in x, mm,
default 126) go to pct_scan.py.

Usage: python -I pct_run.py SCAN OUT_DIR WORK_DIR PER_VIEW Z_CENTRE Z_HALF SEED_BASE WORKERS PHANTOM
                            [N_VIEWS [ENERGY X_HALF]]
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import uproot

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import FLOOR_GB, LOW_PRIORITY, ONE_THREAD, free_gb  # noqa: E402

PROCESS_GB = 1.0
PY = sys.executable


def convert(work, tag, out_path, angle):
    tables = {}
    for plane in ('front', 'rear'):
        a = uproot.open(work / f'{plane}_{tag}.root')[f'ps_{plane}'].arrays(library='np')
        keep = a['TrackID'] == 1
        ev = a['EventID'][keep]
        cols = np.stack([a['PrePosition_X'][keep], a['PrePosition_Z'][keep], a['PreDirection_X'][keep],
                         a['PreDirection_Y'][keep], a['PreDirection_Z'][keep], a['KineticEnergy'][keep]], axis=1)
        ev, first = np.unique(ev, return_index=True)          # a proton that re-enters a plane: keep its first entry
        tables[plane] = (ev, cols[first].astype(np.float32))
    ev, front = tables['front']
    rev, rcols = tables['rear']
    rear = np.full_like(front, np.nan)
    _, i, j = np.intersect1d(ev, rev, return_indices=True)
    rear[i] = rcols[j]
    with open(out_path, 'wb') as f:
        np.savez(f, event=ev.astype(np.uint32), front=front, rear=rear, angle=np.float64(angle))
    return len(ev), int(np.isfinite(rear[:, 0]).sum())


if __name__ == '__main__':
    scan, out_dir, work_root = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    per_view, z_centre, z_half = int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    seed_base, workers, phantom = int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
    n_views = int(sys.argv[10]) if len(sys.argv) > 10 else 360
    energy = float(sys.argv[11]) if len(sys.argv) > 11 else 230.0
    x_half = float(sys.argv[12]) if len(sys.argv) > 12 else 126.0
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / 'progress.log', 'a', buffering=1)
    todo = [v for v in range(n_views) if not (out_dir / f'view_{v:03d}.npz').exists()]
    log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} start {scan}: {len(todo)} of {n_views} views to do, '
              f'{per_view} protons each, {energy:g} MeV, spots x +- {x_half:g} mm, z {z_centre} +- {z_half} mm, '
              f'{workers} workers\n')
    running = {}
    t_start = time.time()
    done = 0
    while todo or running:
        # collect finished views
        for v, (proc, work, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[v]
            tag = f'{scan}{v:03d}'
            if proc.returncode != 0:
                log.write(f'{time.strftime("%H:%M:%S")} view {v} FAILED (exit {proc.returncode}); requeued\n')
                todo.append(v)
                continue
            info = json.loads((work / f'info_{tag}.json').read_text())
            n_front, n_rear = convert(work, tag, out_dir / f'view_{v:03d}.npz', info['angles'][0])
            shutil.copy(work / f'info_{tag}.json', out_dir / f'info_{v:03d}.json')
            shutil.rmtree(work)
            done += 1
            el = time.time() - t_start
            eta = el / done * (len(todo) + len(running))
            ram, commit = free_gb()
            if n_front < 0.99 * per_view:      # about 0.05% are lost in the air before the front plane; more is a fault
                log.write(f'WARNING view {v}: only {n_front / per_view:.4f} of the protons crossed the front plane\n')
            log.write(f'{time.strftime("%H:%M:%S")} view {v:3d} done in {time.time() - t0:5.0f} s: front {n_front}, '
                      f'rear {n_rear}; {done} done, {len(todo)} queued, {len(running)} running; ETA {eta / 3600:.2f} h; '
                      f'free RAM {ram:.1f} GB, commit {commit:.1f} GB\n')
        # start new views while memory allows
        while todo and len(running) < workers:
            ram, commit = free_gb()
            if commit - PROCESS_GB < FLOOR_GB or ram - PROCESS_GB < FLOOR_GB:
                break
            v = todo.pop(0)
            tag = f'{scan}{v:03d}'
            work = work_root / tag
            if work.exists():
                shutil.rmtree(work)
            cmd = [PY, '-I', str(HERE / 'pct_scan.py'), str(work), tag, str(v), '1', str(n_views), str(per_view),
                   str(z_centre), str(z_half), str(seed_base + v), phantom, str(energy), str(x_half)]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(work_root / f'{tag}.err', 'w'),
                                    env={**os.environ, **ONE_THREAD}, **LOW_PRIORITY)
            running[v] = (proc, work, time.time())
            time.sleep(3)                                      # stagger the start-ups, so the commit check sees each
        time.sleep(2)
    log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} finished {scan}: {done} views in '
              f'{(time.time() - t_start) / 3600:.2f} h\n')
