"""Run a GATE pCT scan with the full Bergen DTC (dtc_scan.py) as one GATE process per view, several at once, each view
converted to its compact table (dtc_convert.py) by its own process once GATE finishes, then its raw files deleted.
The runner of pct_run.py: the same memory check (a 10 GB floor on free RAM and commit), below-normal priority, one
BLAS thread per process, resumable (converted views are skipped).

A GATE process stays near 0.5 GB, but a convert of a full 330 MeV view peaks above 2.2 GB, and views started together
finish together: 20 converts at once took free commit to 8 GB (29 Oct). So at most MAX_CONVERT converts run at once,
each only when CONVERT_GB more still leaves the floor, and a waiting convert goes before any new GATE run.

On a restart, GATE processes of this scan that are still running (left by a stopped runner) are adopted and waited
for, and views whose GATE run finished but were never converted are converted, so no finished GATE work is lost.

OUT_DIR/view_XXX.npz per view (see dtc_convert.py), info_XXX.json, and progress.log, which also logs each view's GATE
time, so a short run doubles as a timing test.

Usage: python -I dtc_run.py SCAN OUT_DIR WORK_DIR PER_VIEW Z_CENTRE Z_HALF SEED_BASE WORKERS PHANTOM N_VIEWS ENERGY
                            X_HALF N_LAYERS N_CHIPS N_BANDS [FIRST_VIEW [N_VIEWS_TOTAL]]
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import psutil

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import FLOOR_GB, LOW_PRIORITY, ONE_THREAD, free_gb  # noqa: E402

PROCESS_GB = 1.5                 # a GATE process (measured 0.5-0.65 GB)
CONVERT_GB = 2.8                 # a convert of a 330 MeV view, 74 M raw hits (measured peak 2.2 GB part-way)
MAX_CONVERT = 3
PY = sys.executable


class Adopted:
    """A GATE process left running by an earlier runner, polled like a Popen: it succeeded if it wrote its info file."""

    def __init__(self, proc, info):
        self.proc, self.info, self.returncode = proc, info, None

    def poll(self):
        try:
            if self.proc.is_running() and self.proc.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            pass
        self.returncode = 0 if self.info.exists() else 1
        return self.returncode


def running_gate(work_root, scan):
    """GATE processes of this scan already running: view -> psutil.Process."""
    found = {}
    for p in psutil.process_iter(['cmdline']):
        cmd = p.info['cmdline'] or []
        i = next((k for k, c in enumerate(cmd) if c.endswith('dtc_scan.py')), None)
        if i is None or len(cmd) < i + 4:
            continue
        work, tag, v = pathlib.Path(cmd[i + 1]), cmd[i + 2], int(cmd[i + 3])
        if tag == f'{scan}{v:03d}' and work.name == tag and work.parent.resolve() == work_root.resolve():
            found[v] = p
    return found


if __name__ == '__main__':
    scan, out_dir, work_root = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    per_view, z_centre, z_half = int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    seed_base, workers, phantom = int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
    n_views, energy, x_half, n_layers = int(sys.argv[10]), float(sys.argv[11]), float(sys.argv[12]), int(sys.argv[13])
    n_chips, n_bands = int(sys.argv[14]), int(sys.argv[15])
    first = int(sys.argv[16]) if len(sys.argv) > 16 else 0
    n_total = int(sys.argv[17]) if len(sys.argv) > 17 else 360
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / 'progress.log', 'a', buffering=1)
    views = range(first, first + n_views)
    todo = [v for v in views if not (out_dir / f'view_{v:03d}.npz').exists()]
    running = {}                     # view -> (stage, process, work dir, start time, GATE seconds)
    pending = []                     # views whose GATE run finished, waiting for a convert slot
    for v, p in running_gate(work_root, scan).items():
        if v in todo:
            tag = f'{scan}{v:03d}'
            running[v] = ('gate', Adopted(p, work_root / tag / f'info_{tag}.json'), work_root / tag, p.create_time(), 0.0)
            todo.remove(v)
    for v in list(todo):
        tag = f'{scan}{v:03d}'
        if (work_root / tag / f'info_{tag}.json').exists():
            pending.append(v)
            todo.remove(v)
    log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} start {scan}: {len(todo) + len(running) + len(pending)} of '
              f'{n_views} views to do ({len(running)} GATE runs adopted, {len(pending)} to convert), '
              f'{per_view} protons each, {energy:g} MeV, {n_layers} layers of {n_chips} x {n_bands} chips, '
              f'spots x +- {x_half:g} mm, z {z_centre} +- {z_half} mm, {workers} workers, '
              f'at most {MAX_CONVERT} converts\n')
    t_start = time.time()
    done = 0
    env = {**os.environ, **ONE_THREAD, 'PYTHONIOENCODING': 'utf-8'}
    while todo or running or pending:
        for v, (stage, proc, work, t0, t_gate) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[v]
            tag = f'{scan}{v:03d}'
            if proc.returncode != 0:
                log.write(f'{time.strftime("%H:%M:%S")} view {v} {stage} FAILED (exit {proc.returncode}); requeued\n')
                todo.append(v)
                continue
            if stage == 'gate':
                pending.append(v)
                continue
            counts = (work / 'convert.out').read_text().strip().splitlines()[-1]   # the last line: opengate prints its path first
            shutil.copy(work / f'info_{tag}.json', out_dir / f'info_{v:03d}.json')
            shutil.rmtree(work)
            done += 1
            el = time.time() - t_start
            eta = el / done * (len(todo) + len(running) + len(pending))
            ram, commit = free_gb()
            log.write(f'{time.strftime("%H:%M:%S")} view {v:3d} done in {time.time() - t0:5.0f} s (GATE {t_gate:5.0f} s): '
                      f'{counts}; {done} done, {len(todo)} queued, {len(pending)} to convert, {len(running)} running; '
                      f'ETA {eta / 3600:.2f} h; free RAM {ram:.1f} GB, commit {commit:.1f} GB\n')
        while pending and len(running) < workers and sum(s == 'convert' for s, *_ in running.values()) < MAX_CONVERT:
            ram, commit = free_gb()
            if commit - CONVERT_GB < FLOOR_GB or ram - CONVERT_GB < FLOOR_GB:
                break
            v = pending.pop(0)
            tag = f'{scan}{v:03d}'
            work = work_root / tag
            info = json.loads((work / f'info_{tag}.json').read_text())
            cmd = [PY, '-I', str(HERE / 'dtc_convert.py'), str(work), tag, str(out_dir / f'view_{v:03d}.npz')]
            proc = subprocess.Popen(cmd, stdout=open(work / 'convert.out', 'w'),
                                    stderr=open(work_root / f'{tag}.convert.err', 'w'), env=env,
                                    **LOW_PRIORITY)
            running[v] = ('convert', proc, work, time.time() - info['seconds'], info['seconds'])
            time.sleep(1)
        convert_waits = pending and sum(s == 'convert' for s, *_ in running.values()) < MAX_CONVERT
        while todo and len(running) < workers and not convert_waits:
            ram, commit = free_gb()
            if commit - PROCESS_GB < FLOOR_GB or ram - PROCESS_GB < FLOOR_GB:
                break
            v = todo.pop(0)
            tag = f'{scan}{v:03d}'
            work = work_root / tag
            if work.exists():
                shutil.rmtree(work)
            cmd = [PY, '-I', str(HERE / 'dtc_scan.py'), str(work), tag, str(v), str(n_total), str(per_view),
                   str(z_centre), str(z_half), str(seed_base + v), phantom, str(energy), str(x_half), str(n_layers),
                   str(n_chips), str(n_bands)]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(work_root / f'{tag}.err', 'w'),
                                    env=env, **LOW_PRIORITY)
            running[v] = ('gate', proc, work, time.time(), 0.0)
            time.sleep(3)
        time.sleep(2)
    log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} finished {scan}: {done} views in '
              f'{(time.time() - t_start) / 3600:.2f} h\n')
