"""Convert one view of dtc_scan.py to a compact table (OUT.npz):

    event       uint32  [N]     EventID of every primary proton that crossed the front plane
    front       float32 [N, 6]  x, z (mm), direction x, y, z, kinetic energy (MeV) entering the front plane (truth)
    exit        float32 [N, 6]  the same entering the exit plane, 3 mm before the first tracker (truth); NaN if never
    hit_event   uint32  [H]     one row per cluster: a track's energy in one epitaxial layer, electrons (delta rays)
    hit_plane   uint8   [H]       added to the track that made them. Plane 2 l + h: layer l (0, 1 trackers), half h
    hit_col     int16   [H]     pixel column at the energy-weighted centre, 1024 per chip (0..9215 for Bergen's 9;
                                -1 between two chips' matrices or off the detector)
    hit_row     int16   [H]     pixel row, band * 512 + row (0..6143 for Bergen's 12 bands; -1 outside the bands)
    hit_edep    float32 [H]     deposited energy, keV
    hit_primary uint8   [H]     1 for the primary proton (TrackID 1), 0 for any other track (nuclear secondaries)
    angle       float64         the phantom's rotation, degrees

Hits are sorted by event, then plane. The raw hits are read in chunks, so memory stays near CHUNK rows.

Usage: python -I dtc_convert.py WORK_DIR TAG OUT.npz
"""
import json
import pathlib
import sys

import numpy as np
import uproot

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtc_geometry as dg  # noqa: E402

CHUNK = 4_000_000
COLS = ['EventID', 'TrackID', 'ParentID', 'PDGCode', 'PostPosition_X', 'PostPosition_Y', 'PostPosition_Z',
        'TotalEnergyDeposit']


def plane_table(work, tag, plane):
    a = uproot.open(work / f'{plane}_{tag}.root')[f'ps_{plane}'].arrays(library='np')
    keep = a['TrackID'] == 1
    ev = a['EventID'][keep]
    cols = np.stack([a['PrePosition_X'][keep], a['PrePosition_Z'][keep], a['PreDirection_X'][keep],
                     a['PreDirection_Y'][keep], a['PreDirection_Z'][keep], a['KineticEnergy'][keep]], axis=1)
    ev, first = np.unique(ev, return_index=True)       # a proton that re-enters a plane: keep its first entry
    return ev, cols[first].astype(np.float32)


def clusters(a, plane_y, n_chips=dg.N_CHIPS, n_bands=dg.N_BANDS):
    """Group raw hits by (event, plane, owner); owner = the parent for electrons, else the track itself."""
    y = a['PostPosition_Y']
    plane = np.clip(np.searchsorted(plane_y, y), 1, len(plane_y) - 1)
    plane = np.where(np.abs(y - plane_y[plane - 1]) < np.abs(y - plane_y[plane]), plane - 1, plane)
    owner = np.where(a['PDGCode'] == 11, a['ParentID'], a['TrackID']).astype(np.int64)
    ev = a['EventID'].astype(np.int64)
    key = (ev << 24) | (plane.astype(np.int64) << 16) | np.minimum(owner, 65535)
    uk, inv = np.unique(key, return_inverse=True)
    e = a['TotalEnergyDeposit']
    esum = np.bincount(inv, weights=e)
    w = np.where(esum > 0, esum, 1.0)
    x = np.bincount(inv, weights=e * a['PostPosition_X']) / w
    z = np.bincount(inv, weights=e * a['PostPosition_Z']) / w
    col, row, _ = dg.pixel_index(x, z, n_chips, n_bands)
    return {'hit_event': (uk >> 24).astype(np.uint32), 'hit_plane': ((uk >> 16) & 0xFF).astype(np.uint8),
            'hit_col': col.astype(np.int16), 'hit_row': row.astype(np.int16),
            'hit_edep': (esum * 1000.0).astype(np.float32), 'hit_primary': ((uk & 0xFFFF) == 1).astype(np.uint8)}


def convert(work, tag, out_path):
    work = pathlib.Path(work)
    info = json.loads((work / f'info_{tag}.json').read_text())
    plane_y = np.array(info['dtc']['sensor_y_mm'])
    size = (info['dtc'].get('n_chips', dg.N_CHIPS), info['dtc'].get('n_bands', dg.N_BANDS))
    assert np.all(np.diff(plane_y) > 0)
    ev, front = plane_table(work, tag, 'front')
    xev, xcols = plane_table(work, tag, 'exit')
    exit_ = np.full_like(front, np.nan)
    _, i, j = np.intersect1d(ev, xev, return_indices=True)
    exit_[i] = xcols[j]
    parts = {k: [] for k in ('hit_event', 'hit_plane', 'hit_col', 'hit_row', 'hit_edep', 'hit_primary')}
    tail = None
    n_raw = 0
    for a in uproot.open(work / f'hits_{tag}.root')['hits'].iterate(COLS, step_size=CHUNK, library='np'):
        n_raw += len(a['EventID'])
        if tail is not None:
            a = {k: np.concatenate([tail[k], a[k]]) for k in COLS}
        last = a['EventID'][-1]
        done = a['EventID'] != last                     # hold the last event back: it may continue in the next chunk
        tail = {k: v[~done] for k, v in a.items()}
        if done.any():
            for k, v in clusters({k: v[done] for k, v in a.items()}, plane_y, *size).items():
                parts[k].append(v)
    if tail is not None and len(tail['EventID']):
        for k, v in clusters(tail, plane_y, *size).items():
            parts[k].append(v)
    hits = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
    with open(out_path, 'wb') as f:
        np.savez_compressed(f, event=ev.astype(np.uint32), front=front, exit=exit_, angle=np.float64(info['angles'][0]),
                            **hits)
    return {'front': len(ev), 'exit': int(np.isfinite(exit_[:, 0]).sum()), 'raw_hits': n_raw,
            'clusters': len(hits['hit_event']), 'primary_clusters': int(hits['hit_primary'].sum())}


if __name__ == '__main__':
    print(convert(sys.argv[1], sys.argv[2], sys.argv[3]))
