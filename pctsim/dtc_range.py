"""From one proton's clusters to what the DTC measures: its position and direction behind the object (the two tracker
layers) and its range in the calorimeter (a Bragg-Kleeman fit of its energy per layer, Pettersen et al. 2019), with
a filter for protons that end in a nuclear interaction instead of a Bragg peak.

Track finding is not modelled: a proton's clusters are found by its event number and the primary flag
(dtc_convert.py), so this is the detector's measurement with perfect track finding.

Range fit, per proton (as dtc_beam_check.py, which found it 1.1 mm WET short with an sd of 1.0-1.2 mm on pencil
beams): edep_l = A (R - z_l)^(1/p - 1) over all its calorimeter layers up to the deepest, z_l the layers' depths from
the front face of calorimeter layer 0, A in closed form, R on a GRID_MM grid within the pitch after the deepest
layer. The layers are evenly spaced, so the fit for every proton is one matrix product. Depths are the layers' gap
centres for both halves: the two halves' sensors are 1.5 mm apart across air, about 0.25 mm WET (shifting them by
their physical offset at the stack's mean WET per mm split the errors into two peaks 2.2 mm apart). R is in mm WET.

Filter: a proton that ends in a nuclear interaction loses its primary flag where it interacts, with no Bragg peak, so
its fitted amplitude A is low (median 75-88 against 166-168 for protons that stop by ionisation, at 190-330 MeV) and
its fit poorer. Keep a proton if A is within A_RANGE and its relative residual is at most MAX_REL_RESIDUAL. On the
330 MeV pencil-beam test (2,000 protons per energy, known ends) this keeps 95-97% of the protons that stop by
ionisation and 5-14% of those that end in a nuclear interaction, at 190-330 MeV. Rerun the check with
python -I dtc_range.py TEST_DIR MATERIALS_JSON.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtc_geometry as dg  # noqa: E402

GRID_MM = 0.05
MAX_REL_RESIDUAL = 0.35
A_RANGE = (135.0, 210.0)         # keV: the fit's energy per layer 1 mm (physical) before the end of range


class Stack:
    """The calorimeter's depths and the range fit's constants, from a scan's or test's DTC info and the materials."""

    def __init__(self, dtc_info, materials):
        self.n_layers = dtc_info['n_layers']
        self.n_chips, self.n_bands = dtc_info.get('n_chips', dg.N_CHIPS), dtc_info.get('n_bands', dg.N_BANDS)
        self.sensor_y = np.array(dtc_info['sensor_y_mm'])
        self.layer_y = np.array(dtc_info['layer_y_mm'])
        cal_y = np.array(dtc_info['calorimeter_y_mm'])
        self.z0 = cal_y[0] - dg.GAP / 2 - dg.CAL['carrier'][1]          # front face of calorimeter layer 0
        self.layer_z = cal_y - self.z0                                 # gap centres
        self.wet_per_mm = materials['layer_wet']['cal']['wet_mm'] / dg.PITCH
        self.p = materials['bragg_kleeman_340']['p']
        self.alpha = materials['bragg_kleeman_340']['alpha_mm_per_MeV_p']
        self.q = 1.0 / self.p - 1.0
        g = np.arange(GRID_MM, dg.PITCH + 1e-9, GRID_MM)
        self.grid = g
        j = np.arange(self.n_layers)
        self.F = ((g[:, None] + j[None, :] * dg.PITCH) ** self.q).astype(np.float32)       # [grid, layers back]
        self.F2 = self.F ** 2

    def pixel_mm(self, col, row):
        """Pixel centres, mm (the inverse of dtc_geometry.pixel_index)."""
        chip, c = col // 1024, col % 1024
        x = chip * dg.CHIP_X + (c + 0.5) * dg.PIXEL_X - self.n_chips * dg.CHIP_X / 2
        band, r = row // 512, row % 512
        z = band * dg.PIXELS_Z + (r + 0.5) * dg.PIXEL_Z - self.n_bands * dg.PIXELS_Z / 2
        return x, z


def per_proton(view, events, stack):
    """For each of ``events`` (a sorted array of event numbers): the tracker points and the calorimeter's energy per
    layer, from the view's primary clusters.

    Returns: points [N, 2, 3] (x, y, z in mm at the two tracker layers; NaN where the layer has no primary cluster or
    it fell between two chips' matrices) and cal [N, n_layers] keV (summed over the two halves)."""
    prim = view['hit_primary'] == 1
    ev = view['hit_event'][prim]
    i = np.searchsorted(events, ev)
    ok = (i < len(events)) & (events[np.minimum(i, len(events) - 1)] == ev)
    i, plane = i[ok], view['hit_plane'][prim][ok].astype(np.int64)
    col, row = view['hit_col'][prim][ok].astype(np.int64), view['hit_row'][prim][ok].astype(np.int64)
    edep = view['hit_edep'][prim][ok].astype(np.float32)
    layer = plane // 2
    n = len(events)
    points = np.full((n, 2, 3), np.nan, np.float32)
    trk = (layer < 2) & (col >= 0) & (row >= 0)
    x, z = stack.pixel_mm(col[trk], row[trk])
    order = np.argsort(edep[trk], kind='stable')      # two primary clusters in a layer: the larger is written last
    ii, ll = i[trk][order], layer[trk][order]
    points[ii, ll, 0] = x[order]
    points[ii, ll, 1] = stack.sensor_y[plane[trk][order]]
    points[ii, ll, 2] = z[order]
    c = layer >= 2
    cal = np.zeros((n, stack.n_layers), np.float32)
    np.add.at(cal, (i[c], layer[c] - 2), edep[c])
    return points, cal


def fit_range(cal, stack, chunk=100_000):
    """Bragg-Kleeman range per proton. Returns R (mm WET from the front of calorimeter layer 0; NaN with no
    calorimeter layer or when the proton reached the last one), A (the fitted amplitude), the relative residual
    (residual / sum of squares), and the deepest layer."""
    n = len(cal)
    has = cal > 0
    deepest = np.where(has.any(axis=1), stack.n_layers - 1 - np.argmax(has[:, ::-1], axis=1), -1)
    R = np.full(n, np.nan, np.float32)
    A = np.full(n, np.nan, np.float32)
    rel = np.full(n, np.nan, np.float32)
    back = np.arange(stack.n_layers)
    for s in range(0, n, chunk):
        d = deepest[s:s + chunk]
        idx = d[:, None] - back[None, :]
        valid = idx >= 0
        e = np.where(valid, cal[s:s + chunk][np.arange(len(d))[:, None], np.clip(idx, 0, None)], 0.0)
        e = e.astype(np.float32)
        fe = e @ stack.F.T                                    # [n, grid]
        ff = (e > 0).astype(np.float32) @ stack.F2.T
        ss = (e * e).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            obj = ss[:, None] - fe * fe / ff
        g = np.argmin(np.where(ff > 0, obj, np.inf), axis=1)
        r = np.arange(len(d))
        depth = stack.layer_z[np.clip(d, 0, None)] + stack.grid[g]
        good = (d >= 1) & (d < stack.n_layers - 1)
        R[s:s + chunk] = np.where(good, depth * stack.wet_per_mm, np.nan)
        A[s:s + chunk] = np.where(good, fe[r, g] / ff[r, g], np.nan)
        rel[s:s + chunk] = np.where(good, obj[r, g] / np.maximum(ss, 1e-12), np.nan)
    return R, A, rel, deepest


def exit_track(points, exit_y):
    """Position at y = exit_y and unit direction from the two tracker points; NaN where either is missing."""
    p0, p1 = points[:, 0], points[:, 1]
    d = p1 - p0
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    t = (exit_y - p0[:, 1]) / d[:, 1]
    pos = p0 + t[:, None] * d
    return pos, d


def measure(view, events, stack, exit_y):
    """What the DTC measures for each of ``events``, as a 'rear' table: x, z at y = exit_y (mm), the direction x, y, z,
    and the range in the calorimeter (mm WET) in place of an energy; NaN rows for protons without both tracker points
    or a range, or rejected by the filter. Also returns counts of each loss."""
    points, cal = per_proton(view, events, stack)
    R, A, rel, deepest = fit_range(cal, stack)
    pos, d = exit_track(points, exit_y)
    tracked = np.isfinite(pos).all(axis=1)
    ranged = tracked & np.isfinite(R)
    kept = ranged & (A >= A_RANGE[0]) & (A <= A_RANGE[1]) & (rel <= MAX_REL_RESIDUAL)
    rear = np.full((len(events), 6), np.nan, np.float32)
    rear[kept] = np.column_stack([pos[kept, 0], pos[kept, 2], d[kept], R[kept]])
    counts = {'protons': int(len(events)), 'tracked': int(tracked.sum()), 'ranged': int(ranged.sum()),
              'kept': int(kept.sum()), 'reached_last_layer': int((deepest == stack.n_layers - 1).sum())}
    return rear, counts


if __name__ == '__main__':
    # Set the filter on pencil beams with known ends: which features separate ionisation stops from nuclear ends.
    import uproot
    from dtc_convert import COLS, clusters
    test, mat_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    info = json.loads((test / 'info.json').read_text())
    stack = Stack(info, json.loads(mat_path.read_text()))
    hits = uproot.open(test / 'hits.root')['hits'].arrays(COLS + ['RunID'], library='np')
    stops = uproot.open(test / 'stops.root')['stops'].arrays(library='np')
    n = info['n_per_energy']
    rows = {}
    for r, E in enumerate(info['energies_MeV']):
        sel = hits['RunID'] == r
        c = clusters({k: hits[k][sel] for k in COLS}, stack.sensor_y, stack.n_chips, stack.n_bands)
        events = np.arange(n)
        points, cal = per_proton(c, events, stack)
        R, A, rel, deepest = fit_range(cal, stack)
        s = (stops['RunID'] == r) & (stops['TrackID'] == 1)
        y_stop = np.full(n, np.nan)
        ion = np.zeros(n, bool)
        y_stop[stops['EventID'][s].astype(int)] = stops['PostPosition_Y'][s]
        ion[stops['EventID'][s].astype(int)] = stops['ProcessDefinedStep'][s] == 'hIoni'
        r_true = (y_stop - stack.z0) * stack.wet_per_mm
        rows[E] = dict(R=R, A=A, rel=rel, ion=ion, r_true=r_true)
        ok = np.isfinite(R)
        err = (R - r_true)[ok & ion]
        med = lambda x: float(np.median(x))
        rsd = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))
        print(f'{E:5.0f} MeV: ion {ion.mean():.3f}; range error {med(err):+.2f} +- {rsd(err):.2f} mm WET; '
              f'A ion median {med(A[ok & ion]):.1f} (rsd {rsd(A[ok & ion]):.1f}), nuclear {med(A[ok & ~ion]):.1f}; '
              f'rel residual ion p50/p95 {np.percentile(rel[ok & ion], 50):.3f}/{np.percentile(rel[ok & ion], 95):.3f}, '
              f'nuclear p50 {np.percentile(rel[ok & ~ion], 50):.3f}')
    np.savez(test / 'range_features.npz', **{f'{k}_{E:g}': v for E, d in rows.items() for k, v in d.items()})
