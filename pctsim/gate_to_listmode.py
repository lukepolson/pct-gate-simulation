"""Our GATE proton CT simulations converted into PyTomography's proton CT list mode file (pytomography.io.PCT.
save_listmode), outside PyTomography: everything specific to these simulations and to the scanner's calibration.

  1. read the views (view_XXX.npz: the protons at a plane before and after the object; info_XXX.json: the angle and
     the planes), drop the protons that missed the rear plane, and turn each one's rear energy (or the DTC's range)
     into WEPL with a calibration (thickness against mean energy behind water);
  2. drop, per view and per 2 mm entry square, protons whose WEPL or exit-minus-entry angle is more than 3 standard
     deviations from the mean (Schulte et al. 2008: most nuclear events and large-angle scatters);
  3. for a scan without a front tracker (the entry is the pencil beam's ray from the beam log, spread by the spot),
     combine each proton's spot with its exit traced back to the entry plane: its most likely entry and the spread
     left about it (most_likely_entries);
  4. write the protons as PyTomography's standard form.

Usage: python gate_to_listmode.py SCAN_DIR CALIBRATION OUT.h5 [--spot-sd MM --divergence RAD_PER_MM]
                                  [--likely WATER_CALIBRATION --exit-angle-sd RAD --scale S]
The 330 MeV abdomen (single-sided, DTC):
  python gate_to_listmode.py gate/dtc_abdomen330_spot gate/calibration_330_dtc/calib.npz listmode/abdomen330_spot_likely.h5
         --spot-sd 3.69 --divergence 0.763e-3 --likely gate/calibration_330/calib.npz --exit-angle-sd 1.34e-3 --scale 0.887
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

from pytomography.io.PCT import save_listmode                             # noqa: E402
from pytomography.metadata.PCT import PCTProjMeta                         # noqa: E402
from pytomography.metadata.PCT.pct_metadata import FWHM_PER_SD            # noqa: E402

X0_WATER, M_PROTON = 360.8, 938.272          # mm, MeV
W_TABLE = np.arange(0.0, 601.0, 2.0)         # WEPLs of the traced-back spread's table, mm


def _interp(x, xp, fp):
    """np.interp in torch, for increasing ``xp``: linear between the points, the end values outside them."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return y0 + ((x - x0) / (x1 - x0)).clamp(0, 1) * (y1 - y0)


def _cut(keep, bins, values, n_sigmas, passes=2):
    """Keep the entries of ``values`` within ``n_sigmas`` standard deviations of the mean of their bin, the means and
    deviations recomputed from the kept entries ``passes`` times."""
    v = values.double()
    nb = int(bins.max()) + 1 if bins.numel() else 1
    for _ in range(passes):
        k = keep.double()
        n = torch.bincount(bins, weights=k, minlength=nb).clamp(min=1)
        mean = torch.bincount(bins, weights=k * v, minlength=nb) / n
        sd = torch.sqrt((torch.bincount(bins, weights=k * v * v, minlength=nb) / n - mean ** 2).clamp(min=0))
        keep = keep & ((v - mean[bins]).abs() <= n_sigmas * sd[bins] + 1e-6)
    return keep


def _convert(files, views, cal, z_center, n_sigmas, cut_bin, device):
    """A group of views on ``device``: per view (entry [n, 4], exit [n, 4], WEPL [n]) on the host, cut per view."""
    tables = [np.load(files[v]) for v in views]
    tables = [(t['front'], t['rear']) for t in tables]
    front = torch.cat([torch.from_numpy(f) for f, _ in tables]).to(device)
    rear = torch.cat([torch.from_numpy(r) for _, r in tables]).to(device)
    local = torch.cat([torch.full((f.shape[0],), k) for k, (f, _) in enumerate(tables)]).to(device)
    del tables
    keep = torch.isfinite(rear[:, 0])
    front, rear, local = front[keep], rear[keep], local[keep]
    wepl = _interp(rear[:, 5], cal[0].to(device), cal[1].to(device))
    if len(wepl):
        ix = torch.floor((front[:, 0] + 500) / cut_bin).long()
        iz = torch.floor((front[:, 1] + 500) / cut_bin).long()
        bins = torch.unique((local * 1000 + ix) * 1000 + iz, return_inverse=True)[1]
        keep = _cut(torch.ones(len(wepl), dtype=torch.bool, device=device), bins, wepl, n_sigmas)
        for k in (0, 2):        # exit minus entry angle in x and in z
            dtheta = torch.atan2(rear[:, 2 + k], rear[:, 3]) - torch.atan2(front[:, 2 + k], front[:, 3])
            keep = _cut(keep, bins, dtheta, n_sigmas)
        front, rear, wepl, local = front[keep], rear[keep], wepl[keep], local[keep]

    def plane(t):               # x, z, dx/dy, dz/dy
        return torch.stack([t[:, 0], t[:, 1] - z_center, t[:, 2] / t[:, 3], t[:, 4] / t[:, 3]], dim=1).cpu()
    sizes = torch.bincount(local, minlength=len(views)).tolist()
    return list(zip(plane(front).split(sizes), plane(rear).split(sizes), wepl.cpu().split(sizes)))


def read_gate(scan_dir, calibration, z_center=0.0, n_sigmas=3.0, cut_bin=2.0, entry_fwhm=0.0, beam_divergence=0.0,
              device='cuda', views_per_load=8):
    """The views of ``scan_dir`` as (WEPL [N] in mm, PCTProjMeta); positions in mm, angles in degrees, energies in MeV
    in the files (keV in PyTomography). ``entry_fwhm`` (mm) and ``beam_divergence`` (degrees per mm): the spot's,
    for entries that are the pencil beam's ray."""
    scan_dir = pathlib.Path(scan_dir)
    files = sorted(scan_dir.glob('view_*.npz'))
    infos = [json.loads((scan_dir / f'info_{f.stem.split("_")[1]}.json').read_text()) for f in files]
    c = np.load(calibration)
    cal = (torch.as_tensor(c['energy'][::-1].copy(), dtype=torch.float32),
           torch.as_tensor(c['thickness'][::-1].copy(), dtype=torch.float32))
    beam_energy = 1000.0 * float(np.median(np.load(files[0])['front'][:, 5]))
    parts = []
    for start in range(0, len(files), views_per_load):
        parts += _convert(files, range(start, min(start + views_per_load, len(files))), cal, float(z_center),
                          float(n_sigmas), float(cut_bin), device)
    sizes = [len(w) for _, _, w in parts]
    joined = []
    for k in range(3):          # one table at a time, each view's piece freed as it is copied
        joined.append(torch.cat([p[k] for p in parts]))
        parts = [tuple(None if j == k else q for j, q in enumerate(p)) for p in parts]
    entry, exit, wepl = joined
    proj_meta = PCTProjMeta([i['angles'][0] for i in infos], [i['front_y'] for i in infos],
                            [i['rear_y'] for i in infos], entry, exit, sizes, beam_energy, entry_fwhm=entry_fwhm,
                            beam_divergence=beam_divergence, z_center=z_center)
    return wepl, proj_meta


def _back_spread(front, rear, thickness, energy):
    """Highland's spread (mm) of an exit traced back to the entry plane, for each WEPL of W_TABLE: the scattering
    power at the mean energy after w mm of water, the water taken as centred on the rotation axis."""
    out = []
    for W in W_TABLE:
        if W < 1.0:
            out.append(0.0)
            continue
        w = np.linspace(0.0, W, 400)
        E = np.interp(w, thickness, energy)
        bcp = (E * E + 2 * E * M_PROTON) / (E + M_PROTON)
        k = (13.6 / bcp) ** 2 / X0_WATER
        D0 = -front - W / 2
        corr = (1 + 0.038 * math.log(W / X0_WATER)) ** 2
        out.append(math.sqrt(np.trapezoid(k * (D0 + w) ** 2, w) * corr))
    return np.array(out)


def most_likely_entries(proj_meta, wepl, water_calibration, exit_angle_sd, scale=0.887, n_sigmas=3.0, device='cuda'):
    """Each proton's most likely entry: its spot (the stored entry, Gaussian of proj_meta.entry_fwhm) combined, per
    axis, with its exit traced straight back to the entry plane (Gaussian of Highland's spread times ``scale`` and
    the exit direction's error ``exit_angle_sd`` (rad) times the distance between the planes). The entry moves by
    Delta sigma_s^2 / (sigma_s^2 + sigma_b^2) and its spread narrows to (sigma_s^-2 + sigma_b^-2)^-1/2; its direction
    turns with the offset by the beam's divergence. A proton whose traced-back point is more than ``n_sigmas``
    combined deviations from its spot's ray keeps the spot. scale 0.887 and 1.34 mrad were fitted on the 330 MeV
    abdomen's true entries."""
    cal = np.load(water_calibration)
    kappa = math.radians(proj_meta.beam_divergence)
    tables = {}
    entry = torch.empty_like(proj_meta.entry)
    fwhm = torch.empty(proj_meta.N_protons, dtype=torch.float32)
    for v in range(proj_meta.N_angles):
        lo, hi = int(proj_meta.view_offsets[v]), int(proj_meta.view_offsets[v + 1])
        front, rear = float(proj_meta.front[v]), float(proj_meta.rear[v])
        if (front, rear) not in tables:
            tables[front, rear] = torch.as_tensor(_back_spread(front, rear, cal['thickness'], cal['energy']),
                                                  dtype=torch.float32, device=device)
        table = tables[front, rear]
        exit_term = (exit_angle_sd * (rear - front)) ** 2
        t0, t1 = proj_meta.entry[lo:hi].to(device), proj_meta.exit[lo:hi].to(device)
        g = wepl[lo:hi].to(device)
        sd = proj_meta.entry_fwhm
        sd = (sd[lo:hi].to(device) if isinstance(sd, torch.Tensor) else torch.full_like(g, sd)) / FWHM_PER_SD
        x = g.clamp(0, float(W_TABLE[-1])) / 2.0
        i = x.floor().long().clamp(max=len(W_TABLE) - 2)
        f = x - i
        sb = torch.sqrt((scale * (table[i] * (1 - f) + table[i + 1] * f)) ** 2 + exit_term)
        delta = t1[:, :2] - t1[:, 2:] * (rear - front) - t0[:, :2]      # traced-back point minus the spot's ray
        tot = torch.sqrt(sd ** 2 + sb ** 2)
        fits = (delta.abs() < n_sigmas * tot[:, None]).all(dim=1)
        mu = torch.where(fits, sd ** 2 / tot ** 2, torch.zeros_like(tot))[:, None] * delta
        norm = torch.rsqrt(1 + t0[:, 2] ** 2 + t0[:, 3] ** 2)[:, None]
        entry[lo:hi] = torch.cat([t0[:, :2] + mu, t0[:, 2:] + kappa * mu / norm], dim=1).cpu()
        fwhm[lo:hi] = (torch.where(fits, torch.rsqrt(1.0 / sd ** 2 + 1.0 / sb ** 2), sd) * FWHM_PER_SD).cpu()
    return proj_meta.replace(entry=entry, entry_fwhm=fwhm)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('scan_dir')
    p.add_argument('calibration')
    p.add_argument('out')
    p.add_argument('--spot-sd', type=float, default=0.0, help='the spot sd at the entry plane, mm (0: a front tracker)')
    p.add_argument('--divergence', type=float, default=0.0, help='the spot direction per mm of offset, rad/mm')
    p.add_argument('--likely', help='water calibration for the most likely entries (single-sided scans)')
    p.add_argument('--exit-angle-sd', type=float, default=1.34e-3, help='rad')
    p.add_argument('--scale', type=float, default=0.887)
    p.add_argument('--device', help="where to convert: 'cuda' or 'cpu' (default: cuda when PyTorch sees a GPU)")
    a = p.parse_args()
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.perf_counter()
    wepl, pm = read_gate(a.scan_dir, a.calibration, entry_fwhm=a.spot_sd * FWHM_PER_SD,
                         beam_divergence=math.degrees(a.divergence), device=device)
    t1 = time.perf_counter()
    if a.likely:
        pm = most_likely_entries(pm, wepl, a.likely, a.exit_angle_sd, a.scale, device=device)
    t2 = time.perf_counter()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_listmode(a.out, wepl, pm)
    print(json.dumps({'protons': pm.N_protons, 'read_s': round(t1 - t0, 1), 'entries_s': round(t2 - t1, 1),
                      'write_s': round(time.perf_counter() - t2, 1),
                      'file_gb': round(pathlib.Path(a.out).stat().st_size / 1e9, 2)}))
