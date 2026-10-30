"""Build the voxelized head for the GATE pCT scans from the BigBrain FDG phantom's 511 keV attenuation map.

At 511 keV attenuation is almost all Compton scattering, so mu / mu_water is the relative electron density. Each
voxel gets the HU whose Schneider 2000 material has that electron density (HU capped at 1600, the densest bone before
the metals), so GATE builds its materials with the Schneider tables.

Output (in OUT_DIR): head_hu.mhd/.raw (int16, LPS, centred on the axial bounding box) and head_phantom.npz (mu,
rho_e, HU, the voxel size, and the extent), all on the same grid. The grid is the GATE phantom at view angle 0:
x = i (patient left), y = j (posterior), z = k (superior), centre of the array at the origin.

Usage: python -I build_phantom.py UMAP.nii.gz SCHNEIDER_MATERIALS.txt SCHNEIDER_DENSITIES.txt OUT_DIR VOXEL_MM
"""
import pathlib
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

MU_WATER_511 = 0.0969          # 1/cm, NIST XCOM water at 511 keV (mu/rho 0.09687 cm2/g)
ELEMENTS = {'Hydrogen': (1, 1.008), 'Carbon': (6, 12.011), 'Nitrogen': (7, 14.007), 'Oxygen': (8, 15.999),
            'Sodium': (11, 22.990), 'Magnesium': (12, 24.305), 'Phosphor': (15, 30.974), 'Sulfur': (16, 32.06),
            'Chlorine': (17, 35.45), 'Argon': (18, 39.948), 'Potassium': (19, 39.098), 'Calcium': (20, 40.078),
            'Titanium': (22, 47.867), 'Copper': (29, 63.546), 'Zinc': (30, 65.38), 'Silver': (47, 107.87),
            'Tin': (50, 118.71)}
HU_MAX = 1600

umap_path, mat_path, den_path, out_dir, voxel = sys.argv[1:6]
out_dir, voxel = pathlib.Path(out_dir), float(voxel)
out_dir.mkdir(parents=True, exist_ok=True)

# --- Schneider electron density per HU
lines = pathlib.Path(mat_path).read_text().splitlines()
start = lines.index('[Elements]') + 1
names = []
for ln in lines[start:]:
    if ln.startswith('[/Elements]'):
        break
    names += ln.split()
rows = []
for ln in lines:
    p = ln.split()
    if len(p) == len(names) + 2 and p[0].lstrip('-').isdigit():
        rows.append((int(p[0]), np.array(p[1:-1], dtype=float) / 100.0, p[-1]))
zoa = np.array([ELEMENTS[n][0] / ELEMENTS[n][1] for n in names])
den = np.loadtxt(den_path, comments='#')
hu_t = np.arange(-1000, HU_MAX + 1)
rho_t = np.interp(hu_t, den[:, 0], den[:, 1])
mat_idx = np.searchsorted([r[0] for r in rows], hu_t, side='right') - 1
zoa_t = np.array([rows[i][1] @ zoa for i in mat_idx])
zoa_water = 2 * 1 / 18.015 + 8 / 18.015
rhoe_t = rho_t * zoa_t / zoa_water
rhoe_mono = np.maximum.accumulate(rhoe_t)                   # Schneider rho_e(HU) dips at HU 100/101: keep it monotone
print(f'Schneider rho_e: HU -1000 {rhoe_t[0]:.4f}, HU 0 {rhoe_t[1000]:.4f}, HU {HU_MAX} {rhoe_t[-1]:.4f}')

# --- the attenuation map: smooth, resample to the voxel size, RAS -> LPS
img = nib.load(umap_path)
src = float(img.header.get_zooms()[0])
mu = np.asarray(img.dataobj, dtype=np.float32)
mu = ndimage.gaussian_filter(mu, sigma=(voxel / src) / 2.3548)
mu = ndimage.zoom(mu, src / voxel, order=1)
mu = np.clip(mu, 0, None)[::-1, ::-1, :]                    # RAS i, j -> LPS i, j
body = mu > 0.02
nz = [np.flatnonzero(body.any(axis=tuple(a for a in range(3) if a != k))) for k in range(3)]
margin = int(round(4 / voxel))
lo = [max(int(n[0]) - margin, 0) for n in nz]
hi = [min(int(n[-1]) + margin + 1, mu.shape[k]) for k, n in enumerate(nz)]
mu = mu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
rho_e = mu / MU_WATER_511
hu = np.interp(rho_e, rhoe_mono, hu_t).astype(np.float32)
hu[rho_e <= rhoe_t[0]] = -1000
hu = np.clip(np.round(hu), -1000, HU_MAX).astype(np.int16)

shape = np.array(mu.shape)
centre_idx = (shape - 1) / 2
ii, jj = np.nonzero(body[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any(axis=2))
r = np.hypot((ii - centre_idx[0]) * voxel, (jj - centre_idx[1]) * voxel)
print(f'grid {tuple(shape)} at {voxel} mm = {tuple(np.round(shape * voxel, 1))} mm; '
      f'largest axial radius of the head from the rotation axis {r.max():.1f} mm')
print(f'HU: air {np.mean(hu == -1000):.3f} of voxels, soft (-100..100) {np.mean((hu > -100) & (hu < 100)):.3f}, '
      f'bone (>300) {np.mean(hu > 300):.3f}, capped at {HU_MAX}: {np.mean(hu == HU_MAX):.4f}')

# --- write the GATE image (MetaImage, centred) and the arrays
raw = out_dir / 'head_hu.raw'
hu.transpose(2, 1, 0).astype('<i2').tofile(raw)            # MetaImage is x fastest
origin = -centre_idx * voxel
(out_dir / 'head_hu.mhd').write_text(
    'ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\n'
    'TransformMatrix = 1 0 0 0 1 0 0 0 1\n'
    f'Offset = {origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}\nCenterOfRotation = 0 0 0\n'
    f'ElementSpacing = {voxel} {voxel} {voxel}\nDimSize = {shape[0]} {shape[1]} {shape[2]}\n'
    f'ElementType = MET_SHORT\nElementDataFile = {raw.name}\n')
np.savez_compressed(out_dir / 'head_phantom.npz', mu=mu.astype(np.float32), rho_e=rho_e.astype(np.float32), hu=hu,
                    voxel=voxel, origin=origin, r_max=r.max(), crop_lo=np.array(lo), src_voxel=src)
print('wrote', out_dir / 'head_hu.mhd', 'and head_phantom.npz')
