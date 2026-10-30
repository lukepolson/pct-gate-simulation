"""A voxelised phantom for GATE from any CT: an axial slab of the CT's own voxels (not resampled), the couch and
everything else outside the body set to air, in the frame the scans use.

The CT is a NIfTI file or a folder of one axial DICOM series. Its voxels are put in LPS order (x = i towards the
patient's left, y = j posterior, z = k superior) and HU are clamped to -1000..1600 (scanners pad the field of view
with -3024). The slab is given by native slice indices (--slices K0 K1, K1 not included) or by masks whose axial range,
plus --margin mm either side, it covers (--around). The body, per slice: the largest piece above -400 HU after an
opening of --open mm (which cuts the couch off where it only touches), grown back inside the threshold, holes filled.
In plane the box is the body's plus 4 mm; the array's centre is the origin, which the scans use as the rotation axis.

Organ masks (--masks NAME=PATH ..., any NIfTI grid) are taken onto the phantom's grid, nearest neighbour, inside the
body.

Output (OUT_DIR): NAME_hu.mhd/.raw (int16, MetaImage, x fastest), NAME_phantom.npz (hu, body, voxel, origin, the masks,
info), NAME_phantom.json (info) and NAME_check.png.

Usage: python -I phantom_from_ct.py CT OUT_DIR NAME (--slices K0 K1 | --around MASK [MASK ...] [--margin MM])
                                    [--masks NAME=PATH ...] [--open MM]

The published 330 MeV abdomen: the CT of TCIA PSMA-PET-CT-Lesions patient PSMA_28f9ecc106933531 with --slices 150 207
(the kidneys' range plus 25 mm).
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HU_MIN, HU_MAX = -1000, 1600


def read_ct(path):
    """(LPS-ordered HU array [x, y, z], voxel size (mm), function from LPS indices [n, 3] to LPS world mm)."""
    path = pathlib.Path(path)
    if path.is_dir():
        import pydicom
        slices = [pydicom.dcmread(p) for p in sorted(path.iterdir()) if p.is_file()]
        slices = [s for s in slices if hasattr(s, 'ImagePositionPatient')]
        orient = np.array(slices[0].ImageOrientationPatient, dtype=float)
        if not np.allclose(orient, [1, 0, 0, 0, 1, 0], atol=1e-3):
            raise ValueError(f'only axial series with ImageOrientationPatient 1 0 0 0 1 0 are read, not {orient}')
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))       # inferior to superior
        z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
        lps = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, 'RescaleSlope', 1))
                        + float(getattr(s, 'RescaleIntercept', 0)) for s in slices], axis=-1).transpose(1, 0, 2)
        dx, dy = (float(v) for v in slices[0].PixelSpacing[::-1])        # PixelSpacing is (row, column) spacing
        voxel = np.array([dx, dy, float(np.median(np.diff(z)))])
        x0, y0 = (float(v) for v in slices[0].ImagePositionPatient[:2])

        def to_world(idx):
            idx = np.asarray(idx, dtype=float)
            return np.stack([x0 + idx[:, 0] * dx, y0 + idx[:, 1] * dy, np.interp(idx[:, 2], np.arange(len(z)), z)], 1)
        return lps, voxel, to_world
    img = nib.as_closest_canonical(nib.load(path))                         # RAS
    ras = np.asarray(img.dataobj, dtype=np.float32)
    shape0 = np.array(ras.shape)

    def to_world(idx):
        r = np.array(idx, dtype=float)
        r[:, 0] = shape0[0] - 1 - r[:, 0]
        r[:, 1] = shape0[1] - 1 - r[:, 1]
        w = nib.affines.apply_affine(img.affine, r)                        # RAS world
        return w * np.array([-1.0, -1.0, 1.0])                             # LPS world
    return ras[::-1, ::-1, :], np.array(img.header.get_zooms()[:3], dtype=float), to_world


def mask_on(path, shape, to_world, k_lo, k_hi):
    """A NIfTI mask on slices k_lo..k_hi of the LPS grid, nearest neighbour."""
    m = nib.load(path)
    data = np.asarray(m.dataobj) > 0
    inv = np.linalg.inv(m.affine)
    out = np.zeros((shape[0], shape[1], k_hi - k_lo), bool)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    for k in range(k_lo, k_hi):
        world = to_world(np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k)], axis=1)) * np.array([-1.0, -1.0, 1.0])
        s = np.round(nib.affines.apply_affine(inv, world)).astype(int)
        ok = np.all((s >= 0) & (s < np.array(data.shape)), axis=1)
        v = np.zeros(ii.size, bool)
        v[ok] = data[s[ok, 0], s[ok, 1], s[ok, 2]]
        out[:, :, k - k_lo] = v.reshape(ii.shape)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('ct')
    p.add_argument('out_dir', type=pathlib.Path)
    p.add_argument('name')
    p.add_argument('--slices', type=int, nargs=2, metavar=('K0', 'K1'), help='native slices K0 to K1 - 1')
    p.add_argument('--around', nargs='+', help='masks whose axial range (plus --margin) the slab covers')
    p.add_argument('--margin', type=float, default=25.0, help='mm either side of --around (default 25)')
    p.add_argument('--masks', nargs='*', default=[], help='NAME=PATH organ masks to keep on the phantom grid')
    p.add_argument('--open', type=float, default=4.0, help='radius of the opening that cuts the couch off, mm')
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lps, voxel, to_world = read_ct(a.ct)
    z_world = to_world(np.stack([np.zeros(lps.shape[2]), np.zeros(lps.shape[2]), np.arange(lps.shape[2])], 1))[:, 2]
    if a.slices:
        k_lo, k_hi = a.slices
    elif a.around:
        zs = []
        for path in a.around:
            m = nib.load(path)
            zs.append(nib.affines.apply_affine(m.affine, np.argwhere(np.asarray(m.dataobj) > 0))[:, 2])
        zs = np.concatenate(zs)
        k_lo = int(np.searchsorted(z_world, zs.min() - a.margin))
        k_hi = int(np.searchsorted(z_world, zs.max() + a.margin, side='right'))
    else:
        p.error('give --slices or --around')
    slab = np.clip(lps[:, :, k_lo:k_hi], HU_MIN, HU_MAX)

    # the body, slice by slice
    r_open = max(1, int(round(a.open / voxel[0])))
    g = np.mgrid[-r_open:r_open + 1, -r_open:r_open + 1]
    disk = (g ** 2).sum(0) <= r_open * r_open
    body = np.zeros(slab.shape, bool)
    for k in range(slab.shape[2]):
        m = slab[:, :, k] > -400
        core = ndimage.binary_opening(m, structure=disk)
        lab, n = ndimage.label(core)
        if n == 0:
            continue
        big = lab == (1 + int(np.argmax(ndimage.sum(core, lab, range(1, n + 1)))))
        body[:, :, k] = ndimage.binary_fill_holes(ndimage.binary_dilation(big, structure=disk) & m)
    hu_slab = np.where(body, slab, HU_MIN)

    # crop in plane to the body + 4 mm; the array's centre becomes the rotation axis
    ii, jj = np.nonzero(body.any(axis=2))
    pad = int(np.ceil(4 / voxel[0]))
    i0, i1 = max(ii.min() - pad, 0), min(ii.max() + pad + 1, slab.shape[0])
    j0, j1 = max(jj.min() - pad, 0), min(jj.max() + pad + 1, slab.shape[1])
    hu = np.round(hu_slab[i0:i1, j0:j1]).astype(np.int16)
    body = body[i0:i1, j0:j1]
    shape = np.array(hu.shape)
    centre_idx = (shape - 1) / 2
    origin = -centre_idx * voxel
    masks = {}
    for item in a.masks:
        name, path = item.split('=', 1)
        masks[name] = mask_on(path, lps.shape, to_world, k_lo, k_hi)[i0:i1, j0:j1] & body
    bi, bj = np.nonzero(body.any(axis=2))
    r_max = float(np.hypot((bi - centre_idx[0]) * voxel[0], (bj - centre_idx[1]) * voxel[1]).max())
    info = {'source': str(a.ct), 'grid': shape.tolist(), 'voxel_mm': voxel.round(4).tolist(),
            'size_mm': (shape * voxel).round(1).tolist(), 'native_slices': [k_lo, k_hi],
            'body_max_radius_mm': round(r_max, 1),
            'box_half_diagonal_mm': round(float(np.hypot(shape[0] * voxel[0] / 2, shape[1] * voxel[1] / 2)), 1),
            'body_width_mm': round(float((bi.max() - bi.min() + 1) * voxel[0]), 1),
            'body_depth_mm': round(float((bj.max() - bj.min() + 1) * voxel[1]), 1),
            'voxels': int(shape.prod()), 'masks_voxels': {k: int(v.sum()) for k, v in masks.items()},
            'hu_fraction': {'air': round(float(np.mean(hu == HU_MIN)), 3),
                            'bone_over_300': round(float(np.mean(hu > 300)), 4),
                            'capped_1600': round(float(np.mean(hu == HU_MAX)), 5)}}
    print(json.dumps(info, indent=1))

    raw = a.out_dir / f'{a.name}_hu.raw'
    hu.transpose(2, 1, 0).astype('<i2').tofile(raw)                # MetaImage is x fastest
    (a.out_dir / f'{a.name}_hu.mhd').write_text(
        'ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\n'
        'TransformMatrix = 1 0 0 0 1 0 0 0 1\n'
        f'Offset = {origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}\nCenterOfRotation = 0 0 0\n'
        f'ElementSpacing = {voxel[0]:.6f} {voxel[1]:.6f} {voxel[2]:.6f}\nDimSize = {shape[0]} {shape[1]} {shape[2]}\n'
        f'ElementType = MET_SHORT\nElementDataFile = {raw.name}\n')
    np.savez_compressed(a.out_dir / f'{a.name}_phantom.npz', hu=hu, body=body, voxel=voxel, origin=origin, **masks,
                        info=json.dumps(info))
    (a.out_dir / f'{a.name}_phantom.json').write_text(json.dumps(info, indent=1))

    kc = shape[2] // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    ax[0].imshow(slab[i0:i1, j0:j1, kc].T, cmap='gray', vmin=-200, vmax=300)
    ax[0].contour(np.arange(i1 - i0), np.arange(j1 - j0), body[:, :, kc].T.astype(float), levels=[0.5], colors='C1')
    ax[0].set_title(f'CT, middle slice (z = {(kc - centre_idx[2]) * voxel[2]:+.1f} mm), body outline')
    ax[1].imshow(hu[:, shape[1] // 2, :].T, cmap='gray', vmin=-200, vmax=300, origin='lower',
                 aspect=voxel[2] / voxel[0])
    ax[1].set_title('Phantom, coronal through the centre')
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])
    fig.savefig(a.out_dir / f'{a.name}_check.png', dpi=80)
    print('wrote', a.out_dir / f'{a.name}_hu.mhd')


if __name__ == '__main__':
    main()
