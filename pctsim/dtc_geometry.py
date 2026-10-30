"""The Bergen digital tracking calorimeter (DTC) as GATE 10 volumes: 2 tracker layers and N_LAYERS calorimeter layers
(41 in the design), each layer two half-layers of ALPIDE chips facing each other across a 2 mm gap.

Frame (GATE world, mm): the beam travels along +y, z is vertical (the rotation axis), x is lateral. Layers are normal to
y. A layer's position is the centre of its 2 mm gap; the positions are the ones measured from Bergen's own hits
(Zenodo 7426388): trackers at 225.35 and 283.15 mm, calorimeter layer k at 326.09 + 5.5 k mm, all moved back by SHIFT
for a phantom whose swept circle reaches the first tracker.

One calorimeter layer, upstream to downstream (5.5 mm pitch):
    carrier A, 1 mm Al | glue | flex (polyimide, Al) | chips of half A, epitaxial layer facing +y |  air  |
    chips of half B, epitaxial layer facing -y | flex | glue | carrier B, 1 mm Al | absorber, 1.5 mm Al
so a proton crosses 3.5 mm of aluminium per layer, two flex sheets and, outside the overlaps, one chip. A tracker layer
is the same without the absorber, with 0.2 mm carbon-epoxy carriers and 50 um chips.

Chips: strings of 9 ALPIDEs along x (30 x 15 mm each, 1024 x 512 pixels of 29.24 x 26.88 um), 6 strings per half-layer,
12 per layer (108 chips). A chip's pixel matrix is 13.76 mm tall; its 1.24 mm periphery lies behind the other half's
pixels, so the two halves tile z = -82.56 .. 82.56 mm: band b = 0..11 (13.76 mm each) is covered by half b % 2.

Sources: Alme et al. 2020, Front. Phys. 8 568243 (the layer, the strings, the chips, the trackers); Pettersen et al.
2019, Phys. Med. 63 87 (the flex and glue budget per layer: 120 um Al, 160 um polyimide, 30 um epoxy, split evenly here
between the two halves). Not given in either, so chosen here: no gaps between the chips of a string; each flex is a
continuous sheet 5 mm beyond the chips on every side, carriers and absorbers 10 mm beyond; air fills the rest of the
gap; the carbon-epoxy is 60% carbon fibre by volume.

A larger detector (more chips per string, more strings, more layers) keeps the same layers and pitch: N_CHIPS and
N_BANDS set its width (30 mm per chip) and height (13.76 mm per band).
"""
import numpy as np

import opengate as gate

TRACKER_Y = (225.35, 283.15)
CAL_Y0, PITCH = 326.09, 5.5
GAP = 2.0
N_CHIPS, N_BANDS = 9, 12             # Bergen: 9 chips per string (270 mm), 12 strings per layer (165 mm)
CHIP_Z, PIXELS_Z = 15.0, 13.76       # chip height; its pixel matrix (512 x 26.88 um)
Z_LO = -N_BANDS * PIXELS_Z / 2       # -82.56 mm for Bergen's 12 bands
PIXEL_X, PIXEL_Z = 0.02924, 0.02688
CHIP_X, MATRIX_X = 30.0, 1024 * 0.02924
EPI = 0.025
GLUE, POLYIMIDE, FLEX_AL = 0.015, 0.080, 0.060

CAL = {'carrier': ('G4_Al', 1.0), 'absorber': ('G4_Al', 1.5), 'chip': 0.100}
TRK = {'carrier': ('DTC_CarbonEpoxy', 0.2), 'chip': 0.050}


def add_materials(sim):
    """Epoxy (bisphenol-A diglycidyl ether, C21H24O4, 1.16 g/cm3) and a carbon-epoxy sheet (60% carbon fibre at
    1.80 g/cm3 by volume: 1.544 g/cm3)."""
    gcm3 = gate.g4_units.g_cm3
    db = sim.volume_manager.material_database
    db.add_material_nb_atoms('DTC_Epoxy', ['C', 'H', 'O'], [21, 24, 4], 1.16 * gcm3)
    m_c, m_h, m_o = 21 * 12.011, 24 * 1.008, 4 * 15.999
    epoxy = np.array([m_c, m_h, m_o]) / (m_c + m_h + m_o)
    rho_cf, rho_ep, f = 1.80, 1.16, 0.60
    rho = f * rho_cf + (1 - f) * rho_ep
    w_cf = f * rho_cf / rho
    w = (1 - w_cf) * epoxy + np.array([w_cf, 0, 0])
    db.add_material_weights('DTC_CarbonEpoxy', ['C', 'H', 'O'], [float(v) for v in w], rho * gcm3)


def half_layer_slabs(kind):
    """The slabs of one half-layer, from its carrier's face into the gap: (name, material, thickness mm). The chip is
    last; its epitaxial layer is on the side away from the carrier."""
    chip = (CAL if kind == 'cal' else TRK)['chip']
    return [('glue', 'DTC_Epoxy', GLUE), ('polyimide', 'G4_KAPTON', POLYIMIDE), ('flexal', 'G4_Al', FLEX_AL),
            ('chip', 'G4_Si', chip)]


def layer_slabs(kind):
    """Every slab a proton crosses in one layer, upstream to downstream, as (name, material, y0, y1) in mm from the
    gap centre, crossing the chip of half A only (outside the overlaps)."""
    p = CAL if kind == 'cal' else TRK
    cm, ct = p['carrier']
    out = [('carrier A', cm, -GAP / 2 - ct, -GAP / 2)]
    y = -GAP / 2
    for name, mat, t in half_layer_slabs(kind):
        out.append((f'{name} A', mat, y, y + t))
        y += t
    y_b = GAP / 2
    tail = []
    for name, mat, t in half_layer_slabs(kind):
        if name == 'chip':
            continue                                   # the proton crosses half B's flex but not its chip here
        tail.append((f'{name} B', mat, y_b - t, y_b))
        y_b -= t
    out.append(('air', 'G4_AIR', y, y_b))
    out += tail[::-1]
    out.append(('carrier B', cm, GAP / 2, GAP / 2 + ct))
    if kind == 'cal':
        am, at = p['absorber']
        out.append(('absorber', am, GAP / 2 + ct, GAP / 2 + ct + at))
    return out


def _box(sim, name, mother, size, y, material, z=0.0, translations=None):
    mm = gate.g4_units.mm
    v = sim.add_volume('Box', name)
    v.mother = mother
    v.size = [s * mm for s in size]
    v.material = material
    if translations is not None:
        v.translation = [[t[0] * mm, t[1] * mm, t[2] * mm] for t in translations]
    else:
        v.translation = [0, y * mm, z * mm]
    return v


def extent(n_chips=N_CHIPS, n_bands=N_BANDS):
    """A detector's lateral size: strings of n_chips chips along x, n_bands strings in z. The plates and flex sheets
    extend 10 and 5 mm beyond the chips on every side."""
    string_x = n_chips * CHIP_X
    z_lo = -n_bands * PIXELS_Z / 2
    return {'n_chips': n_chips, 'n_bands': n_bands, 'string_x': string_x, 'z_lo': z_lo,
            'sensitive_mm': [string_x, n_bands * PIXELS_Z],
            'plate_xz': (string_x + 20.0, n_bands * PIXELS_Z + 2 * (CHIP_Z - PIXELS_Z) + 20.0),
            'flex_xz': (string_x + 10.0, n_bands * PIXELS_Z + 2 * (CHIP_Z - PIXELS_Z) + 10.0)}


def _add_layer_type(sim, kind, positions, ext):
    """One layer volume (air box spanning the layer) placed at every position, with its slabs and chips as
    daughters."""
    PLATE_XZ, FLEX_XZ, STRING_X, z_lo = ext['plate_xz'], ext['flex_xz'], ext['string_x'], ext['z_lo']
    p = CAL if kind == 'cal' else TRK
    cm, ct = p['carrier']
    lo = -GAP / 2 - ct
    hi = GAP / 2 + ct + (p['absorber'][1] if kind == 'cal' else 0.0)
    centre = (lo + hi) / 2
    layer = sim.add_volume('Box', f'{kind}_layer')
    mm = gate.g4_units.mm
    layer.size = [PLATE_XZ[0] * mm, (hi - lo) * mm, PLATE_XZ[1] * mm]
    layer.material = 'G4_AIR'
    layer.translation = [[0, (y + centre) * mm, 0] for y in positions]
    ln = layer.name

    def local(y):
        return y - centre

    _box(sim, f'{kind}_carrierA', ln, (PLATE_XZ[0], ct, PLATE_XZ[1]), local(-GAP / 2 - ct / 2), cm)
    _box(sim, f'{kind}_carrierB', ln, (PLATE_XZ[0], ct, PLATE_XZ[1]), local(GAP / 2 + ct / 2), cm)
    if kind == 'cal':
        am, at = p['absorber']
        _box(sim, f'{kind}_absorber', ln, (PLATE_XZ[0], at, PLATE_XZ[1]), local(GAP / 2 + ct + at / 2), am)
    chip_t = p['chip']
    epis = []
    for half, sign in (('A', 1.0), ('B', -1.0)):
        y = -sign * GAP / 2                              # the carrier's face, then into the gap
        for name, mat, t in half_layer_slabs(kind):
            yc = y + sign * t / 2
            if name == 'chip':
                bands = range(0 if half == 'A' else 1, ext['n_bands'], 2)
                zc = [z_lo + b * PIXELS_Z + CHIP_Z / 2 for b in bands]
                chip = _box(sim, f'{kind}_chip{half}', ln, (STRING_X, t, CHIP_Z), None, mat,
                            translations=[[0, local(yc), z] for z in zc])
                epi_y = sign * (chip_t / 2 - EPI / 2)          # the side facing the gap
                epi_z = -CHIP_Z / 2 + PIXELS_Z / 2             # the matrix at the chip's lower edge
                epis.append(_box(sim, f'{kind}_epi{half}', chip.name, (STRING_X, EPI, PIXELS_Z), epi_y, 'G4_Si',
                                 z=epi_z))
            else:
                _box(sim, f'{kind}_{name}{half}', ln, (FLEX_XZ[0], t, FLEX_XZ[1]), local(yc), mat)
            y += sign * t
    return layer, epis


def add_bergen_dtc(sim, n_layers=41, shift=0.0, si_cut_mm=0.01, n_chips=N_CHIPS, n_bands=N_BANDS):
    """Add the DTC to the world: Bergen's when n_chips = 9 and n_bands = 12, a larger detector of the same layers
    otherwise. Returns (sensitive volume names, info dict with every sensor plane's y)."""
    add_materials(sim)
    ext = extent(n_chips, n_bands)
    trk_y = [y + shift for y in TRACKER_Y]
    cal_y = [CAL_Y0 + PITCH * k + shift for k in range(n_layers)]
    _, trk_epi = _add_layer_type(sim, 'trk', trk_y, ext)
    _, cal_epi = _add_layer_type(sim, 'cal', cal_y, ext)
    mm = gate.g4_units.mm
    for kind in ('trk', 'cal'):
        for half in 'AB':
            sim.physics_manager.set_production_cut(f'{kind}_chip{half}', 'all', si_cut_mm * mm)
    layer_y = trk_y + cal_y
    info = {'n_layers': n_layers, 'n_chips': n_chips, 'n_bands': n_bands, 'sensitive_mm': ext['sensitive_mm'],
            'shift_mm': shift, 'layer_y_mm': layer_y, 'si_cut_mm': si_cut_mm,
            'sensor_y_mm': sensor_planes(layer_y), 'tracker_y_mm': trk_y, 'calorimeter_y_mm': cal_y}
    return [v.name for v in trk_epi + cal_epi], info


def sensor_planes(layer_y):
    """The centre y of each epitaxial layer: plane 2 l is layer l's half A, 2 l + 1 its half B (layers 0 and 1 are the
    trackers)."""
    out = []
    for l, y in enumerate(layer_y):
        chip_t = (TRK if l < 2 else CAL)['chip']
        d = GAP / 2 - (GLUE + POLYIMIDE + FLEX_AL + chip_t) + EPI / 2
        out += [y - d, y + d]
    return out


def pixel_index(x, z, n_chips=N_CHIPS, n_bands=N_BANDS):
    """Global pixel column (1024 per chip: 0..9215 for Bergen's 9 chips; -1 in the 60 um between two chips'
    matrices or off the detector) and row (band * 512 + row: 0..6143 for 12 bands; -1 outside the bands), and the
    half (0 = A, 1 = B) that covers z."""
    u = x + n_chips * CHIP_X / 2
    chip = np.floor(u / CHIP_X).astype(np.int64)
    within = u - chip * CHIP_X
    col = np.floor(within / PIXEL_X).astype(np.int64)
    ok_x = (chip >= 0) & (chip < n_chips) & (col < 1024)
    col = np.where(ok_x, chip * 1024 + col, -1)
    v = z + n_bands * PIXELS_Z / 2
    band = np.floor(v / PIXELS_Z).astype(np.int64)
    row = np.floor((v - band * PIXELS_Z) / PIXEL_Z).astype(np.int64)
    ok_z = (band >= 0) & (band < n_bands)
    row = np.where(ok_z, band * 512 + np.minimum(row, 511), -1)
    return col, row, np.where(ok_z, band % 2, -1)
