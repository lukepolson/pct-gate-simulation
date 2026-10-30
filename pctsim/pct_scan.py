"""One block of views of a GATE 10 proton CT scan of a voxelized phantom (the head, the abdomen), in a Bergen-like
geometry.

Frame (GATE world, mm): the beam travels along +y, z is vertical and the rotation axis, x is lateral. The phantom sits
at the origin with its LPS axes along the world axes at view angle 0 (x patient left, y posterior, z superior), so view
0 is anterior to posterior. View v turns it by theta_v = 360 v / N_VIEWS_TOTAL degrees about +z (step and shoot).

Beam: GATE's pencil-beam treatment-plan source with the Bergen beam measured from Zenodo 7426388 and Solie 2020:
ENERGY MeV protons (default 230), nozzle 500 mm before the isocentre, scanning magnets 6780 mm (x) and 7430 mm
(vertical) before it, spot sigma 3.0 mm, divergence 2.8 mrad, emittance 0.003 mm rad, divergent. The spot grid (7 mm
pitch, given at the isocentre plane) is x -X_HALF..X_HALF mm (default 126: 37 columns) and z Z_CENTRE +- Z_HALF mm.
Spots are fired in order, an equal share each, so a proton's spot follows from its EventID.

Detectors: two ideal planes of air, 'front' before the phantom and 'rear' at the first Bergen tracker (y = +225.35 mm),
or farther out for a large phantom; max(500, 2 X_HALF + 300) x 400 mm (x, z). Both sit at least 15 mm outside the circle
the phantom's image box sweeps as it turns (front y = -(half-diagonal + 15) mm: -165 mm for the head's 197 x 226 mm
box; rear y = max(225.35, half-diagonal + 15) mm), so they never overlap. Each stores the primary protons entering it:
EventID, RunID (the view within this block), position (mm), direction and kinetic energy (MeV).

Usage: python -I pct_scan.py OUT_DIR TAG FIRST_VIEW N_VIEWS N_VIEWS_TOTAL PER_VIEW Z_CENTRE Z_HALF SEED PHANTOM_MHD
                             [ENERGY [X_HALF]] [static]
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

import opengate as gate
from opengate.contrib.beamlines.ionbeamline import BeamlineModel
from opengate.contrib.tps.ionbeamtherapy import SpotInfo

REAR_Y = 225.35
PITCH = 7.0
PLANE_GAP = 15.0


def swept_radius(mhd):
    """Radius of the circle the image box's corners sweep turning about z (mm): its half-diagonal in x, y."""
    h = dict(ln.split(' = ', 1) for ln in pathlib.Path(mhd).read_text().splitlines() if ' = ' in ln)
    size = np.array(h['DimSize'].split(), dtype=float) * np.array(h['ElementSpacing'].split(), dtype=float)
    return float(np.hypot(size[0] / 2, size[1] / 2))

if __name__ == '__main__':
    out_dir, tag = pathlib.Path(sys.argv[1]), sys.argv[2]
    first, n_views, n_total, per_view = (int(v) for v in sys.argv[3:7])
    z_centre, z_half, seed = float(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9])
    phantom = sys.argv[10]
    numbers = [a for a in sys.argv[11:] if a != 'static']
    energy = float(numbers[0]) if numbers else 230.0
    x_half = float(numbers[1]) if len(numbers) > 1 else 126.0
    out_dir.mkdir(parents=True, exist_ok=True)
    mm, m, MeV, gcm3, sec = (gate.g4_units.mm, gate.g4_units.m, gate.g4_units.MeV, gate.g4_units.g_cm3,
                             gate.g4_units.s)

    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False
    sim.output_dir = out_dir
    sim.random_engine = 'MersenneTwister'
    sim.random_seed = seed
    sim.progress_bar = False
    sim.world.size = [1.2 * m, 1.6 * m, 1.0 * m]
    sim.world.material = 'G4_AIR'

    # the head, turned by theta_v about +z in run v
    head = sim.add_volume('Image', 'head')
    head.image = phantom
    head.material = 'G4_AIR'
    data = gate.utility.get_data_folder()
    head.voxel_materials, _ = gate.geometry.materials.HounsfieldUnit_to_material(
        sim, 0.05 * gcm3, str(data / 'Schneider2000MaterialsTable.txt'), str(data / 'Schneider2000DensitiesTable.txt'))
    views = list(range(first, first + n_views))
    angles = [360.0 * v / n_total for v in views]
    rotations = [Rotation.from_euler('z', a, degrees=True).as_matrix() for a in angles]
    head.rotation = rotations[0]
    if n_views > 1 and 'static' not in sys.argv[11:]:      # 'static': keep view 0's angle (debugging)
        head.add_dynamic_parametrisation(rotation=rotations, translation=[[0.0, 0.0, 0.0]] * n_views)
    sim.run_timing_intervals = [(k * sec, (k + 1) * sec) for k in range(n_views)]

    # ideal tracking planes, outside the circle the head's box sweeps
    r_swept = swept_radius(phantom)
    FRONT_Y = -(r_swept + PLANE_GAP)
    rear_y = max(REAR_Y, r_swept + PLANE_GAP)
    plane_x = max(500.0, 2 * x_half + 300.0)
    planes = {}
    for name, y in (('front', FRONT_Y), ('rear', rear_y)):
        p = sim.add_volume('Box', name)
        p.size = [plane_x * mm, 0.001 * mm, 400 * mm]
        p.translation = [0, y * mm, 0]
        p.material = 'G4_AIR'
        planes[name] = p

    # physics
    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMY'
    sim.physics_manager.set_production_cut('world', 'all', 1 * m)
    sim.physics_manager.set_production_cut('head', 'all', 1 * mm)

    # beam: the Bergen spot-scanned pencil beam
    beam = BeamlineModel()
    beam.name = 'bergen'
    beam.radiation_types = 'proton'
    beam.distance_nozzle_iso = 500.0
    beam.distance_stearmag_to_isocenter_x = 6780.0
    beam.distance_stearmag_to_isocenter_y = 7430.0
    beam.energy_mean_coeffs = [energy]
    beam.energy_spread_coeffs = [0.0]
    beam.sigma_x_coeffs = [3.0]
    beam.theta_x_coeffs = [0.0028]
    beam.epsilon_x_coeffs = [0.003]
    beam.sigma_y_coeffs = [3.0]
    beam.theta_y_coeffs = [0.0028]
    beam.epsilon_y_coeffs = [0.003]
    xs = np.arange(-x_half, x_half + 1e-6, PITCH)
    nz_half = int(round(z_half / PITCH * 2)) / 2
    zs = z_centre + PITCH * (np.arange(int(2 * nz_half) + 1) - nz_half)
    spots = []
    for zz in zs:
        for xx in xs:
            s = SpotInfo(float(xx), float(zz), 1.0, energy)
            s.beamFraction = 1.0 / (len(xs) * len(zs))
            spots.append(s)
    tps = sim.add_source('TreatmentPlanPBSource', 'beam')
    tps.beam_model = beam
    tps.particle = 'proton'
    tps.n = per_view
    tps.sorted_spot_generation = True
    tps.beam_data_dict = {'spots': spots, 'gantry_angle': 0.0}

    # phase spaces of protons (the threshold filter takes only floating-point attributes, so the primaries,
    # TrackID 1, are kept when the files are converted)
    fp = sim.add_filter('ParticleFilter', 'protons')
    fp.particle = 'proton'
    for name in planes:
        ps = sim.add_actor('PhaseSpaceActor', f'ps_{name}')
        ps.attached_to = name
        ps.attributes = ['EventID', 'RunID', 'TrackID', 'KineticEnergy', 'PrePosition', 'PreDirection']
        ps.output_filename = f'{name}_{tag}.root'
        ps.steps_to_store = 'entering'
        ps.filters.append(fp)
    stats = sim.add_actor('SimulationStatisticsActor', 'stats')

    t0 = time.time()
    sim.run()
    dt = time.time() - t0
    info = {'tag': tag, 'views': views, 'angles': angles, 'per_view': per_view, 'spots_x': xs.tolist(),
            'spots_z': zs.tolist(), 'n_spots': len(spots), 'seed': seed, 'seconds': dt,
            'events': int(stats.counts.events), 'front_y': FRONT_Y, 'rear_y': rear_y, 'phantom': phantom,
            'energy_MeV': energy, 'x_half': x_half, 'plane_x_mm': plane_x}
    (out_dir / f'info_{tag}.json').write_text(json.dumps(info, indent=1))
    print(stats)
    print(f'{tag}: {n_views} views x {per_view} protons, {len(spots)} spots, {dt:.0f} s, '
          f'{info["events"] / max(dt, 1e-9):.0f} events/s')
