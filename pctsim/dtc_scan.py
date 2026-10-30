"""One view of a GATE 10 proton CT scan of a voxelized phantom with the full Bergen DTC behind it (dtc_geometry.py):
the scan of pct_scan.py (same phantom, frame, rotation and Bergen pencil beam) with the two ideal planes kept only as
truth and the detector built from real volumes.

Truth: 'front' (an ideal air plane before the phantom, as in pct_scan.py) and 'exit' (an ideal air plane 3 mm before
the first tracker) store each proton entering them: EventID, position, direction, kinetic energy (MeV).
Detector: every step that deposits energy in an epitaxial layer: EventID, TrackID, ParentID, PDG code, PostPosition
(mm), TotalEnergyDeposit (MeV). The DTC is moved back by SHIFT = max(0, r_swept + 15 - 225.35) mm, so its first tracker
clears the circle the phantom's box sweeps (26.60 mm for the abdomen, 0 for the head).

Usage: python -I dtc_scan.py OUT_DIR TAG VIEW N_VIEWS_TOTAL PER_VIEW Z_CENTRE Z_HALF SEED PHANTOM_MHD ENERGY X_HALF
                             N_LAYERS [N_CHIPS N_BANDS [SI_CUT_MM]]

N_CHIPS (chips per string, 30 mm each) and N_BANDS (strings, 13.76 mm each) set the detector's width and height:
Bergen's 9 and 12 (270 x 165 mm) by default.
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dtc_geometry as dg  # noqa: E402
from pct_scan import PITCH, PLANE_GAP, REAR_Y, swept_radius  # noqa: E402

if __name__ == '__main__':
    out_dir, tag = pathlib.Path(sys.argv[1]), sys.argv[2]
    view, n_total, per_view = (int(v) for v in sys.argv[3:6])
    z_centre, z_half, seed = float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])
    phantom = sys.argv[9]
    energy, x_half, n_layers = float(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12])
    n_chips = int(sys.argv[13]) if len(sys.argv) > 13 else dg.N_CHIPS
    n_bands = int(sys.argv[14]) if len(sys.argv) > 14 else dg.N_BANDS
    si_cut = float(sys.argv[15]) if len(sys.argv) > 15 else 0.01
    out_dir.mkdir(parents=True, exist_ok=True)
    mm, m, gcm3 = gate.g4_units.mm, gate.g4_units.m, gate.g4_units.g_cm3

    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False
    sim.output_dir = out_dir
    sim.random_engine = 'MersenneTwister'
    sim.random_seed = seed
    sim.progress_bar = False
    sim.check_volumes_overlap = False                 # checked once in dtc_beam_test.py
    sim.world.size = [1.2 * m, 2.2 * m, 1.0 * m]        # y to +1.1 m: room for a deep calorimeter
    sim.world.material = 'G4_AIR'

    head = sim.add_volume('Image', 'head')
    head.image = phantom
    head.material = 'G4_AIR'
    data = gate.utility.get_data_folder()
    head.voxel_materials, _ = gate.geometry.materials.HounsfieldUnit_to_material(
        sim, 0.05 * gcm3, str(data / 'Schneider2000MaterialsTable.txt'), str(data / 'Schneider2000DensitiesTable.txt'))
    angle = 360.0 * view / n_total
    head.rotation = Rotation.from_euler('z', angle, degrees=True).as_matrix()

    r_swept = swept_radius(phantom)
    front_y = -(r_swept + PLANE_GAP)
    rear_y = max(REAR_Y, r_swept + PLANE_GAP)
    shift = rear_y - REAR_Y
    exit_y = rear_y - 3.0
    plane_x = max(500.0, 2 * x_half + 300.0)
    planes = {}
    for name, y in (('front', front_y), ('exit', exit_y)):
        p = sim.add_volume('Box', name)
        p.size = [plane_x * mm, 0.001 * mm, 400 * mm]
        p.translation = [0, y * mm, 0]
        p.material = 'G4_AIR'
        planes[name] = p
    epis, dtc_info = dg.add_bergen_dtc(sim, n_layers=n_layers, shift=shift, si_cut_mm=si_cut, n_chips=n_chips,
                                       n_bands=n_bands)

    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMY'
    sim.physics_manager.set_production_cut('world', 'all', 1 * m)
    sim.physics_manager.set_production_cut('head', 'all', 1 * mm)

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

    fp = sim.add_filter('ParticleFilter', 'protons')
    fp.particle = 'proton'
    for name in planes:
        ps = sim.add_actor('PhaseSpaceActor', f'ps_{name}')
        ps.attached_to = name
        ps.attributes = ['EventID', 'TrackID', 'KineticEnergy', 'PrePosition', 'PreDirection']
        ps.output_filename = f'{name}_{tag}.root'
        ps.steps_to_store = 'entering'
        ps.filters.append(fp)
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'hits')
    hits.attached_to = epis
    hits.attributes = ['EventID', 'TrackID', 'ParentID', 'PDGCode', 'PostPosition', 'TotalEnergyDeposit']
    hits.output_filename = f'hits_{tag}.root'
    hits.authorize_repeated_volumes = True             # one actor for every copy; the plane comes from y
    stats = sim.add_actor('SimulationStatisticsActor', 'stats')

    t0 = time.time()
    sim.run()
    dt = time.time() - t0
    info = {'tag': tag, 'views': [view], 'angles': [angle], 'per_view': per_view, 'spots_x': xs.tolist(),
            'spots_z': zs.tolist(), 'n_spots': len(spots), 'seed': seed, 'seconds': dt,
            'events': int(stats.counts.events), 'front_y': front_y, 'exit_y': exit_y, 'phantom': phantom,
            'energy_MeV': energy, 'x_half': x_half, 'plane_x_mm': plane_x, 'dtc': dtc_info}
    (out_dir / f'info_{tag}.json').write_text(json.dumps(info, indent=1))
    print(stats)
    print(f'{tag}: {per_view} protons, {len(spots)} spots, {n_layers} layers of {n_chips} x {n_bands}, {dt:.0f} s, '
          f'{info["events"] / max(dt, 1e-9):.0f} events/s')
