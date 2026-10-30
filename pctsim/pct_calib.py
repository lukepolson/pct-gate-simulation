"""WEPL calibration for the GATE pCT scans: the kinetic energy at the rear plane of ENERGY MeV protons (default 230)
after water slabs 0..T_MAX mm thick (default 300, every 10 mm), in the scan's geometry and physics (pct_scan.py), the
rear plane at REAR_Y mm (default 225.35, the head scan's). WEPL(E) is the slab thickness at which the mean exit energy is
E.

Each thickness is its own GATE process (one central spot, water slab centred on the isocentre, 400 x 400 mm across),
up to WORKERS at once (default 1), one BLAS thread each, below normal priority. Writes OUT_DIR/calib.npz: thickness
(mm), mean and sd of the rear energy (MeV) after a 3 sigma cut, the fraction of primaries that reached the rear plane,
and the energy and rear plane used.

Usage: python -I pct_calib.py OUT_DIR N_PER_THICKNESS [ENERGY REAR_Y T_MAX WORKERS]      (drives all thicknesses)
       python -I pct_calib.py OUT_DIR N one THICKNESS SEED ENERGY REAR_Y               (one thickness; the driver)
"""
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import LOW_PRIORITY  # noqa: E402

ONE_THREAD = {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                               'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')}


def one(out_dir, n, t, seed, energy, rear_y):
    import opengate as gate
    from opengate.contrib.beamlines.ionbeamline import BeamlineModel
    from opengate.contrib.tps.ionbeamtherapy import SpotInfo
    mm, m = gate.g4_units.mm, gate.g4_units.m
    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False
    sim.output_dir = out_dir
    sim.random_engine = 'MersenneTwister'
    sim.random_seed = seed
    sim.progress_bar = False
    sim.world.size = [1.2 * m, 1.6 * m, 1.0 * m]
    sim.world.material = 'G4_AIR'
    if t > 0:
        w = sim.add_volume('Box', 'water')
        w.size = [400 * mm, t * mm, 400 * mm]
        w.material = 'G4_WATER'
        sim.physics_manager.set_production_cut('water', 'all', 1 * mm)
    rear = sim.add_volume('Box', 'rear')
    rear.size = [500 * mm, 0.001 * mm, 400 * mm]
    rear.translation = [0, rear_y * mm, 0]
    rear.material = 'G4_AIR'
    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMY'
    sim.physics_manager.set_production_cut('world', 'all', 1 * m)
    beam = BeamlineModel()
    beam.name = 'bergen'
    beam.radiation_types = 'proton'
    beam.distance_nozzle_iso = 500.0
    beam.distance_stearmag_to_isocenter_x = 6780.0
    beam.distance_stearmag_to_isocenter_y = 7430.0
    beam.energy_mean_coeffs = [energy]
    beam.energy_spread_coeffs = [0.0]
    beam.sigma_x_coeffs = beam.sigma_y_coeffs = [3.0]
    beam.theta_x_coeffs = beam.theta_y_coeffs = [0.0028]
    beam.epsilon_x_coeffs = beam.epsilon_y_coeffs = [0.003]
    s = SpotInfo(0.0, 0.0, 1.0, energy)
    s.beamFraction = 1.0
    tps = sim.add_source('TreatmentPlanPBSource', 'beam')
    tps.beam_model = beam
    tps.particle = 'proton'
    tps.n = n
    tps.beam_data_dict = {'spots': [s], 'gantry_angle': 0.0}
    fp = sim.add_filter('ParticleFilter', 'protons')
    fp.particle = 'proton'
    ps = sim.add_actor('PhaseSpaceActor', 'ps_rear')
    ps.attached_to = 'rear'
    ps.attributes = ['EventID', 'TrackID', 'KineticEnergy', 'PrePosition', 'PreDirection']
    ps.output_filename = f'rear_{t:03d}.root'
    ps.steps_to_store = 'entering'
    ps.filters.append(fp)
    sim.run()


if __name__ == '__main__':
    out_dir, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 3 and sys.argv[3] == 'one':
        one(out_dir, n, int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
        sys.exit(0)
    energy = float(sys.argv[3]) if len(sys.argv) > 3 else 230.0
    rear_y = float(sys.argv[4]) if len(sys.argv) > 4 else 225.35
    t_max = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    thicknesses = list(range(0, t_max + 1, 10))
    todo = [(k, t) for k, t in enumerate(thicknesses) if not (out_dir / f'rear_{t:03d}.root').exists()]
    running = []
    while todo or running:
        running = [p for p in running if p.poll() is None]
        while todo and len(running) < workers:
            k, t = todo.pop(0)
            running.append(subprocess.Popen(
                [sys.executable, '-I', __file__, str(out_dir), str(n), 'one', str(t), str(300000 + k), str(energy),
                 str(rear_y)], stdout=subprocess.DEVNULL, stderr=open(out_dir / f'rear_{t:03d}.err', 'w'),
                env={**os.environ, **ONE_THREAD}, **LOW_PRIORITY))
        time.sleep(1)
    import uproot
    rows = []
    for t in thicknesses:
        a = uproot.open(out_dir / f'rear_{t:03d}.root')['ps_rear'].arrays(library='np')
        _, first = np.unique(a['EventID'][a['TrackID'] == 1], return_index=True)
        e = a['KineticEnergy'][a['TrackID'] == 1][first]
        for _ in range(2):                                   # 3 sigma cut, iterated
            mu, sd = np.median(e), np.std(e)
            e = e[np.abs(e - mu) < 3 * sd]
        rows.append((t, e.mean(), e.std(), len(first) / n))
        print(f'{t:3d} mm water: rear energy {e.mean():7.2f} MeV, sd {e.std():.2f} MeV, reached {len(first) / n:.3f}')
    r = np.array(rows)
    np.savez(out_dir / 'calib.npz', thickness=r[:, 0], energy=r[:, 1], energy_sd=r[:, 2], reached=r[:, 3],
             beam_energy=energy, rear_y=rear_y)
    # WEPL sd from the energy sd: the slope of thickness against energy
    slope = np.gradient(r[:, 0], r[:, 1])
    for t, sd, s in zip(r[:, 0], r[:, 2], slope):
        if t in (100, 200, 250, 300, 350):
            print(f'  WEPL sd at {t:.0f} mm: {abs(s) * sd:.2f} mm')
    print('wrote', out_dir / 'calib.npz')
