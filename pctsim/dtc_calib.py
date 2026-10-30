"""WEPL calibration of the DTC itself: the range the detector measures (dtc_range.py: tracker points, Bragg-Kleeman
fit, nuclear filter) behind water slabs 0..T_MAX mm thick (every 10 mm), in a scan's geometry: pct_calib.py's slab and
central spot, the DTC moved back by SHIFT mm as in dtc_scan.py, and the scan's ideal exit plane (3 mm before the first
tracker) kept as truth. WEPL(R) is the slab thickness at which the mean measured range is R.

Each thickness is its own GATE process, up to WORKERS at once, one BLAS thread each, below normal priority. Writes
OUT_DIR/calib.npz in pct_calib.py's form, so io.PCT.read_gate reads it unchanged: 'energy' holds the mean measured
range (mm WET, after a 3 sigma cut) and 'energy_sd' its sd, 'reached' the fraction of primaries the DTC kept; also
'quantity' = 'range_wet_mm', the truth exit energy ('exit_energy', MeV) and the counts of each loss. Its first row is
a straight extension to -EXTEND_MM, so a noisy range beyond the no-water mean gives a negative WEPL, not 0.

Usage: python -I dtc_calib.py OUT_DIR N_PER_THICKNESS ENERGY SHIFT T_MAX WORKERS N_LAYERS N_CHIPS N_BANDS MATERIALS_JSON
       python -I dtc_calib.py OUT_DIR N one THICKNESS SEED ENERGY SHIFT N_LAYERS N_CHIPS N_BANDS     (one thickness)
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LOW_PRIORITY  # noqa: E402
EXTEND_MM = 100.0                # the table is extended straight on to -EXTEND_MM (see the end)
ONE_THREAD = {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                               'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')}


def one(out_dir, n, t, seed, energy, shift, n_layers, n_chips, n_bands):
    import opengate as gate
    from opengate.contrib.beamlines.ionbeamline import BeamlineModel
    from opengate.contrib.tps.ionbeamtherapy import SpotInfo
    import dtc_geometry as dg
    from pct_scan import REAR_Y
    mm, m = gate.g4_units.mm, gate.g4_units.m
    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False
    sim.output_dir = out_dir
    sim.random_engine = 'MersenneTwister'
    sim.random_seed = seed
    sim.progress_bar = False
    sim.check_volumes_overlap = False
    sim.world.size = [1.2 * m, 2.2 * m, 1.0 * m]
    sim.world.material = 'G4_AIR'
    exit_y = REAR_Y + shift - 3.0
    assert t / 2 < exit_y, (t, exit_y)
    if t > 0:
        w = sim.add_volume('Box', 'water')
        w.size = [400 * mm, t * mm, 400 * mm]
        w.material = 'G4_WATER'
        sim.physics_manager.set_production_cut('water', 'all', 1 * mm)
    ex = sim.add_volume('Box', 'exit')
    ex.size = [500 * mm, 0.001 * mm, 400 * mm]
    ex.translation = [0, exit_y * mm, 0]
    ex.material = 'G4_AIR'
    epis, dtc_info = dg.add_bergen_dtc(sim, n_layers=n_layers, shift=shift, n_chips=n_chips, n_bands=n_bands)
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
    ps = sim.add_actor('PhaseSpaceActor', 'ps_exit')
    ps.attached_to = 'exit'
    ps.attributes = ['EventID', 'TrackID', 'KineticEnergy', 'PrePosition', 'PreDirection']
    ps.output_filename = f'exit_{t:03d}.root'
    ps.steps_to_store = 'entering'
    ps.filters.append(fp)
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'hits')
    hits.attached_to = epis
    hits.attributes = ['EventID', 'TrackID', 'ParentID', 'PDGCode', 'PostPosition', 'TotalEnergyDeposit']
    hits.output_filename = f'hits_{t:03d}.root'
    hits.authorize_repeated_volumes = True
    sim.run()
    (out_dir / f'info_{t:03d}.json').write_text(json.dumps({'thickness': t, 'n': n, 'seed': seed, 'exit_y': exit_y,
                                                            'dtc': dtc_info}, indent=1))


def analyse(out_dir, t, n, materials):
    """Measured range per proton behind t mm of water, the truth exit energies, and the loss counts."""
    import uproot
    import dtc_range as dr
    from dtc_convert import COLS, clusters
    info = json.loads((out_dir / f'info_{t:03d}.json').read_text())
    stack = dr.Stack(info['dtc'], materials)
    a = uproot.open(out_dir / f'hits_{t:03d}.root')['hits'].arrays(COLS, library='np')
    c = clusters(a, stack.sensor_y, stack.n_chips, stack.n_bands)
    rear, counts = dr.measure(c, np.arange(n), stack, info['exit_y'])
    x = uproot.open(out_dir / f'exit_{t:03d}.root')['ps_exit'].arrays(library='np')
    _, first = np.unique(x['EventID'][x['TrackID'] == 1], return_index=True)
    return rear[:, 5], x['KineticEnergy'][x['TrackID'] == 1][first], counts


if __name__ == '__main__':
    out_dir, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 3 and sys.argv[3] == 'one':
        one(out_dir, n, int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]),
            *(int(v) for v in sys.argv[8:11]))
        sys.exit(0)
    energy, shift, t_max, workers = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    n_layers, n_chips, n_bands = (int(v) for v in sys.argv[7:10])
    materials = json.loads(pathlib.Path(sys.argv[10]).read_text())
    thicknesses = list(range(0, t_max + 1, 10))
    todo = [(k, t) for k, t in enumerate(thicknesses) if not (out_dir / f'info_{t:03d}.json').exists()]
    running = []
    t0 = time.time()
    while todo or running:
        running = [p for p in running if p.poll() is None]
        while todo and len(running) < workers:
            k, t = todo.pop(0)
            running.append(subprocess.Popen(
                [sys.executable, '-I', __file__, str(out_dir), str(n), 'one', str(t), str(400000 + k), str(energy),
                 str(shift), str(n_layers), str(n_chips), str(n_bands)], stdout=subprocess.DEVNULL,
                stderr=open(out_dir / f'run_{t:03d}.err', 'w'),
                env={**os.environ, **ONE_THREAD, 'PYTHONIOENCODING': 'utf-8'}, **LOW_PRIORITY))
            time.sleep(2)
        time.sleep(2)
    print(f'GATE runs done in {time.time() - t0:.0f} s')
    rows, loss = [], []
    for t in thicknesses:
        r, e_exit, counts = analyse(out_dir, t, n, materials)
        r = r[np.isfinite(r)]
        kept = len(r)
        for _ in range(2):                                   # 3 sigma cut, iterated, as pct_calib.py
            mu, sd = np.median(r), np.std(r)
            r = r[np.abs(r - mu) < 3 * sd]
        e_exit = e_exit[np.abs(e_exit - np.median(e_exit)) < 3 * np.std(e_exit)]
        rows.append((t, r.mean(), r.std(), kept / n, e_exit.mean()))
        loss.append([counts[k] for k in ('protons', 'tracked', 'ranged', 'kept', 'reached_last_layer')])
        print(f'{t:3d} mm water: range {r.mean():7.2f} mm WET, sd {r.std():.2f}, kept {kept / n:.3f} '
              f'(tracked {counts["tracked"] / n:.3f}, ranged {counts["ranged"] / n:.3f}); exit energy '
              f'{rows[-1][4]:.2f} MeV')
    # Extend the table straight on to -EXTEND_MM (a line through 0-50 mm): a proton through air has a measured range
    # 7 mm (sd) either side of the mean, and the lookup holds its end values outside the table, so without this every
    # air or skin chord's WEPL is held at >= 0, a bias of sd / sqrt(2 pi) = +2.7 mm that FBP turns into a bright rim
    # and a darker inside (29 Oct, the 330 MeV abdomen).
    b, a0 = np.polyfit(np.array([r[0] for r in rows[:6]]), np.array([r[1] for r in rows[:6]]), 1)
    rows.insert(0, (-EXTEND_MM, a0 - b * EXTEND_MM, rows[0][2], rows[0][3], np.nan))
    a = np.array(rows)
    np.savez(out_dir / 'calib.npz', thickness=a[:, 0], energy=a[:, 1], energy_sd=a[:, 2], reached=a[:, 3],
             exit_energy=a[:, 4], beam_energy=energy, rear_y=json.loads((out_dir / 'info_000.json').read_text())[
                 'exit_y'], quantity='range_wet_mm', loss=np.array(loss),
             loss_columns=np.array(['protons', 'tracked', 'ranged', 'kept', 'reached_last_layer']))
    slope = np.gradient(a[:, 0], a[:, 1])
    for t, sd, s in zip(a[:, 0], a[:, 2], slope):
        if t in (100, 200, 250, 300, 350):
            print(f'  WEPL sd at {t:.0f} mm: {abs(s) * sd:.2f} mm (measured range sd x d thickness / d range)')
    print('wrote', out_dir / 'calib.npz')
