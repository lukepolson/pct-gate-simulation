"""Pencil beams straight into the DTC, no phantom: one run per energy, to check the geometry (overlaps, the layers'
positions) and measure the detector's response: the energy each proton deposits in every epitaxial layer and where
each primary stops (MC truth range), and the time per proton.

The beam starts 25 mm before the first tracker (y = 200 mm) at x = 10, z = 5 mm (inside chip 4 of string band 6, half A).
Writes OUT_DIR/hits.root (every step depositing energy in an epitaxial layer), OUT_DIR/stops.root (every proton
step ending at zero kinetic energy, with the process), OUT_DIR/info.json.

Usage: python -I dtc_beam_test.py OUT_DIR N_LAYERS N_PER_ENERGY E1,E2,... [SI_CUT_MM [N_CHIPS N_BANDS]]
"""
import json
import pathlib
import sys
import time

import opengate as gate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtc_geometry as dg  # noqa: E402

out = pathlib.Path(sys.argv[1])
n_layers, n_per = int(sys.argv[2]), int(sys.argv[3])
energies = [float(e) for e in sys.argv[4].split(',')]
si_cut = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
n_chips = int(sys.argv[6]) if len(sys.argv) > 6 else dg.N_CHIPS
n_bands = int(sys.argv[7]) if len(sys.argv) > 7 else dg.N_BANDS
out.mkdir(parents=True, exist_ok=True)
mm, m, MeV, sec = gate.g4_units.mm, gate.g4_units.m, gate.g4_units.MeV, gate.g4_units.s

sim = gate.Simulation()
sim.g4_verbose = False
sim.visu = False
sim.output_dir = out
sim.random_engine = 'MersenneTwister'
sim.random_seed = 4242
sim.progress_bar = False
sim.check_volumes_overlap = True
sim.world.size = [1.2 * m, 2.2 * m, 1.0 * m]
sim.world.material = 'G4_AIR'
epis, info = dg.add_bergen_dtc(sim, n_layers=n_layers, si_cut_mm=si_cut, n_chips=n_chips, n_bands=n_bands)
sim.physics_manager.physics_list_name = 'QGSP_BIC_EMY'
sim.physics_manager.set_production_cut('world', 'all', 1 * m)
sim.run_timing_intervals = [(k * sec, (k + 1) * sec) for k in range(len(energies))]
for k, e in enumerate(energies):
    s = sim.add_source('GenericSource', f'beam{k}')
    s.particle = 'proton'
    s.energy.mono = e * MeV
    s.position.type = 'point'
    s.position.translation = [10 * mm, 200 * mm, 5 * mm]
    s.direction.type = 'momentum'
    s.direction.momentum = [0, 1, 0]
    s.n = n_per
    s.start_time = k * sec
    s.end_time = (k + 1) * sec

hits = sim.add_actor('DigitizerHitsCollectionActor', 'hits')
hits.attached_to = epis
hits.attributes = ['EventID', 'RunID', 'TrackID', 'ParentID', 'PDGCode', 'PostPosition', 'TotalEnergyDeposit']
hits.output_filename = 'hits.root'
hits.authorize_repeated_volumes = True      # one actor for every copy of the layers and chips; the plane comes from y
fp = sim.add_filter('ParticleFilter', 'protons')
fp.particle = 'proton'
fs = sim.add_filter('ThresholdAttributeFilter', 'stopped')
fs.attribute = 'PostKineticEnergy'
fs.value_min = 0.0
fs.value_max = 1e-9 * MeV
stops = sim.add_actor('PhaseSpaceActor', 'stops')
stops.attached_to = 'world'
stops.attributes = ['EventID', 'RunID', 'TrackID', 'PostPosition', 'ProcessDefinedStep']
stops.steps_to_store = 'all'
stops.filters.extend([fp, fs])
stops.output_filename = 'stops.root'
stats = sim.add_actor('SimulationStatisticsActor', 'stats')

t0 = time.time()
sim.run()
dt = time.time() - t0
info.update({'energies_MeV': energies, 'n_per_energy': n_per, 'seconds': dt, 'events': int(stats.counts.events),
             'start': [10.0, 200.0, 5.0]})
(out / 'info.json').write_text(json.dumps(info, indent=1))
print(stats)
print(f'{info["events"]} protons in {dt:.1f} s')
