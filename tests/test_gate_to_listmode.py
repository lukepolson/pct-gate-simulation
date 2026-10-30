"""Tests of gate_to_listmode.py on small inputs (CPU): the reader's cuts and WEPL, and the most likely entries.

Usage: python -m pytest tests -q   (needs PyTomography 4)
"""
import json
import math

import numpy as np
import pytest
import torch

from gate_to_listmode import most_likely_entries, read_gate
from pytomography.metadata.PCT import PCTProjMeta
from pytomography.metadata.PCT.pct_metadata import FWHM_PER_SD


def _calibration(path):
    thickness = np.arange(0.0, 490.0, 10.0)
    np.savez(path, thickness=thickness, energy=330.0 - 0.4 * thickness)
    return path


def test_most_likely_entry_combines_spot_and_exit(tmp_path):
    entry = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    # exits whose straight lines back reach the entry plane 2 mm and 60 mm from the spot's ray (the second: no fit)
    exit = torch.tensor([[2.0, -1.0, 0.0, 0.0], [60.0, 0.0, 0.0, 0.0]])
    pm = PCTProjMeta([0.0], -250.0, 250.0, entry, exit, [2], 330e3, entry_fwhm=3.69 * FWHM_PER_SD,
                     beam_divergence=math.degrees(0.763e-3))
    out = most_likely_entries(pm, torch.zeros(2), _calibration(tmp_path / 'calib.npz'), 1.34e-3, device='cpu')
    sb = 1.34e-3 * 500                                  # no water: the exit direction's error alone
    gain = 3.69 ** 2 / (3.69 ** 2 + sb ** 2)
    assert out.entry[0, :2].tolist() == pytest.approx([2.0 * gain, -1.0 * gain], rel=1e-5)
    assert float(out.entry_fwhm[0]) == pytest.approx((3.69 ** -2 + sb ** -2) ** -0.5 * FWHM_PER_SD, rel=1e-5)
    assert out.entry[0, 2] == pytest.approx(0.763e-3 * 2.0 * gain, rel=1e-4)
    assert out.entry[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert float(out.entry_fwhm[1]) == pytest.approx(3.69 * FWHM_PER_SD, rel=1e-6)


def test_read_gate(tmp_path):
    rng = np.random.default_rng(0)
    for v in range(2):
        n = 400
        front = np.zeros((n, 6), np.float32)
        front[:, 0] = rng.uniform(-10, 10, n)
        front[:, 1] = rng.uniform(-5, 5, n)
        front[:, 3], front[:, 5] = 1.0, 330.0
        rear = front.copy()
        rear[:, 5] = 330.0 - 0.4 * 100.0 + rng.uniform(-0.3, 0.3, n)  # 100 mm of water, no tails to cut
        rear[:3, 5] = 250.0                                               # three nuclear events: cut
        rear[3, 0] = np.nan                                               # missed the rear plane
        np.savez(tmp_path / f'view_{v:03d}.npz', front=front, rear=rear)
        (tmp_path / f'info_{v:03d}.json').write_text(json.dumps({'angles': [90.0 * v], 'front_y': -60.0, 'rear_y': 60.0}))
    wepl, pm = read_gate(tmp_path, _calibration(tmp_path / 'calib.npz'), cut_bin=50.0, device='cpu')
    assert pm.view_sizes.tolist() == [396, 396]
    assert float(wepl.mean()) == pytest.approx(100.0, abs=0.5)
    assert pm.angles.tolist() == [0.0, 90.0]
