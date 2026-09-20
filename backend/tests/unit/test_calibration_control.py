"""CalibratedClassifierCV reliability diagram is a train/deploy control."""

import numpy as np

from calibration import calibration_deploy_control, reliability_diagram


def test_perfect_calibration_is_near_diagonal():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 20)
    p = y.astype(float)
    diagram = reliability_diagram(y, p, n_bins=5)
    gate = calibration_deploy_control(diagram)
    assert diagram["near_diagonal"] is True
    assert gate["calibration_ok"] is True
    assert gate["kelly_cap_if_miscalibrated"] is None


def test_miscalibrated_probabilities_fail_the_deploy_control():
    y = np.array([0, 1] * 80)
    p = np.full(y.shape, 0.9)
    diagram = reliability_diagram(y, p, n_bins=8)
    gate = calibration_deploy_control(diagram)
    assert gate["calibration_ok"] is False
    assert gate["kelly_cap_if_miscalibrated"] == 0.01
    assert gate["method"] == "CalibratedClassifierCV"
