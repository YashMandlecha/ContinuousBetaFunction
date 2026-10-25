import numpy as np
import gvar as gv

from betafn.processing.bartlett import (
    _bartlett_long_run_covariance,
    _lugsail_long_run_covariance,
    automatic_bartlett_bandwidth,
    bartlett_spectral_average,
    bartlett_spectral_covariance,
    lugsail_bartlett_spectral_covariance,
    original_bartlett_spectral_covariance,
    overlapping_batch_means_covariance,
    plugin_bartlett_bandwidth,
)


def _manual_bartlett(centered, bandwidth):
    n = len(centered)
    result = centered.T @ centered / n
    for lag in range(1, bandwidth + 1):
        lag_covariance = centered[:-lag].T @ centered[lag:] / n
        result += (1.0 - lag / (bandwidth + 1.0)) * (
            lag_covariance + lag_covariance.T
        )
    return result


def test_moving_sum_form_matches_bartlett_lag_window_exactly():
    values = np.random.default_rng(123).normal(size=(37, 5))
    centered = values - values.mean(axis=0)
    actual = _bartlett_long_run_covariance(centered, bandwidth=7)
    expected = _manual_bartlett(centered, bandwidth=7)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    assert np.linalg.eigvalsh(actual).min() >= -1e-12


def test_covariance_is_covariance_of_mean_and_reports_sensitivity_checks():
    rng = np.random.default_rng(7)
    n = 400
    innovations = rng.normal(size=(n, 2))
    values = np.zeros_like(innovations)
    for index in range(1, n):
        values[index] = 0.75 * values[index - 1] + innovations[index]

    means, covariance, diagnostics = bartlett_spectral_covariance(
        values, bandwidth=12
    )
    np.testing.assert_allclose(means, values.mean(axis=0))
    expected = _manual_bartlett(values - values.mean(axis=0), 12) / n
    np.testing.assert_allclose(covariance, expected, rtol=2e-14, atol=2e-14)
    assert diagnostics.bandwidth == 12
    assert diagnostics.scan_lower_bandwidth == 6
    assert diagnostics.scan_upper_bandwidth == 24
    assert np.isfinite(diagnostics.obm_median_se_ratio)
    assert np.isfinite(diagnostics.scan_max_abs_log_se_ratio)


def test_joint_average_preserves_between_operator_covariance():
    rng = np.random.default_rng(11)
    common = rng.normal(size=300)
    data = {
        "wilson_Ep_4.0": common + 0.15 * rng.normal(size=300),
        "wilson_Ec_4.0": 0.8 * common + 0.15 * rng.normal(size=300),
    }
    averaged, diagnostics = bartlett_spectral_average(
        data, bandwidth=5, return_diagnostics=True
    )
    correlation = gv.evalcorr(list(averaged.values()))
    assert correlation[0, 1] > 0.8
    assert diagnostics.n_observables == 2


def test_automatic_bandwidth_is_common_and_bounded():
    values = np.random.default_rng(19).normal(size=(80, 25))
    bandwidth, tau = automatic_bartlett_bandwidth(
        values, max_fraction=0.2
    )
    assert 2 <= bandwidth <= 16
    assert tau.shape == (25,)


def test_overlapping_batch_means_covariance_is_symmetric_psd():
    values = np.random.default_rng(23).normal(size=(120, 4))
    covariance = overlapping_batch_means_covariance(values, batch_size=10)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_zero_lugsail_is_published_linear_combination():
    values = np.random.default_rng(29).normal(size=(180, 3))
    centered = values - values.mean(axis=0)
    actual, vanilla, inner = _lugsail_long_run_covariance(
        centered, 18, r=2.0, c=0.5
    )
    assert inner == 9
    expected = 2.0 * vanilla - _bartlett_long_run_covariance(centered, 9)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_plugin_bandwidth_is_bounded_and_reacts_to_persistence():
    rng = np.random.default_rng(31)
    white = rng.normal(size=(1200, 3))
    persistent = np.zeros_like(white)
    for index in range(1, len(persistent)):
        persistent[index] = 0.9 * persistent[index - 1] + white[index]
    white_bandwidth, white_details = plugin_bartlett_bandwidth(white)
    persistent_bandwidth, persistent_details = plugin_bartlett_bandwidth(persistent)
    assert 1 <= white_bandwidth <= 300
    assert 1 <= persistent_bandwidth <= 300
    assert persistent_bandwidth > white_bandwidth
    assert persistent_details["plugin_unbounded_bandwidth"] > white_details[
        "plugin_unbounded_bandwidth"
    ]


def test_lugsail_public_wrapper_is_psd_and_reports_adjustment_state():
    rng = np.random.default_rng(37)
    values = rng.normal(size=(500, 5))
    for index in range(1, len(values)):
        values[index] += 0.8 * values[index - 1]
    _, covariance, diagnostics = lugsail_bartlett_spectral_covariance(values)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    assert diagnostics.lugsail_r == 2.0
    assert diagnostics.lugsail_c == 0.5
    assert diagnostics.bandwidth_method == "plugin"


def test_original_bartlett_preset_reproduces_former_defaults():
    values = np.random.default_rng(41).normal(size=(300, 4))
    actual_mean, actual_covariance, diagnostics = (
        original_bartlett_spectral_covariance(values)
    )
    expected_mean, expected_covariance, _ = bartlett_spectral_covariance(
        values, bandwidth_method="self-consistent", lugsail_r=1.0
    )
    np.testing.assert_allclose(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_covariance, expected_covariance)
    assert diagnostics.estimator == "bartlett"
    assert diagnostics.bandwidth_method == "self-consistent"
