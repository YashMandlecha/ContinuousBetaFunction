from __future__ import annotations

import numpy as np

from betafn.fit12 import (
    continuum_beta,
    interpolation_beta,
    interpolation_p0,
    joint_beta,
    joint_design_matrix,
    joint_parameter_names,
    lattice_slope,
    rescale_interpolation_parameters,
    rescaled_interpolation_beta,
    solve_linear_gls,
)
from betafn.perturbative import PerturbativeBetaFunction


def test_finite_spacing_interpolation_has_free_a0_and_fixed_c0():
    perturbative = PerturbativeBetaFunction(nf=4, nc=3)
    x = np.asarray([1.0, 2.0, 4.0])
    parameters = interpolation_p0(3)
    parameters["beta_const"] = [0.07]
    parameters["pt_c1"] = [-0.3]
    expected = 0.07 + perturbative(x, loops=3) * (
        1.0 - 0.3 * x / perturbative.nrm
    )
    np.testing.assert_allclose(
        interpolation_beta(x, parameters, perturbative, 3), expected
    )


def test_rescaling_free_constant_at_fixed_flow_time_is_invariant():
    """Writing a0=z*A0 cannot change a finite-time interpolation curve."""
    perturbative = PerturbativeBetaFunction(nf=4, nc=3)
    x = np.asarray([1.0, 2.0, 4.0])
    z = 0.2
    a0 = 0.013
    A0 = a0 / z
    direct = interpolation_p0(4)
    direct["beta_const"] = [a0]
    for power in range(1, 5):
        value = (-1.0) ** power * 0.1 * power
        direct[f"pt_c{power}"] = [value]
    rescaled = rescale_interpolation_parameters(direct, z, 4)

    assert rescaled["A0"][0] == A0
    assert rescaled["_fit12_z"][0] == z

    np.testing.assert_allclose(
        interpolation_beta(x, direct, perturbative, 4),
        rescaled_interpolation_beta(x, rescaled, perturbative, 4),
        rtol=0.0,
        atol=0.0,
    )


def test_joint_model_enforces_perturbative_continuum_intercept():
    perturbative = PerturbativeBetaFunction(nf=4, nc=3)
    order = 4
    parameters = {
        name: [0.1 * (index + 1)]
        for index, name in enumerate(joint_parameter_names(order))
    }
    x = np.asarray([0.8, 1.5, 3.0, 4.5])
    u = x / perturbative.nrm
    expected_continuum = perturbative(x, loops=3) * (
        1.0 + sum(
            parameters[f"c{power}"][0] * u**power
            for power in range(1, order + 1)
        )
    )
    np.testing.assert_allclose(
        np.asarray(
            continuum_beta(x, parameters, perturbative, order), dtype=float
        ),
        expected_continuum,
    )

    z = np.asarray([0.0, 0.05, 0.1, 0.2])
    expected = expected_continuum + z * np.asarray(
        lattice_slope(x, parameters, perturbative, order), dtype=float
    )
    np.testing.assert_allclose(
        np.asarray(joint_beta(x, z, parameters, perturbative, order), dtype=float),
        expected,
    )


def test_joint_design_and_block_gls_recover_known_coefficients():
    perturbative = PerturbativeBetaFunction(nf=4, nc=3)
    order = 3
    names = joint_parameter_names(order)
    assert names == ("A0", "c1", "c2", "c3")
    expected = np.linspace(-0.3, 0.4, len(names))
    designs = []
    targets = []
    covariances = []
    for x in (1.0, 2.0, 3.0, 4.0, 4.8):
        z = np.asarray([0.12, 0.18, 0.25, 0.32])
        design = joint_design_matrix(
            np.full(z.shape, x), z, perturbative, order
        )
        designs.append(design)
        targets.append(design @ expected)
        covariances.append(np.diag(np.linspace(0.01, 0.02, len(z)) ** 2))

    solution = solve_linear_gls(designs, targets, covariances)
    assert solution["rank"] == len(names)
    assert solution["n_data"] == 20
    np.testing.assert_allclose(
        solution["coefficients"], expected, rtol=1e-9, atol=1e-9
    )
    assert solution["chi2"] < 1e-12


def test_high_order_joint_gls_is_full_rank_after_column_normalization():
    perturbative = PerturbativeBetaFunction(nf=4, nc=3)
    for order in (4, 5):
        expected = np.linspace(-0.2, 0.25, len(joint_parameter_names(order)))
        designs = []
        targets = []
        covariances = []
        for x in np.linspace(0.9, 4.9, 9):
            z = 1.0 / np.linspace(3.0, 8.0, 11)
            design = joint_design_matrix(
                np.full(z.shape, x), z, perturbative, order
            )
            designs.append(design)
            targets.append(design @ expected)
            covariances.append(np.eye(len(z)))
        solution = solve_linear_gls(designs, targets, covariances)
        assert solution["rank"] == len(expected)
        np.testing.assert_allclose(
            solution["coefficients"], expected, rtol=1e-6, atol=1e-6
        )
