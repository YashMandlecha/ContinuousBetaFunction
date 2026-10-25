# Hybrid weak/strong interpolation and model averaging

## Scope and compatibility

The standard single-interpolation pipeline is unchanged and remains in the
fit-specific notebooks.  The hybrid-only analysis is in
`nf4_hybrid_weak_fit4_fit11_pade_bma.ipynb`; it contains no copied standard-fit
branch.  Its batch counterpart is `scripts/run_nf4_hybrid_bma.py`, submitted by
`scripts/slurm/nf4_hybrid_bma.sbatch`.  The reusable implementation is in
`fitting/hybrid.py`, `fitting/model_average.py`, and `fitting/goodness.py`.

## Data partition

The partition is made with bare-coupling labels, never by fitting or cutting on
noisy renormalized couplings:

- weak: beta_b = 20 through 8;
- strong: beta_b = 7.5, 7.25, 7.125, and 7, after the chiral and
  infinite-volume limits.

At each flow time and operator, the lower and upper edges of the matching gap
are the posterior means of g_GF^2 at beta_b = 8 and 7.5.  Matching fractions
are no longer separate discrete models.  Every candidate has one global
matching fraction shared by all flow times and operators.  With

    q_match ~ Normal(0,1),
    f_match = 0.2 + 0.6 Phi(q_match),

the induced prior is exactly Uniform(0.2,0.8).  The physical matching point
for each target is

    x_m = x_8 + f_match (x_7.5 - x_8).

The interior bounds avoid a poorly conditioned attachment directly at either
neighboring data point while allowing the data to determine the location.

## Candidate interpolation

The weak branch is selectable between Fit 4 and Fit 11, with the allowed order
specified separately for each family.  The current production choice uses
order 4 for both.  With
x = g_GF^2 and u = x/(4 pi), Fit 4 is

    W(x) = beta_PT,3(x) [1 + sum_{k=1}^N c_k u^k],

and Fit 11 adds a finite-flow constant a0.

The strong branch is a Pade-style deformation,

    beta(x) = W(x) + x^2 z^q P_m(z) / Q_n(z),
    z = (x - x_m) / Delta_x.

The default q=2 gives C1 matching.  q=3 is available for C2 sensitivity
tests.  The current production comparison uses [2/1] and [3/2], both of which
make the rational factor asymptotically linear in z.  Lower orders remain
available as explicit sensitivity models.

For denominator order n,

    Q_n(z) = product_{j=1}^n (1 + z/rho_j),
    0 < rho_1 < rho_2 < ... < rho_n.

The first scale is exponential in a fitted log parameter.  Every subsequent
scale is the preceding scale plus an exponential positive increment.  All
poles therefore lie at negative z and cannot enter the physical strong branch
z >= 0.  The strict ordering removes the root-label permutation degeneracy,
which is essential when comparing marginal likelihoods.

## Evidence and priors

Every candidate is one joint errors-in-variables fit to every selected flow
time and all three operators in a continuum window.  The g_GF^2 values are
latent variables carrying the complete IV covariance.  Consequently `logGBF`
is a joint Gaussian marginal likelihood for the window.  The code does not sum
evidences from correlated one-dimensional fits.

All coefficient priors are proper, because prior-free marginal likelihood is
undefined.  Defaults are:

- weak multiplicative coefficients: 0 +/- 5;
- Fit 11 additive constant: 0 +/- 0.25;
- coefficients of the x^2-scaled strong correction: 0 +/- 0.10;
- first log pole distance and each ordered log increment: 0 +/- 0.70.
- global matching fraction: Uniform(0.2,0.8), implemented through the latent
  standard-normal transformation above.

These are analysis choices, not universal constants.  Final results require a
prior-predictive check and a preregistered width-sensitivity analysis.  All
candidate models receive equal prior probability by default.  Because the
Cartesian grid is balanced, this also gives equal marginal prior probability
to each varied weak family, weak order, Pade order, and matching continuity.

## Propagation and uncertainty budget

The continuum limit is performed separately for every complete candidate.
Predictions are then combined with the law of total covariance:

    Cov(Y) = E_M[Cov(Y | M)] + Cov_M(E[Y | M]).

The first term is called the within-model statistical covariance and the
second the interpolation-model systematic covariance.  The non-Gaussian model
mixture is only moment matched for plotting and linear summaries.  Nonlinear
Lambda integrals are evaluated candidate by candidate before averaging.

Shapley variance effects divide the between-model variance among the discrete
choices that actually vary: weak family, weak order, Pade order, and matching
continuity.  Matching-location uncertainty is marginalized inside every
candidate and therefore enters its within-model posterior covariance.  This
allocation includes interactions and sums to the between-model variance.
Reported percentages are partial relative standard uncertainties,
100 sqrt(V_k)/|mean beta|; they do not add linearly.

## Exact candidate-level parallelism and restartability

Candidate models are conditionally independent calculations using the same
joint data vector.  The implementation evaluates them in separate processes
and never divides a candidate by operator, flow time, or coupling.  Before
workers start, the parent constructs one joint vector containing every latent
g_GF^2 value and beta value and extracts its full mean and covariance.  Each
worker reconstructs that exact joint Gaussian object, uses the unchanged
errors-in-variables likelihood and proper priors, and returns its coefficient
posterior with `gvar.dumps`, which preserves the complete within-model
covariance.  Model weights remain normalized from the same `logGBF` values.

Every completed candidate is written atomically to a signature-checked
checkpoint.  The signature contains the full data mean and covariance,
target geometry, perturbative coefficients, candidate definition, and prior
widths, so a checkpoint cannot be reused after a scientifically relevant
input changes.  Results are restored in the configured candidate order,
independent of process completion order.  Workers restrict internal BLAS and
OpenMP pools to one thread; SLURM parallelism is therefore across candidate
models without CPU oversubscription.  Adding models only extends the worker
queue and does not require changing the scheduling logic.

The notebook and batch runner also produce candidate-level interpolation
diagnostics without changing any fit.  Each weak-family/order, Pade-order,
matching-fraction, and operator combination has its own PDF/PNG; selected C1
and C2 matching variants are overlaid in that figure.  Separate plots retain
the model-averaged curve.  `hybrid_fitted_coefficients.csv` is the
machine-readable coefficient ledger and `hybrid_fitted_coefficients.txt` is
the corresponding grouped text table.  Both include coefficient means and
standard deviations, joint log evidence, and posterior model probability.

Progress reporting is diagnostic only.  Processing, chiral and
infinite-volume stages use the existing verbosity machinery; hybrid
interpolation and continuum stages report candidate counts, elapsed time, and
an empirical ETA; plot generation reports completed figures.  No progress
setting enters a likelihood, prior, covariance, fit, or model weight.

The fixed-fit notebooks additionally report the statistical component and the
available continuum-window and covariance-treatment sensitivities.  They mark
chiral-, IV-, and interpolation-model systematics as not estimated when the
corresponding alternative models have not been run.  A single deterministic
pipeline cannot uniquely assign the same underlying Monte Carlo fluctuations
to successive fit stages, so such a split must not be fabricated.

## Bruno--Sommer diagnostics

Continuum fits retain ordinary chi-squared, dof, and p-values and also report
the generalized expectation and p-value of Bruno and Sommer.  For actual data
covariance C, weight square root W, Jacobian J, and

    P = W J (J^T W^2 J)^(-1) J^T W,

the null distribution is

    chi^2 = sum_k lambda_k z_k^2,

where lambda_k are the eigenvalues of

    C^(1/2) W (1-P) W C^(1/2).

This calibrates goodness of fit when diagonal, shrinkage, or otherwise
regularized weights differ from C.  It does not estimate C, propagate
covariance-estimator uncertainty, or contribute another error term.  For the
linear continuum model the Jacobian is parameter independent, so the Appendix
A uncertainty in P vanishes.  The generic implementation nevertheless computes
the model Hessian, Eq. (A.3) sandwich covariance, Eq. (A.5) expectation
gradient, and the corresponding uncertainty in `E[chi2]`; the saved value is
therefore an explicit zero-result check rather than an assumption.

## Required validation before a physics result

1. Perform prior-predictive plots for every candidate family.
2. Repeat the BMA for at least two defensible prior-width sets.
3. Inspect posterior model weights for collapse onto prior boundaries.
4. Verify no physical-interval poles and negative beta throughout the Lambda
   integration domain for every model with non-negligible weight.
5. Compare evidence weights with blocked predictive stacking as a sensitivity
   analysis; do not call stacking probabilities Bayesian posterior model
   probabilities.
6. Add explicit alternative chiral and IV ansatzes before quoting those two
   model-systematic components.
7. Keep all Lambda outputs blinded until analysis choices are frozen.
