# Monte Carlo covariance-estimation memo

Last updated: 2026-10-24

This memo records the statistical choices for estimating the covariance of
ensemble means from ordered configuration histories.  It is intended to keep
the implementation and its alternatives understandable across future analysis
sessions.  Here the history index is the configuration/trajectory number;
gradient-flow time and discretization/operator are components of the vector
measured on each configuration.

## Current production choice

The scripts and active Fit 1--4 notebooks use a **joint multivariate
zero-lugsail Bartlett estimator**.  Given the centered configuration vector
`z_n`, the vanilla Bartlett long-run covariance at bandwidth `M` is

    Sigma_B(M) = Gamma(0)
                 + sum_{k=1}^M [1-k/(M+1)] [Gamma(k)+Gamma(k)^T].

The covariance of the sample mean is `Sigma_B(M)/N`.  The production lugsail
estimate is

    Sigma_L(M) = [Sigma_B(M) - c Sigma_B(floor(M/r))] / (1-c),

with `r=2` and `c=1/2`.  This is the zero-lugsail/flat-top choice: it removes
the leading negative finite-sample bias of vanilla Bartlett under positive
serial correlation without deliberately adding the positive first-order bias
of the more conservative `r=3` choice.

Every retained flow time and operator in one ensemble is estimated jointly.
The Symanzik linear combination is made at the raw-history level first, so its
shared Wilson/rectangle fluctuations enter the covariance correctly.  No
pre-binning is applied before a spectral estimate.

The automatic bandwidth is the Liu--Vats--Flegal parametric plug-in estimate
used by modern `mcmcse` implementations:

1. Compute biased marginal autocovariances, consistent with the spectral
   estimator's `1/N` convention.
2. Fit thresholded stationary AR models to every nonconstant marginal.  Orders
   are examined up to `min(p, N-2, floor(10 log10 N))`; the first final AR
   coefficient below `qnorm(0.975)/sqrt(N)` terminates the fit.
3. Standardize each nonconstant marginal to unit variance for bandwidth
   selection only.  This makes the common bandwidth invariant to the very
   different units and scales of energy-density and topology components.
4. From each fitted AR model estimate the long-run variance `Sigma_i` and the
   leading bias constant `Gamma_i`.
5. For Bartlett/OBM, use

       M_raw = [(3/2) N sum_i Gamma_i^2 / sum_i Sigma_i^2]^(1/3).

6. Apply transparent sample-size and configured
   `M <= floor(max_fraction*N)` bounds.  The usual `mcmcse` full-rank bound
   `M <= floor(N/(p+1))` is intentionally inapplicable here because the dense
   flow-time vector has `p>N`; applying it would force `M=1`.  The scalar
   marginal pilot does not invert a `p`-dimensional covariance, while rank and
   PSD behavior of the final matrix are diagnosed separately.  The unbounded
   plug-in value and fitted AR orders are saved.

Strictly, the displayed plug-in formula minimizes the leading AMSE for vanilla
Bartlett/OBM.  A zero-lugsail removes that leading bias, so it has no nonzero
AMSE-optimal bandwidth at the same expansion order.  Following modern
`mcmcse` practice, the Bartlett plug-in value is therefore used as a stable
pilot bandwidth for lugsail, not described as an exact lugsail optimum.  The
`M/2` and `2M` scan remains mandatory.

A user-specified fixed bandwidth overrides the automatic selector.  The old
self-consistent Madras--Sokal-style selector remains available for comparison.

## Positive-semidefinite handling

Unlike vanilla Bartlett, a lugsail linear combination is not guaranteed to be
positive semidefinite at finite `N`.

- If its diagonal contains a nonpositive variance, the code reverts to the
  vanilla Bartlett matrix and records the reason.
- If the diagonal is positive but the matrix is indefinite, the correlation
  matrix is eigen-projected to positive semidefinite form and rescaled back to
  the original variances.
- Diagnostics record the raw minimum eigenvalue, whether adjustment occurred,
  the number of adjusted eigenvalues, and the relative Frobenius-norm change.

A large PSD correction is a warning about insufficient configurations,
excessive dimension, or an unstable bandwidth.  It must not be interpreted as
routine numerical cleanup.  The command-line switch `--no-spectral-psd-adjust`
instead makes an indefinite lugsail estimate fail explicitly.

## Required diagnostics and comparisons

Each ensemble records:

- selected and unbounded bandwidths, bandwidth fraction, and AR orders;
- median and maximum integrated-autocorrelation estimates from the independent
  self-consistent diagnostic;
- raw and final eigenvalue/rank information and all PSD-adjustment information;
- the median ratio of vanilla-Bartlett to lugsail standard errors;
- overlapping-batch-means versus production standard errors;
- production standard errors at approximately `M/2` and `2M`.

The scripts save these to `spectral_covariance_diagnostics.csv`.  A result
should not be trusted solely because a downstream fit has an acceptable
chi-squared: the bandwidth scan, OBM comparison, and PSD corrections must also
be inspected.

## Gamma method: retained alternative, not an extra correction

`processing/gamma.py` remains available through
`covariance_estimator='gamma'` or `--covariance-estimator gamma`.  It estimates
the same configuration-history covariance of the mean using a rectangular
Gamma-method autocovariance sum with an automatic window.  It is **mutually
exclusive** with Bartlett/lugsail in one run; uncertainties must not be
multiplied or corrected by both estimators.

The current production scripts and active notebooks therefore do not use the
Gamma method: they use lugsail Bartlett.  Gamma is retained as a sensitivity
run.  The present local Gamma implementation also has limitations: it uses a
simple positive-correlation window, has no exponential-tail correction, and
groups observable types rather than estimating the complete joint vector.

## Further options retained for future work

### Vanilla Bartlett

The explicit comparison preset is
`--covariance-estimator original-bartlett`.  It restores both defining pieces
of the pre-lugsail implementation: the vanilla Bartlett window (`r=1`) and the
former common self-consistent bandwidth.  Its results are written under a
separate family such as `fit4_nopriors_original_bartlett/order_3`, so they
cannot overwrite production lugsail outputs.  The Python API equivalents are
`original_bartlett_spectral_covariance` and
`original_bartlett_spectral_average`.

The lower-level `covariance_estimator='bartlett'` remains available when a
custom fixed or plug-in bandwidth is desired.  Vanilla Bartlett is PSD by
construction under the
biased `1/N` lag-covariance convention and remains the baseline comparison.
Its main weakness is downward finite-sample bias for persistent positive
autocorrelation.

### Conservative over-lugsail

Use lugsail with `r=3`, `c=1/2`.  It intentionally introduces positive
first-order bias and can improve coverage for highly persistent chains.  It is
a conservative systematic check, not the current central result, because
inflated covariance can artificially suppress continuum-fit chi-squared.

### Gamma method with exponential slow-mode tail

Modern lattice implementations of the Wolff Gamma method may supplement the
measured autocorrelation window with an exponential tail governed by an
externally supported slow timescale, often obtained from topology.  This can
cover a slow mode whose tail is too noisy to resolve directly.  It should be
added only when a defensible `tau_exp` and amplitude prescription exist and
reported as a slow-mode systematic; neither Bartlett nor lugsail can infer an
unobserved tail from a short history.

### Weighted/overlapping batch means

Weighted batch means are a computational representation of lag-window
estimators.  OBM is already used here as an asymptotically equivalent standard-
error cross-check.  It may become the production form if memory or observable
dimension makes the direct spectral matrix too expensive.

### Multivariate initial-sequence estimators

Multivariate initial-sequence estimators provide another principled stable
long-run covariance estimate.  They are attractive as a lower-dimensional
benchmark, but determinant/eigenvalue operations become expensive or unstable
when the number of flow-time/operator components approaches the number of
configurations.  They are not the production default for this data layout.

### Higher-order flat-top kernels and recent inference-optimal rules

Higher-order flat-top kernels can reduce truncation bias but add tuning and PSD
issues.  Inference-optimal lugsail bandwidth research is promising but too new
to displace the mature plug-in rule without dedicated validation.

### Globally centered estimators

Global centering reduces slow-mixing bias when several independent chains or
replicas target the same expectation.  It is not applicable to a single
history per ensemble.  Reconsider it if independently initialized streams are
added and retained separately.

### Configuration-level resampling

A blocked jackknife or moving/stationary block bootstrap through the nonlinear
pipeline would propagate covariance-estimation uncertainty, which any single
plug-in covariance matrix omits.  This remains the strongest eventual
validation, particularly for the final continuum and Lambda results, but it is
substantially more expensive than one analytic spectral estimate.

### Bruno--Sommer generalized chi-squared and p-values

The continuum-stage implementation follows M. Bruno and R. Sommer, *On fits to
correlated and auto-correlated data*, Computer Physics Communications 285
(2023) 108643, especially Appendix A.  This is not another estimator of the
Monte Carlo covariance and must remain conceptually separate from
Bartlett/lugsail and the Gamma method.  Its purpose is to give a statistically
correct goodness-of-fit when the matrix used to weight a fit is diagonal,
shrunk, block-regularized, SVD-cut, rank-deficient, or otherwise different
from the best estimate of the true data covariance.

For

    chi2(a) = [y - phi(a)]^T W^2 [y - phi(a)],

let `C` be the actual propagated covariance of `y`, `J` the model Jacobian at
the optimum, and

    H = J^T W^2 J,
    P = W J H^-1 J^T W.

`P` projects weighted data fluctuations onto directions that can be absorbed
by refitting the parameters; `I-P` selects the residual goodness-of-fit
directions.  Appendix A gives the parameter response and sandwich covariance

    delta_a = H^-1 J^T W^2 delta_y,
    Cov(a) = H^-1 J^T W^2 C W^2 J H^-1.

The generalized expectation of the minimized chi-squared is

    E[chi2] = tr[W C W (I-P)].

It equals `Ndata-Nparameters` only in the ideal case `W^2=C^-1`.  For arbitrary
weights define

    nu = C^(1/2) W (I-P) W C^(1/2).

If `lambda_k` are the positive eigenvalues of `nu`, the null distribution is

    chi2 = sum_k lambda_k z_k^2,

with independent standard-normal `z_k`.  The robust quality-of-fit is
`Prob(sum_k lambda_k z_k^2 >= chi2_observed)`, and the natural generalized
reduced statistic is `chi2_observed/E[chi2]`.  The corresponding variance is
`2 tr(nu^2)`.

Appendix A also differentiates `P` with respect to the fitted parameters and
combines this derivative with `Cov(a)` to estimate the part of the uncertainty
in `E[chi2]` caused by a statistically fluctuating projector.  For the standard
continuum model `beta=beta_cont+slope*(a^2/t)`, `J` is independent of the fit
parameters, so this projector-uncertainty contribution is exactly zero.  The
same is true for the present prior-free interpolation, chiral, and
infinite-volume models whenever they are linear in their coefficients.  It
must be reconsidered for genuinely nonlinear models, uncertain x coordinates,
or prior-regularized fits.

Implementation status:

1. `fitting/goodness.py` retains the propagated covariance `C` separately from
   the covariance used to construct `W`, builds `P` and `nu` with a common
   pseudoinverse tolerance, and evaluates the weighted-chi-squared tail by a
   characteristic-function inversion.
2. The continuum stage saves ordinary `chi2`, dof, and p-value alongside
   `generalized_expectation`, `generalized_effective_dof`,
   `generalized_p_value`, and `generalized_rank`; it does not alter central
   values or parameter errors.
3. Appendix A is implemented explicitly.  A numerical model Hessian constructs
   `K_alpha`, the gradient of `E[chi2]` is evaluated from Eq. (A.5), the
   sandwich parameter covariance from Eq. (A.3), and
   `generalized_expectation_parameter_sdev` stores the resulting projector
   contribution.  Unit tests verify the ordinary full-covariance limit and a
   nonlinear nonzero case.  For the present linear continuum model the model
   Hessian is zero, so this contribution evaluates to zero as required.
4. The hybrid weak/strong notebook exports candidate-level generalized
   diagnostics.  The fixed-fit notebooks include them in the continuum error
   budget table after their continuum cases are regenerated.
5. Informative-prior fits are deliberately reported without a generalized
   p-value until priors are represented as augmented Gaussian observations;
   applying the prior-free formula to them would be incorrect.

This method will make the diagonal and regularized continuum-fit p-values
meaningful and will help distinguish model failure from covariance-weighting
artifacts.  It does not improve `C` itself and does not propagate uncertainty
in the estimated covariance; lugsail/Gamma methods and blocked resampling
remain responsible for those separate problems.  The paper's reference Python
implementation is available at https://github.com/mbruno46/chiexp.

## Literature and implementations

- Vats and Flegal, *Lugsail lag windows for estimating time-average covariance
  matrices*, Biometrika 109 (2022): https://arxiv.org/abs/1809.04541
- Liu, Vats, and Flegal, *Batch size selection for variance estimators in
  MCMC*, Methodology and Computing in Applied Probability 24 (2022):
  https://arxiv.org/abs/1804.05975
- Liu and Flegal, *Weighted batch means estimators in Markov chain Monte
  Carlo*: https://arxiv.org/abs/1805.08283
- Vats, Flegal, and Jones, *Multivariate output analysis for Markov chain Monte
  Carlo*, Biometrika 106 (2019): https://arxiv.org/abs/1512.07713
- Dai and Jones, *Multivariate initial sequence estimators in Markov chain
  Monte Carlo*: https://arxiv.org/abs/1706.00853
- Current `mcmcse` reference implementation:
  https://dvats.r-universe.dev/mcmcse/doc/manual.html
- Lattice-oriented Gamma-method implementation and slow-tail interface:
  https://fjosw.github.io/pyerrors/pyerrors.html
- Bruno and Sommer, *On fits to correlated and auto-correlated data*:
  https://doi.org/10.1016/j.cpc.2022.108643
- Bruno--Sommer `chiexp` reference implementation:
  https://github.com/mbruno46/chiexp

## Reproducibility rule

Every analysis output must record the estimator name, bandwidth method,
bandwidth or override, lugsail parameters, PSD policy, and diagnostics.  A
future change to the central estimator requires explicit comparison against
the saved vanilla Bartlett, OBM, bandwidth-scan, and Gamma-method results; it
must not be made solely because one choice produces more favorable p-values.
