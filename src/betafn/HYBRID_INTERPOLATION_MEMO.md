# Hybrid weak/strong interpolation and model averaging

## Current weak-ansatz revision — 2026-10-27

The user reran the C1 bridge and requested a rethinking of the ansatz itself.
The regenerated beta/g^4 plots still had a large hump, and the median
augmented chi2/dof was about 3.9 for all four old candidates. The old gap
diagnostics also exposed incompatible endpoint slopes. Merely improving
join differentiability was not sufficient.

The hybrid notebook and batch runner now select `monotone` and
`monotone_constant`, each with N=6, 7, and 8, combined with the Eq. (7)
[1/2], [2/3], and [3/4] strong forms:

    W(x) = a0 - 2*b0*integral_0^x s sum_k w_k B_{k,N}(s/L) ds
    L = 4*pi, w0 = 1, w_k = exp(q_k), q_k ~ N(0,1.5^2).

Here b0=0.0527714498137..., B are Bernstein polynomials, and a0 is zero
for `monotone` or has the unchanged N(0,0.25^2) prior for
`monotone_constant`. `weak_width` is the standard deviation of q for these
families; stored coefficient keys are `logw_0_k`. There are N slope-shape
parameters, and W is a polynomial of degree N+2. It has the one-loop anchor
W=a0-b0*x^2+O(x^3), but does not inherit the spurious three-loop zero.
Its slope is strictly negative on 0<x<=L. This shape restriction is an
explicit model assumption motivated by the plotted finite-time data; it is
not a universal statement about all beta functions. In particular it does
not itself impose monotonicity of beta/x^2.

The positive-coefficient Bernstein representation follows the general
shape-restricted polynomial strategy discussed by Wang and Ghosh (2012):
https://www.sciencedirect.com/science/article/pii/S0167947312001004
The specific anchored derivative and priors above are our analysis choices,
not a formula claimed to be in that paper. The integral is evaluated exactly
as finite Bernstein sums, including analytic first/second derivatives; there
is no numerical quadrature inside the fitter. A preflight check rejects
weak data/join geometry outside the declared basis interval.

The C1 gap-only bridge and the independent Equation-7 strong likelihood remain.
The established `fit4`--`fit11` weak definitions remain selectable and
unchanged. The notebook configuration lists Fits 5--10 as commented examples,
so experimenting with one requires only uncommenting its family/order entry.
All families use proper Gaussian coefficient priors in the hybrid analysis,
because evidence is undefined for improper prior-free parameters. Reproducing
the earlier Fit4/Fit11 prior specification requires `weak_width=5`, not the
current log-weight width. No global fit over flow times or operators is used.

The current output root/analysis identifier stays stable. New families have distinct
checkpoint names, so their interpolation must be fitted anew. Old Fit4/Fit11
checkpoints are never reused for the new likelihood. The notebook's stale
downstream displays were cleared; upstream displays and existing result
files were retained. Rerun configuration, then the independent interpolation
section and its interpolation/quality plot cells; inspect fit quality before
continuum.

Validation includes analytic integral versus quadrature, analytic derivatives,
strictly negative weak slopes, absence of the inherited PT3 zero, correlated
gvar matching between the optimized and reference evaluators, C1 joins for
both beta and beta/x^2, and a 15-point synthetic errors-in-variables fit
through the production fitter and checkpoint restoration. This is code and
synthetic validation, not evidence of an acceptable fit to the actual data;
the full notebook interpolation has deliberately not been run by the agent.

## Earlier joining correction — 2026-10-27 (historical)

The hybrid notebook introduced `eq7_linear`, `continuity=1`. The internal
form name and the analysis/output root are retained for compatibility. This
mode means a **gap-only C1 rational Hermite bridge**, not an exactly linear
segment. The older descriptions below document historical alternatives.

Inspection of the saved beta and beta/g^4 candidate plots found that the C2
shoulders changed the curve between the measured beta_b=8.5 and 8 points.
The labelled-data likelihood still used the original weak polynomial there.
That plotting/prediction inconsistency is removed from the notebook's selected
mode: the exact weak branch is now used through x_8 plus 5% of the gap to
x_7.5, and the exact strong branch is used from x_7.5 onward. Both endpoint
values and slopes are joined only inside the unmeasured interval.

The numerical bridge is Delbourgo--Gregory's rational cubic Hermite formula,
TR/10/83, Eqs. (2.3)--(2.7):
https://bura.brunel.ac.uk/bitstream/2438/2200/1/TR_10_83.pdf
For secant D and endpoint derivatives d0,d1, the chosen tension is
r=3+(d0/D-1)^2+(d1/D-1)^2. It is >=3 and >=(d0+d1)/D. Hence the sufficient
monotonicity condition in their Eq. (3.8) holds when both slopes agree in
sign with D. A nearly zero central secant uses ordinary Hermite, without a
monotonicity guarantee. Opposite-sign slopes are retained and reported in
`hybrid_gap_diagnostics.csv`: clipping them would break differentiability.
This prescription is distinct from the physics paper's Eq. (7), which
motivates the strong branch. The positive ordered-pole parameterization is
a restriction of that paper's general denominator. Neither the 5% extension
nor the bridge is claimed to be fitted or statistically preferred by the
data in the empty gap.

All endpoint coefficient correlations, including the tension dependence, are
propagated using gvar. Geometry remains conditional on measured endpoint
means, as previously. Gap modelling and endpoint-location systematics are
not automatically part of the statistical band. Existing C0/C2 coefficient
checkpoints can be reused for C1 only when their data, covariance, prior and
candidate signatures match: these modes have identical likelihoods. Curves
and all subsequent continuum/Lambda outputs still need regenerating.

There is also an unresolved fit-quality issue, separate from the join:
the saved Wilson t/a^2=6 Fit4 order4 [1/2] candidate has augmented
chi2/dof=118.250161/15. Its c4=32.0(2.7) is far from its 0+/-5 prior, and
its beta derivative turns positive near the weak endpoint. A smooth join
cannot make that underlying fit adequate. Changes to orders/priors were
proposed to the user; none were silently applied. The notebook quality plot
now explicitly labels the statistics as augmented/nominal lsqfit values.
In addition, the current Nf=4 GF three-loop prefactor has a positive zero
at x=8.778622164910281 (computed from `PerturbativeBetaFunction(4,3).b`).
Fit4 necessarily inherits that zero; extra multiplicative polynomial orders
cannot remove it. Fit11 can shift the value through its additive constant.
This truncated-perturbative zero is not established nonperturbative physics.
Any weak-domain extension approaching it must be checked explicitly before
claiming a smooth, well-fitting physical interpolation.

Validation: regression tests in `fitting/test_hybrid_gap.py` check exact
weak/strong preservation (including gvar correlations), prepared/reference
likelihood consistency at data points, C1 joins for beta and beta/x^2,
monotonicity for compatible extreme slopes, covariance versus a numerical
Jacobian, and finite Eq. (7) derivatives at removable singularities. No
upstream ensemble processing was run for this correction.

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
are no longer parameters or separate discrete models. The weak curve is kept
through x_8 plus 5% of the empty gap, the independent strong curve starts at
x_7.5, and the two endpoint values and derivatives are connected by the C1
rational Hermite bridge described above. The 5% extension is a fixed modelling
choice and should eventually be varied as a gap-prescription systematic.

## Candidate interpolation

The active weak branches are the anchored monotone and
monotone-with-constant families at orders 6--8. Fits 4--11 remain optional,
with the order specified independently for each family in the notebook's
single model-selection dictionary. With x = g_GF^2 and u = x/(4 pi), Fit 4 is

    W(x) = beta_PT,3(x) [1 + sum_{k=1}^N c_k u^k],

and Fit 11 adds a finite-flow constant a0. Fits 5--10 use exactly the same
fixed/free coefficient conventions as the standard analysis runner.

The independent strong branch is the paper's Eq. (7)-style ratio,

    beta_strong(x) = -p0 x^2 P_N(u) / Q_{N+1}(u),
    u = x/(4 pi),

with p0 positive, P_N(0)=1, and the currently selected [1/2], [2/3], and
[3/4] orders. The extra denominator power makes beta(x) asymptotically linear.
The weak and strong fits are not statistically stitched through a shared fit
parameter; their posterior endpoint jets are connected only in the data-free
gap.

For denominator order n,

    Q_n(u) = product_{j=1}^n (1 + u/rho_j),
    0 < rho_1 < rho_2 < ... < rho_n.

The first scale is exponential in a fitted log parameter.  Every subsequent
scale is the preceding scale plus an exponential positive increment.  All
poles therefore lie at negative u and cannot enter the physical branch u >= 0.
The strict ordering removes the root-label permutation degeneracy,
which is essential when comparing marginal likelihoods.

## Evidence and priors

Every candidate is one errors-in-variables fit to the approximately 15
coupling points at one operator and one flow time. Its g_GF^2 values are latent
variables carrying that target's IV covariance. Candidate `logGBF` values are
compared only within that same target, so every operator/flow-time interpolation
has its own posterior model probabilities. Evidences are never summed across
flow times or operators.

All coefficient priors are proper, because prior-free marginal likelihood is
undefined.  Defaults are:

- active monotone log-weights: 0 +/- 1.5;
- Fit 11 additive constant: 0 +/- 0.25;
- strong numerator coefficients: 0 +/- 2;
- first log pole distance and each ordered log increment: 0 +/- 1.

Optional Fits 4--10 currently inherit the selected scalar `weak_width`; a
prior-width sensitivity study is required before comparing their evidence to
the active monotone families.

These are analysis choices, not universal constants.  Final results require a
prior-predictive check and a preregistered width-sensitivity analysis.  All
candidate models receive equal prior probability by default.  Because the
Cartesian grid is balanced, this also gives equal marginal prior probability
to each varied weak family, weak order, Pade order, and matching continuity.

## Propagation and uncertainty budget

Candidate predictions are combined locally, before the continuum limit, with
the law of total covariance:

    Cov(Y) = E_M[Cov(Y | M)] + Cov_M(E[Y | M]).

The first term is called the within-model statistical covariance and the
second the interpolation-model systematic covariance. At each g_GF^2, the
resulting local BMA predictions at different flow times are the inputs to the
ordinary continuum extrapolation. This avoids imposing an artificial global
model indicator across otherwise independent interpolations.

Shapley variance effects divide each target's between-model variance among the
discrete choices that actually vary: weak family, weak order, and Padé order
(plus continuity if it is deliberately varied). Those local contributions are
propagated through the linear continuum fit.
Reported percentages are partial relative standard uncertainties,
100 sqrt(V_k)/|mean beta|; they do not add linearly.

## Exact target/candidate parallelism and restartability

Target/candidate fits are independent calculations. A target is exactly one
operator and one flow time and contains the weak and strong coupling points
needed by the two-branch interpolation. The implementation evaluates these
tasks in separate processes and never divides a target by coupling. Before
workers start, the parent constructs one x/y mean-and-covariance object per
target. Each worker uses the unchanged errors-in-variables likelihood and
proper priors and returns its coefficient posterior with `gvar.dumps`. Model
weights are normalized separately from the candidate `logGBF` values of each
target.

Parallel serialization does not discard correlations between flow times. Each
worker also returns the fitted parameters' linear response to its local x/y
data. The parent reconstructs those parameters from the original shared gvar
data and an independent residual covariance. Local BMA model indicators remain
independent, while covariance induced by common Monte Carlo fluctuations is
averaged over both targets' model probabilities and retained in the correlated
continuum fit.

Every completed target/candidate pair is written atomically to a
signature-checked checkpoint. The signature contains the target data mean and covariance,
target geometry, perturbative coefficients, candidate definition, and prior
widths, so a checkpoint cannot be reused after a scientifically relevant
input changes. Results are restored in target and configured candidate order,
independent of process completion order. Workers restrict internal BLAS and
OpenMP pools to one thread; SLURM parallelism is therefore across
operator/flow-time/model tasks without CPU oversubscription. Adding models or
flow times only extends the worker queue.

The notebook and batch runner also produce candidate-level interpolation
diagnostics without changing any fit.  Each weak-family/order, Pade-order,
matching-fraction, and operator combination has its own PDF/PNG; selected C1
and C2 matching variants are overlaid in that figure.  Separate plots retain
the model-averaged curve.  `hybrid_fitted_coefficients.csv` is the
machine-readable coefficient ledger and `hybrid_fitted_coefficients.txt` is
the corresponding grouped text table.  Both include coefficient means and
standard deviations, target-local log evidence, and posterior model probability.

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
