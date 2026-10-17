# NF4 analysis plot browser

This Streamlit app is adapted from the original gradient-flow plot browser in
`/Users/yaman/Contbetafn/src/betafn/plot_browser`.

The **Analysis plots** tab reads Fit 1 through Fit 12 results from
`/Volumes/T7/ContinuousBetaFunction/hpcc_outputs`. Standard and thinned
continuum results are discovered from the output hierarchy, including all
available `dt_*` spacings and both diagonal and correlated covariance modes.
Fits 1–3 use their completed no-prior polynomial orders 3–5. Fit 4 provides
its no-prior and available prior-width configurations. Fits 5–7 use only their
no-prior result folders, and Fits 8–10 use only the current zero-fixed-coefficient
variants. For Fit 12, only the corrected
`order_*_rescaled_a0_standard_continuum_nopriors` directories are displayed;
the older `order_*_joint_a0_continuum_nopriors` directories are excluded.

The **Infinite-volume extrapolation** tab reads the completed Fit 4 no-prior
orders from T7. The **Chiral extrapolation** tab reads the six order-level
PDFs for $g_{\rm GF}^2$ and $\beta_{\rm GF}$ at bare couplings 7, 7.25, and
7.5. Its fit and order selectors use the same dynamic T7 discovery as the
analysis tab, so newly copied Fit 6, 9, and 10 results require only a rescan.

PDF figures are preferred throughout whenever a matching PDF and PNG both
exist, preserving vector-quality zooming. The infinite-volume tab is fixed to
the `fit4_nopriors` results and offers only its order selector.

The browser has five result sections:

1. **Analysis plots** presents Fit 1–12 standard or thinned results from T7.
   It includes interpolation plots for unthinned
   analyses, the order-level processed massless-limit and massless
   infinite-volume input plots, the complete selected diagonal/correlated
   continuum set, the covariance comparison, Figure 11 and extended
   diagnostics when available, and the order-level blinded Lambda summary.
   PDF files are preferred.
2. **Infinite-volume extrapolation** is a separate, order-level section. These
   figures do not depend on the later flow-time-window selection:

- `infinite_volume_all_beta_fit4`
- `infinite_volume_beta_vs_invV_fit4`
- `infinite_volume_beta_over_g4_vs_invV_fit4`

3. **Chiral extrapolation** displays the six strong-coupling order-level plots.
4. **Autocorr / binsize diagnostics** retains the original diagnostic filters.
5. **Data summary table** is one unfiltered table reporting configuration
   counts for every available bare coupling and volume.

Install and run with:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For convenience during development, the app also detects fit4 folders in the
parent directory. A folder pasted beside `app.py` takes precedence.
