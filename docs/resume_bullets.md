# Resume bullets: Time Series Forecasting Benchmark

Every number below is measured, and the source file is named next to it. Pick 2-3 bullets
per resume. The `Resumes/` folder has not been edited.

**Project line:** Time Series Forecasting Benchmark | Python, statsforecast, statsmodels,
Prophet, LightGBM, SciPy, Streamlit, Docker, GitHub Actions |
github.com/GunjanPR/04-time-series-forecasting

1. Improved MASE by 26-28% against seasonal naive with an ETS + ARIMA + Theta combination,
   ranked first of 10 methods on two independent samples of 500 M4-Monthly series (1,500
   rolling-origin series-windows each, h = 18). The gain was confirmed with per-series
   Diebold-Mariano tests, Holm-corrected Wilcoxon tests (p = 3e-51) and Friedman-Nemenyi
   rank tests. *(docs/results/m4-monthly/report.md, docs/results/m4-monthly-seed7/report.md)*

2. Re-ran the whole benchmark on a second, independent sample to test the conclusions:
   the 10-method ranking replicated (Spearman 0.92) and Prophet was last on both samples
   (10-13% worse than seasonal naive, 61% coverage of 95% intervals), but AutoETS's
   statistical tie with the best method did not (Nemenyi mean-rank gap 0.73 > CD 0.61). It
   stays within 4-5% of the best MASE at 5% of the compute (150 ms vs 2.9 s per series).
   *(README, "Robustness"; docs/results/m4-monthly-seed7/sample_comparison.csv)*

3. Cut the LightGBM prediction-interval score (MSIS) by 51% (22.4 → 11.0) at unchanged 95%
   coverage across 1,500 series-windows, by making split-conformal intervals scale with
   each series' volatility. *(README, "Optimization")*

4. Tested whether the conclusions transfer to a second domain (342 electricity clients,
   daily, 1,026 series-windows). The ranking flipped: a global LightGBM was most accurate
   (−3.0% MASE vs seasonal naive, Holm p = 1e-6), while three of the statistical models
   were no better than the baseline. *(docs/results/electricity-daily/report.md)*

5. Fixed under-covering LightGBM prediction intervals on daily load data (95% intervals
   covered 90.0%) by calibrating split-conformal errors over a year of forecast origins
   instead of the last 14 days: coverage 90.0% → 94.0%, MSIS −6.6% (11.64 → 10.87), the
   best interval score of 10 methods, with point forecasts and runtime unchanged.
   *(README, "Optimization 2")*

6. Built the benchmark as reusable infrastructure: 10 forecasters behind one interface,
   resumable runs, MSIS and OWA metrics, a "where baselines win" analysis by STL
   trend/seasonal strength, a Streamlit dashboard, Docker, and CI that runs a real
   M4 smoke benchmark on every push (65 tests, coverage floor).

**Short version (one line):** Benchmarked 10 forecasting methods on 2 public datasets with
rolling-origin CV and significance tests. The best method changed with the data (a
statistical combination −26 to −28% MASE on M4, a global LightGBM best on electricity load), and
Prophet came last on both.
