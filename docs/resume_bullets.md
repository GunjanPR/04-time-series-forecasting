# Resume bullets: Time Series Forecasting Benchmark

Every number below is measured, and the source file is named next to it. Pick 2-3 bullets
per resume. The `Resumes/` folder has not been edited.

**Project line:** Time Series Forecasting Benchmark | Python, statsforecast, statsmodels,
Prophet, LightGBM, SciPy, Streamlit, Docker, GitHub Actions |
github.com/GunjanPR/04-time-series-forecasting

1. Improved MASE by 28.0% against seasonal naive across 1,500 rolling-origin series-windows
   (500 M4-Monthly series, h = 18) with an ETS + ARIMA + Theta combination. The gain was
   confirmed with per-series Diebold-Mariano tests, Holm-corrected Wilcoxon tests
   (p = 3e-51) and a Friedman-Nemenyi rank test across 10 methods.
   *(docs/results/m4-monthly/report.md)*

2. Showed that AutoETS is statistically tied with the best method (Nemenyi) while using 5%
   of its single-core compute (150 ms vs 2.9 s per series), and that Prophet was 9.7%
   worse than seasonal naive, with 95% intervals that covered only 61% of outcomes.
   *(docs/results/m4-monthly/leaderboard.csv, cost_profile_single_core.csv)*

3. Cut the LightGBM prediction-interval score (MSIS) by 51% (22.4 → 11.0) at unchanged 95%
   coverage across 1,500 series-windows, by making split-conformal intervals scale with
   each series' volatility. *(README, "Optimization")*

4. Tested whether the conclusions transfer to a second domain (342 electricity clients,
   daily, 1,026 series-windows). The ranking flipped: a global LightGBM was most accurate
   (−3.0% MASE vs seasonal naive, Holm p = 1e-6), while three of the statistical models
   were no better than the baseline. *(docs/results/electricity-daily/report.md)*

5. Built the benchmark as reusable infrastructure: 10 forecasters behind one interface,
   resumable runs, MSIS and OWA metrics, a "where baselines win" analysis by STL
   trend/seasonal strength, a Streamlit dashboard, Docker, and CI that runs a real
   M4 smoke benchmark on every push (61 tests).

**Short version (one line):** Benchmarked 10 forecasting methods on 2 public datasets with
rolling-origin CV and significance tests. The best method changed with the data (a
statistical combination −28% MASE on M4, a global LightGBM best on electricity load), and
Prophet came last on both.
