# Data provenance

| File | Source run | Notes |
|---|---|---|
| e1_ltsf_sampled.csv | E1, 300 windows/dataset, H in {96,192,336,720}, L=512, seed 42 | |
| e2_interventions_summary.csv, e2_interventions_grouped.csv | E2, 100 windows, H=96 | P1 noise: sigma_eps = 0.1 * sigma_context (amplitude SNR 10) |
| diag_p1_solar_summary.csv | Solar P1 diagnostic | snrinf = noise-free; snr10 = same noise as E2 |
| e3_synthetic_summary.csv | E3 synthetic generators | no bootstrap CIs recorded |
| e5_grouping_summary.csv | E5 single pass (32-variate cap bypassed), H=96, 50 windows x 64 stratified targets | 3,200 forecasts per strategy and K |
| e5_official_chunked_from_log.csv | E5 with reference evaluator (32-variate blocks), 50 windows; extracted from master_pipeline.log | windows/targets drawn separately from the single-pass run (K=1 errors differ) |

All CSVs record `git_commit = unknown_commit`; the authors confirmed that all runs used
commit `b06beb70d6ddc485df972f7b3f5710573b87e0a0` (short: `b06beb7`).
