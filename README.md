# relight-test

Test task for MyArchitectAI: change the sun direction or time of day in an architectural image without breaking the building.

- `inputs/`: original images (sources in `inputs/SOURCES.md`)
- `outputs/<option>/<image>__<variant>.png`: results per tool, for example `outputs/sun_direction/exterior_01__sun_left_low.png`
- `results/runs.csv`: hand-filled log (time, cost, 1–5 scores, notes)
- `python relight_eval.py` → `results/metrics.csv` + before/after grids with a light-change heatmap
- `REPORT.md`: the one-page write-up
