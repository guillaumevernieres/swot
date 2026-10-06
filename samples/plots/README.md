# SWOT sample plots

- `swot_adt_overview.png` / `.pdf`: combined ADT map for all three passes.
- `swot_pass_227_variables`, `swot_pass_228_variables`, `swot_pass_229_variables` (PNG/PDF): nine-panel geographic maps of ADT, calibrated model-based and radiometer-based SSHA, MDT, crossover correction, the difference between the SSHA variants, their quality flags, and crossover quality.
- `summary.json`: retained pixel counts and ADT percentiles calculated at full sampling.

Reproduce from the workspace root with `python3 plot_samples.py` (requires NumPy, netCDF4, Matplotlib, Cartopy and its Natural Earth 110m coastline data).

Height panels use the conservative mask described in `../VARIABLE_GUIDE.md`. Quality panels show all available flags. Quality bitmasks are grouped by their highest severity, with the explicit degraded summary bit treated as degraded. Good quality alone does not imply an ocean measurement. Fill values remain masked.

Shared color limits across passes: ADT/MDT ±1.5 m; SSHA ±0.5 m; crossover correction ±3 m; model-minus-radiometer SSHA ±0.1 m. Colorbar extensions indicate saturation. Rendering shows every second along-track row and cross-track column without averaging; computations use all pixels. Global maps emphasize coverage and large-scale variability; local zooms are needed to examine small ocean features.
