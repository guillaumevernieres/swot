# SWOT Level-2 Expert: SSH anomaly and absolute dynamic topography

Inspected all three September 18, 2026 Version D (PID0) sample files, cycle 56, passes 227–229. Each has 9,866 along-track lines and 69 cross-track pixels. Heights below are in meters; sampling is nominally 2 km, not a statement of effective feature resolution.

## Recommended starting fields

There is no explicit ADT variable, but the ingredients are present:

```text
calibrated_ssha = ssha_karin_2 + height_cor_xover
adt = calibrated_ssha + mean_dynamic_topography
```

The file metadata explicitly says to ADD `height_cor_xover`: this correction is not already applied to the provided SSH/SSHA fields. The sea-surface-height anomaly already subtracts mean sea surface, solid Earth tide, FES ocean tide, non-equilibrium ocean tide, HRET internal tide, pole tide, and dynamic atmospheric correction. Do not subtract those corrections again.

`ssha_karin_2` uses model-based wet-troposphere and associated sea-state-bias corrections. `ssha_karin` uses radiometer-based corrections. Both are valid analysis choices; the model-based version has more available, unflagged samples in these three files, making it a useful starting point here. The `_2` suffix is not a processing-level or resolution designation. Use the quality flag matching the selected anomaly field.

`mean_dynamic_topography` is the supplied CNES_CLS22-CMEMS2020 MDT. Use this together with SSHA for oceanographic ADT. `ssh_karin` and `ssh_karin_2` are heights relative to an ellipsoid and are not ADT. Subtracting the supplied EGM2008 geoid is a different construction and does not reproduce the supplied-MDT-based ADT. This calculation is derived from L2 and does not reproduce all of AVISO's L3 processing or reference adjustments.

## Useful variables

| Variable | Use |
| --- | --- |
| `ssha_karin_2` | Starting SSH anomaly, model-based wet-troposphere correction |
| `ssha_karin` | Alternative anomaly, radiometer-based correction |
| `height_cor_xover` | Additive crossover calibration correction |
| `mean_dynamic_topography` | Add to calibrated anomaly to obtain ADT |
| `ssha_karin_2_qual` / `ssha_karin_qual` | Bitmask quality flags for the respective anomaly |
| `height_cor_xover_qual` | Calibration quality: 0 good, 1 suspect, 2 bad |
| `latitude`, `longitude`, `time` | Swath coordinates and observation times |
| `ancillary_surface_classification_flag` | Surface class: 0 is open ocean |
| `dynamic_ice_flag`, `rain_flag` | Ice/rain screening: 0 indicates no ice/rain |
| `distance_to_coast`, `cross_track_distance` | Diagnose coastal and swath-edge effects |
| `ssh_karin_uncert` | Measurement uncertainty; not total ADT uncertainty |
| `mean_dynamic_topography_uncert` | MDT accuracy information |
| `dac`, tide fields, mean sea surface fields | Inspect or deliberately replace corrections/reference surfaces |

## Conservative first-look extraction

This uses the locally installed `netCDF4` library, which automatically decodes packing scale factors and fill values. Do not multiply the decoded heights by `scale_factor` again.

```python
from netCDF4 import Dataset
import numpy as np

with Dataset("path/to/SWOT_L2_LR_SSH_Expert_file.nc") as ds:
    ssha = ds["ssha_karin_2"][:] + ds["height_cor_xover"][:]
    adt = ssha + ds["mean_dynamic_topography"][:]
    good = np.ma.filled(
        (ds["ssha_karin_2_qual"][:] == 0)
        & (ds["height_cor_xover_qual"][:] == 0)
        & (ds["ancillary_surface_classification_flag"][:] == 0)
        & (ds["dynamic_ice_flag"][:] == 0)
        & (ds["rain_flag"][:] == 0),
        False,
    )
    ssha = np.ma.array(ssha, mask=np.ma.getmaskarray(ssha) | ~good)
    adt = np.ma.array(adt, mask=np.ma.getmaskarray(adt) | ~good)
    latitude = ds["latitude"][:]
    longitude = ds["longitude"][:]
    time = ds["time"][:]
    time_units = ds["time"].units
```

Requiring a zero anomaly flag rejects all suspect, degraded, and bad flags. It is a conservative starting policy rather than a universal scientific requirement. The anomaly flags are bitmasks, not ordinal quality scores. Land/ice/rain masks may need refinement for coastal or specialized studies. Missing MDT means ADT is unavailable at that pixel even when SSHA remains usable. These checks do not eliminate every possible outlier or residual instrumental/geophysical error.

## Evidence

- Variable comments, flag definitions, and model sources were read directly from each downloaded NetCDF file.
- NASA's [quality-flag tutorial](https://podaac.github.io/tutorials/notebooks/datasets/SWOT_quality_flag_demo.html) demonstrates adding `height_cor_xover` to `ssha_karin_2` and using surface/quality masks. Its example uses Version C; the Version D sample metadata was used for the definitions above.
- [NASA Expert Version D product page](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_LR_SSH_Expert_D).
