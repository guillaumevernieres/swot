"""Plot the downloaded SWOT L2 Expert samples; run with python3 plot_samples.py."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'samples' / 'plots'
OUT.mkdir(exist_ok=True)
PC = ccrs.PlateCarree()
summary = []
combined = []

def valid(a):
    return ~np.ma.getmaskarray(a) & np.isfinite(np.ma.getdata(a))

def masked(a, good):
    return np.ma.array(a, mask=~good | ~valid(a))

def setup(ax):
    ax.set_global()
    ax.set_facecolor('#eef1f4')
    ax.coastlines(resolution='110m', linewidth=.45, color='#555555')
    gl = ax.gridlines(linewidth=.3, color='gray', alpha=.5, draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

def points(ax, lon, lat, z, **kwargs):
    # Thin rendering only; calculations and reported counts use every pixel.
    lo, la, zz = lon[::2, ::2], lat[::2, ::2], z[::2, ::2]
    keep = valid(lo) & valid(la) & valid(zz)
    return ax.scatter(lo[keep], la[keep], c=zz[keep], s=1.2,
                      linewidths=0, rasterized=True, transform=PC, **kwargs)

def categories(flag):
    # Product flags are bitmasks: bits 0–14 suspect, 15–23 degraded,
    # 24–31 bad, except bit 30 is the explicit degraded summary bit.
    raw = np.ma.filled(flag, 0).astype('uint32')
    result = np.zeros(raw.shape, dtype='uint8')
    result[raw != 0] = 1
    result[(raw & np.uint32(0x00FF8000 | (1 << 30))) != 0] = 2
    result[(raw & np.uint32(0xBF000000)) != 0] = 3
    return np.ma.array(result, mask=np.ma.getmaskarray(flag))

for path in sorted((ROOT / 'samples').glob('*.nc')):
    with Dataset(path) as ds:
        names = ['latitude', 'longitude', 'ssha_karin', 'ssha_karin_2',
                 'height_cor_xover', 'mean_dynamic_topography',
                 'ssha_karin_qual', 'ssha_karin_2_qual', 'height_cor_xover_qual',
                 'ancillary_surface_classification_flag', 'dynamic_ice_flag', 'rain_flag']
        d = {n: ds[n][:] for n in names}
        available_times = ds['time'][:].compressed()
        times = num2date([available_times.min(), available_times.max()], ds['time'].units)
    lon, lat = d['longitude'], d['latitude']
    ocean = np.ma.filled((d['ancillary_surface_classification_flag'] == 0)
                        & (d['dynamic_ice_flag'] == 0) & (d['rain_flag'] == 0), False)
    calibration = np.ma.filled(d['height_cor_xover_qual'] == 0, False)
    good2 = ocean & calibration & np.ma.filled(d['ssha_karin_2_qual'] == 0, False)
    good1 = ocean & calibration & np.ma.filled(d['ssha_karin_qual'] == 0, False)
    sla2 = masked(d['ssha_karin_2'] + d['height_cor_xover'], good2)
    sla1 = masked(d['ssha_karin'] + d['height_cor_xover'], good1)
    adt = masked(sla2 + d['mean_dynamic_topography'], good2)
    mdt = masked(d['mean_dynamic_topography'], good2 & valid(adt))
    correction = masked(d['height_cor_xover'], good2)
    difference = sla2 - sla1
    pass_number = path.name.split('_')[6]
    # Center each map on its swath, while retaining a consistent global extent.
    center = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(lon.compressed()))))))
    projection = ccrs.PlateCarree(central_longitude=center)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), subplot_kw={'projection': projection})
    fields = [
        ('Absolute dynamic topography', adt, 'RdBu_r', -1.5, 1.5),
        ('Calibrated SSHA — model wet correction', sla2, 'RdBu_r', -.5, .5),
        ('Calibrated SSHA — radiometer wet correction', sla1, 'RdBu_r', -.5, .5),
        ('Mean dynamic topography', mdt, 'RdBu_r', -1.5, 1.5),
        ('Crossover correction (added to SSHA)', correction, 'RdBu_r', -3, 3),
        ('SSHA difference: model − radiometer', difference, 'RdBu_r', -.1, .1),
    ]
    for ax, (title, z, cmap, vmin, vmax) in zip(axes.flat, fields):
        setup(ax)
        sc = points(ax, lon, lat, z, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(sc, ax=ax, orientation='horizontal', pad=.07, shrink=.88,
                     fraction=.055, extend='both', label='meters')
        ax.set_title(title, fontsize=11)
    colors = ['#218c49', '#e7b416', '#e87524', '#bb2441']
    cmap = ListedColormap(colors)
    for ax, name, title in zip(list(axes.flat)[6:8],
                              ['ssha_karin_2_qual', 'ssha_karin_qual'],
                              ['Model SSHA quality', 'Radiometer SSHA quality']):
        setup(ax)
        sc = points(ax, lon, lat, categories(d[name]), cmap=cmap,
                    norm=BoundaryNorm(np.arange(-.5, 4.5), 4))
        cb = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=.07,
                         shrink=.88, fraction=.055, ticks=range(4))
        cb.ax.set_xticklabels(['Good', 'Suspect', 'Degraded', 'Bad'])
        ax.set_title(title + ' — all available pixels', fontsize=11)
    ax = axes[2, 2]
    setup(ax)
    sc = points(ax, lon, lat, d['height_cor_xover_qual'],
                cmap=ListedColormap([colors[0], colors[1], colors[3]]),
                norm=BoundaryNorm(np.arange(-.5, 3.5), 3))
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=.07,
                     shrink=.88, fraction=.055, ticks=range(3))
    cb.ax.set_xticklabels(['Good', 'Suspect', 'Bad'])
    ax.set_title('Crossover quality — all available pixels', fontsize=11)
    fig.suptitle(f'SWOT Level-2 Expert · cycle 056 · pass {pass_number}\n'
                 f'{times[0]:%Y-%m-%d %H:%M}–{times[1]:%H:%M} UTC', fontsize=17, y=.98)
    fig.text(.5, .025, 'Height panels: good SSHA + good crossover; open ocean, no ice/rain; missing values excluded.\n'
             'ADT = SSHA₂ + crossover + MDT. Quality panels are unfiltered. '
             'Shared scales across passes; saturated colors exceed the shown range.\n'
             'Maps show longitude/latitude; rendering uses every second row and column (no averaging).',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=.89, bottom=.11, hspace=.42, wspace=.16)
    stem = OUT / f'swot_pass_{pass_number}_variables'
    fig.savefig(stem.with_suffix('.png'), dpi=170)
    fig.savefig(stem.with_suffix('.pdf'), dpi=170)
    plt.close(fig)
    summary.append({'file': path.name, 'pass': pass_number,
                    'good_adt_pixels': int(adt.count()),
                    'good_ssha_model_pixels': int(sla2.count()),
                    'good_ssha_radiometer_pixels': int(sla1.count()),
                    'adt_percentiles_1_50_99_m': np.percentile(adt.compressed(), [1, 50, 99]).tolist()})
    combined.append((lon, lat, adt, pass_number))
    print('Created', stem.name, flush=True)

fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': ccrs.Robinson()})
setup(ax)
for lon, lat, adt, number in combined:
    sc = points(ax, lon, lat, adt, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
fig.colorbar(sc, ax=ax, orientation='horizontal', shrink=.65, pad=.05,
             fraction=.045, extend='both', label='Absolute dynamic topography (m)')
ax.set_title('SWOT absolute dynamic topography · September 18, 2026\n'
             'Level-2 Expert · cycle 056 · passes 227–229', fontsize=16, pad=18)
fig.subplots_adjust(bottom=.20, top=.88)
fig.text(.5, .035, 'ADT = ssha_karin_2 + height_cor_xover + mean_dynamic_topography\n'
         'Good-quality, ice-free, rain-free ocean pixels. Gaps reflect swath coverage and screening.',
         ha='center', fontsize=10)
fig.savefig(OUT / 'swot_adt_overview.png', dpi=180, bbox_inches='tight')
fig.savefig(OUT / 'swot_adt_overview.pdf', dpi=180, bbox_inches='tight')
plt.close(fig)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Created overview and summary', flush=True)
