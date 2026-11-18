"""Plot SWOT KaRIn SWH, KaRIn wind speed and nadir-altimeter SWH over a date range.

Usage: python plot_wave_wind.py --start YYYYMMDD [--end YYYYMMDD] [--res 0.25]
                                [--data-dir data] [--out plots/wave_wind]

Files are streamed one at a time, so long ranges do not need much memory. Outputs:
  swot_wave_wind_maps_<tag>.png    range-mean maps on a regular grid
  swot_wave_wind_stats_<tag>.png   daily SWH/wind differences (only when the range spans > 1 day)
  swot_wave_wind_stats_<tag>.csv   the daily statistics behind that plot
  swot_wave_wind_pass<NNN>_<day>.png  along-track detail of the pass with the most good KaRIn SWH
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
PC = ccrs.PlateCarree()
# Bits 15-31 of the *_qual flags are degraded/bad; bits 0-14 are only "suspect",
# and every KaRIn wave/wind pixel carries at least one suspect bit, so keep those.
DEGRADED_OR_BAD = np.uint32(0xFFFF8000)
SWH_RANGE = (0, 8)
WIND_RANGE = (0, 20)
# Nadir is a single along-track line, so it is binned on a coarser grid to stay visible.
NADIR_COARSEN = 4
# Daily comparisons: name -> (label, color). Colors follow the comparison, not its rank.
PAIRS = {'swh_karin_nadir': ('KaRIn − nadir', '#2a78d6'),
         'swh_karin_model': ('KaRIn − model', '#eb6834'),
         'swh_nadir_model': ('Nadir − model', '#1baf7a'),
         'wind_karin_model': ('KaRIn − model', '#eb6834')}


def read(path):
    names = ['latitude', 'longitude', 'cross_track_distance', 'swh_karin', 'swh_karin_qual',
             'wind_speed_karin', 'wind_speed_karin_qual', 'swh_nadir_altimeter', 'swh_model',
             'wind_speed_model_u', 'wind_speed_model_v',
             'ancillary_surface_classification_flag', 'dynamic_ice_flag', 'rain_flag']
    with Dataset(path) as ds:
        d = {n: ds[n][:] for n in names}
    ocean = np.ma.filled((d['ancillary_surface_classification_flag'] == 0)
                         & (d['dynamic_ice_flag'] == 0) & (d['rain_flag'] == 0), False)

    def good(var):
        qual = np.ma.filled(d[var + '_qual'], np.uint32(0xFFFFFFFF)).astype('uint32')
        return np.ma.array(d[var], mask=np.ma.getmaskarray(d[var]) | ~ocean
                           | ((qual & DEGRADED_OR_BAD) != 0))

    def on_ocean(a):
        return np.ma.array(a, mask=np.ma.getmaskarray(a) | ~ocean)

    # swh_nadir_altimeter is the along-track nadir value repeated across the swath.
    nadir_col = int(np.argmin(np.abs(np.ma.filled(d['cross_track_distance'], np.inf)).mean(axis=0)))
    nadir = np.ma.array(d['swh_nadir_altimeter'][:, nadir_col],
                        mask=np.ma.getmaskarray(d['swh_nadir_altimeter'][:, nadir_col]) | ~ocean[:, nadir_col])
    wind_model = np.ma.sqrt(d['wind_speed_model_u'] ** 2 + d['wind_speed_model_v'] ** 2)
    return {'lon': d['longitude'], 'lat': d['latitude'], 'xtrack': d['cross_track_distance'],
            'swh': good('swh_karin'), 'wind': good('wind_speed_karin'),
            'swh_model': on_ocean(d['swh_model']), 'wind_model': on_ocean(wind_model),
            'nadir': nadir, 'nadir_col': nadir_col}


def inner_swath_means(p):
    """Average each line over the inner swath (10-30 km from nadir, both sides)."""
    xt = np.abs(np.ma.filled(p['xtrack'], np.nan))
    inner = (xt >= 10e3) & (xt <= 30e3)
    return {k: np.ma.array(p[k], mask=np.ma.getmaskarray(p[k]) | ~inner).mean(axis=1)
            for k in ('swh', 'swh_model', 'wind', 'wind_model')}


def differences(p):
    m = inner_swath_means(p)
    pairs = {'swh_karin_nadir': m['swh'] - p['nadir'],
             'swh_karin_model': m['swh'] - m['swh_model'],
             'swh_nadir_model': p['nadir'] - m['swh_model'],
             'wind_karin_model': m['wind'] - m['wind_model']}
    return {k: v.compressed() for k, v in pairs.items()}


class Grid:
    """Running sum/count of a field on a regular lon/lat grid."""

    def __init__(self, res):
        self.res = res
        self.nlon, self.nlat = int(round(360 / res)), int(round(180 / res))
        self.sum = np.zeros(self.nlat * self.nlon)
        self.count = np.zeros(self.nlat * self.nlon)

    def add(self, lon, lat, z):
        keep = ~np.ma.getmaskarray(z) & ~np.ma.getmaskarray(lon) & ~np.ma.getmaskarray(lat)
        lo = (np.ma.getdata(lon)[keep] + 180) % 360
        la = np.clip(np.ma.getdata(lat)[keep] + 90, 0, 180 - 1e-9)
        idx = (la // self.res).astype(int) * self.nlon + (lo // self.res).astype(int)
        self.sum += np.bincount(idx, weights=np.ma.getdata(z)[keep], minlength=self.sum.size)
        self.count += np.bincount(idx, minlength=self.count.size)

    def mean(self):
        with np.errstate(invalid='ignore'):
            m = self.sum / self.count
        return np.ma.masked_invalid(m.reshape(self.nlat, self.nlon))

    def edges(self):
        return np.linspace(-180, 180, self.nlon + 1), np.linspace(-90, 90, self.nlat + 1)


def day_stats(diffs):
    """Mean, std, median and robust std (1.4826 MAD) of each difference for one day."""
    row = {}
    for k in PAIRS:
        x = np.concatenate([d[k] for d in diffs]) if diffs else np.empty(0)
        if x.size:
            med = np.median(x)
            row.update({f'{k}_n': x.size, f'{k}_mean': x.mean(), f'{k}_std': x.std(),
                        f'{k}_median': med, f'{k}_rstd': 1.4826 * np.median(np.abs(x - med))})
        else:
            row.update({f'{k}_{s}': np.nan for s in ('mean', 'std', 'median', 'rstd')})
            row[f'{k}_n'] = 0
    return row


def setup(ax):
    ax.set_global()
    ax.set_facecolor('#eef1f4')
    ax.coastlines(resolution='110m', linewidth=.45, color='#555555')
    ax.gridlines(linewidth=.3, color='gray', alpha=.5)


def points(ax, lon, lat, z, step=2, s=.6, **kw):
    lo, la, zz = lon[::step, ::step], lat[::step, ::step], z[::step, ::step]
    keep = ~np.ma.getmaskarray(lo) & ~np.ma.getmaskarray(la) & ~np.ma.getmaskarray(zz)
    return ax.scatter(lo[keep], la[keep], c=zz[keep], s=s, linewidths=0,
                      rasterized=True, transform=PC, **kw)


def maps(grids, label, tag, nfiles, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 17), subplot_kw={'projection': ccrs.Robinson()})
    panels = [('KaRIn significant wave height  (swh_karin)', 'swh', 'viridis', SWH_RANGE, 'm'),
              ('KaRIn wind speed  (wind_speed_karin)', 'wind', 'cividis', WIND_RANGE, 'm/s'),
              ('Nadir altimeter significant wave height  (swh_nadir_altimeter)', 'nadir', 'viridis',
               SWH_RANGE, 'm')]
    for ax, (title, key, cmap, (vmin, vmax), units) in zip(axes, panels):
        setup(ax)
        g = grids[key]
        x, y = g.edges()
        pm = ax.pcolormesh(x, y, g.mean(), cmap=cmap, vmin=vmin, vmax=vmax, transform=PC,
                           rasterized=True)
        fig.colorbar(pm, ax=ax, orientation='horizontal', shrink=.6, pad=.03, fraction=.04,
                     extend='max', label=units)
        ax.set_title(f'{title}\nmean on {g.res:g}° grid · {int(g.count.sum()):,} good ocean values',
                     fontsize=12)
    fig.suptitle(f'SWOT L2 LR Expert · {label} · {nfiles} passes', fontsize=16, y=.995)
    fig.text(.5, .01, 'KaRIn: ocean, no ice/rain, no degraded/bad quality bits (suspect bits kept; all pixels carry them).\n'
             'Nadir: along-track values (constant across swath in the product), taken at the nadir column.',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=.95, bottom=.05, hspace=.28)
    fig.savefig(out / f'swot_wave_wind_maps_{tag}.png', dpi=150)
    plt.close(fig)


def stats_plot(rows, label, tag, out):
    days = [r['date'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = [(axes[0, 0], 'median', ['swh_karin_nadir', 'swh_karin_model', 'swh_nadir_model'],
               'SWH difference, median', 'm'),
              (axes[0, 1], 'rstd', ['swh_karin_nadir', 'swh_karin_model', 'swh_nadir_model'],
               'SWH difference, robust std (1.4826·MAD)', 'm'),
              (axes[1, 0], 'median', ['wind_karin_model'], 'Wind speed KaRIn − model, median', 'm/s'),
              (axes[1, 1], 'rstd', ['wind_karin_model'], 'Wind speed KaRIn − model, robust std', 'm/s')]
    for ax, stat, keys, title, units in panels:
        for k in keys:
            name, color = PAIRS[k]
            ax.plot(days, [r[f'{k}_{stat}'] for r in rows], color=color, lw=2, marker='o', ms=4,
                    label=name)
        if stat == 'median':
            ax.axhline(0, color='#888888', lw=.8)
        ax.set_title(title, fontsize=11, loc='left')
        ax.set_ylabel(units)
        ax.grid(color='#dddddd', lw=.5)
        ax.spines[['top', 'right']].set_visible(False)
        if len(keys) > 1:
            ax.legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle(f'SWOT daily wave/wind differences · {label}\n'
                 'line means over the inner swath (10–30 km from nadir) vs nadir altimeter and model',
                 fontsize=13)
    fig.subplots_adjust(top=.88, bottom=.08, left=.06, right=.98, hspace=.25, wspace=.15)
    fig.savefig(out / f'swot_wave_wind_stats_{tag}.png', dpi=150)
    plt.close(fig)


def along_track(name, p, day, out):
    m = inner_swath_means(p)
    lat = p['lat'][:, p['nadir_col']]

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.4, 1])
    axm = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    setup(axm)
    sc = points(axm, p['lon'], p['lat'], p['swh'], step=1, s=.8, cmap='viridis',
                vmin=SWH_RANGE[0], vmax=SWH_RANGE[1])
    fig.colorbar(sc, ax=axm, orientation='vertical', pad=.02, fraction=.02, extend='max', panchor='C',
                 label='swh_karin (m)')
    axm.set_title(f'Pass {name} track (KaRIn SWH)', fontsize=12)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(lat, m['swh_model'], color='#9a9a9a', lw=1.5, label='Wave model (swh_model)')
    ax1.plot(lat, m['swh'], color='#2a6fdb', lw=1, alpha=.9, label='KaRIn, inner swath 10–30 km')
    ax1.plot(lat, p['nadir'], color='#d9480f', lw=1.5, label='Nadir altimeter')
    ax1.set_ylabel('Significant wave height (m)')
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, loc='upper right')
    ax1.set_title('Significant wave height', fontsize=11, loc='left')
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2.plot(lat, m['wind_model'], color='#9a9a9a', lw=1.5, label='Model (wind_speed_model_u/v)')
    ax2.plot(lat, m['wind'], color='#2a6fdb', lw=1, label='KaRIn, inner swath 10–30 km')
    ax2.set_title('Wind speed', fontsize=11, loc='left')
    ax2.set_ylabel('Wind speed (m/s)')
    ax2.set_xlabel('Latitude (°N)')
    ax2.set_ylim(0, None)
    ax2.legend(frameon=False, loc='upper right')
    for ax in (ax1, ax2):
        ax.grid(color='#dddddd', lw=.5)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'SWOT along-track comparison · {day:%Y-%m-%d} · pass {name}', fontsize=14)
    fig.subplots_adjust(top=.94, bottom=.05, left=.07, right=.97, hspace=.3)
    fig.savefig(out / f'swot_wave_wind_pass{name}_{day:%Y%m%d}.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', '--date', dest='start', default='20260801', help='first day, YYYYMMDD')
    ap.add_argument('--end', help='last day (inclusive), YYYYMMDD; defaults to --start')
    ap.add_argument('--res', type=float, default=0.25, help='map grid spacing in degrees')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    ap.add_argument('--out', type=Path, default=ROOT / 'plots' / 'wave_wind')
    args = ap.parse_args()
    start = datetime.strptime(args.start, '%Y%m%d')
    end = datetime.strptime(args.end or args.start, '%Y%m%d')
    if end < start:
        ap.error('--end is before --start')
    args.out.mkdir(parents=True, exist_ok=True)
    one_day = start == end
    tag = f'{start:%Y%m%d}' if one_day else f'{start:%Y%m%d}-{end:%Y%m%d}'
    label = f'{start:%Y-%m-%d}' if one_day else f'{start:%Y-%m-%d} to {end:%Y-%m-%d}'

    grids = {'swh': Grid(args.res), 'wind': Grid(args.res), 'nadir': Grid(args.res * NADIR_COARSEN)}
    rows, nfiles = [], 0
    best = (-1, None, None, None)  # (good KaRIn SWH count, pass, day, data)
    day = start
    while day <= end:
        files = sorted(args.data_dir.glob(f'SWOT_L2_LR_SSH_Expert_???_???_{day:%Y%m%d}T*.nc'))
        if not files:
            print(f'{day:%Y-%m-%d}: no files, skipping', flush=True)
        diffs = []
        for f in files:
            p = read(f)
            grids['swh'].add(p['lon'], p['lat'], p['swh'])
            grids['wind'].add(p['lon'], p['lat'], p['wind'])
            grids['nadir'].add(p['lon'][:, p['nadir_col']], p['lat'][:, p['nadir_col']], p['nadir'])
            diffs.append(differences(p))
            if p['swh'].count() > best[0]:
                best = (p['swh'].count(), f.name.split('_')[6], day, p)
        if files:
            rows.append({'date': day, 'files': len(files), **day_stats(diffs)})
            nfiles += len(files)
            print(f'{day:%Y-%m-%d}: {len(files)} files', flush=True)
        day += timedelta(days=1)
    if not nfiles:
        raise SystemExit(f'No SWOT files for {label} in {args.data_dir}')

    maps(grids, label, tag, nfiles, args.out)
    with open(args.out / f'swot_wave_wind_stats_{tag}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({**r, 'date': f"{r['date']:%Y-%m-%d}"} for r in rows)
    if len(rows) > 1:
        stats_plot(rows, label, tag, args.out)
    along_track(best[1], best[3], best[2], args.out)
    print('Wrote plots to', args.out)


if __name__ == '__main__':
    main()
