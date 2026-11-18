"""Visual check of the SWOT -> IODA converter: SWOT L2 input vs. IODA output.

python plot_ioda_check.py <converter yaml> [--zoom lon0 lon1 lat0 lat1]
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import yaml
from netCDF4 import Dataset

PC = ccrs.PlateCarree()
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
ADT_RANGE = dict(vmin=-1.5, vmax=1.5, cmap='RdYlBu_r')


def read_input(fname, ssha_name):
    """ADT on the swath, without QC and with the converter's QC."""
    with Dataset(fname) as ds:
        get = lambda v: np.ma.asarray(ds[v][:])
        adt = get(ssha_name) + get('height_cor_xover') + get('mean_dynamic_topography')
        qc = ((get(ssha_name + '_qual').filled(1) == 0)
              & (get('height_cor_xover_qual').filled(1) == 0)
              & (get('ancillary_surface_classification_flag').filled(1) == 0)
              & (get('dynamic_ice_flag').filled(1) == 0)
              & (get('rain_flag').filled(1) == 0)
              & ~np.ma.getmaskarray(get('cross_track_distance')))
        lon = get('longitude')
        lon = np.ma.where(lon >= 180, lon - 360, lon)
        lat = get('latitude')
        time = np.repeat(get('time').filled(np.nan)[:, None], adt.shape[1], axis=1)
    ok = ~np.ma.getmaskarray(adt) & ~np.ma.getmaskarray(lon) & ~np.ma.getmaskarray(lat)
    adt_f = adt.filled(np.nan)
    good = ok & qc & (adt_f > -4) & (adt_f < 4)
    return dict(lon=lon.filled(np.nan), lat=lat.filled(np.nan), adt=adt_f, time=time, ok=ok, good=good)


def read_ioda(fname):
    with Dataset(fname) as ds:
        md = ds['MetaData']
        out = {k: md[k][:] for k in ('longitude', 'latitude', 'dateTime', 'originalDateTime', 'pass')}
        out['adt'] = ds['ObsValue/absoluteDynamicTopography'][:]
        out['err'] = ds['ObsError/absoluteDynamicTopography'][:]
    return out


def setup(ax, extent=None):
    if extent:
        ax.set_extent(extent, crs=PC)
    else:
        ax.set_global()
    ax.set_facecolor('#eef1f4')
    ax.coastlines(resolution='50m' if extent else '110m', linewidth=.45, color='#555555')
    gl = ax.gridlines(linewidth=.3, color='gray', alpha=.5, draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 7}


def scatter(ax, lon, lat, z, s, **kw):
    return ax.scatter(lon, lat, c=z, s=s, linewidths=0, rasterized=True, transform=PC, **kw)


def in_box(lon, lat, box):
    return (lon >= box[0]) & (lon <= box[1]) & (lat >= box[2]) & (lat <= box[3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--zoom', nargs=4, type=float, metavar=('LON0', 'LON1', 'LAT0', 'LAT1'))
    p.add_argument('--outdir', type=Path, help='Where to write the figures (default: next to the yaml).')
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = args.outdir or Path(args.config).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(cfg['output file']).name.replace('.nc', '').replace('.ioda', '')
    if tag.startswith('gdas.t'):  # .../gdas.YYYYMMDD/HH/ocean/<type>/gdas.tHHz.<obs space>.nc
        out = Path(cfg['output file'])
        tag = f"{out.parents[3].name}{out.parents[2].name}.{tag.split('.', 2)[2]}"
    ssha_name = cfg.get('ssha variable', 'ssha_karin_2')
    wb = datetime.fromisoformat(str(cfg['window begin']).replace('Z', '+00:00'))
    we = datetime.fromisoformat(str(cfg['window end']).replace('Z', '+00:00'))

    inputs = [read_input(f, ssha_name) for f in cfg['input files']]
    cat = lambda k, m: np.concatenate([d[k][d[m]] for d in inputs])
    raw = {k: cat(k, 'ok') for k in ('lon', 'lat', 'adt', 'time')}
    qcd = {k: cat(k, 'good') for k in ('lon', 'lat', 'adt', 'time')}
    io = read_ioda(cfg['output file'])

    n_pix = sum(d['ok'].size for d in inputs)
    print(f'input pixels: {n_pix}, non-missing: {raw["adt"].size}, after QC: {qcd["adt"].size}, '
          f'IODA obs: {io["adt"].size}')
    for name, z in (('input (QC)', qcd['adt']), ('IODA', io['adt'])):
        print(f'{name:11s} ADT min/mean/max: {np.nanmin(z):.3f} {np.nanmean(z):.3f} {np.nanmax(z):.3f}')
    print(f'IODA obs error min/max: {io["err"].min():.4f} {io["err"].max():.4f}')

    thin = slice(None, None, 3)  # render-only thinning of the full-resolution swath

    # Global maps
    fig, axs = plt.subplots(2, 2, figsize=(16, 9.5), subplot_kw=dict(projection=PC), constrained_layout=True)
    panels = [
        (axs[0, 0], raw['lon'][thin], raw['lat'][thin], raw['adt'][thin], .3, ADT_RANGE,
         f'Input ADT, no QC ({raw["adt"].size:,} pixels)', 'ADT (m)'),
        (axs[0, 1], qcd['lon'][thin], qcd['lat'][thin], qcd['adt'][thin], .3, ADT_RANGE,
         f'Input ADT, converter QC ({qcd["adt"].size:,} pixels)', 'ADT (m)'),
        (axs[1, 0], io['longitude'], io['latitude'], io['adt'], .6, ADT_RANGE,
         f'IODA ObsValue absoluteDynamicTopography ({io["adt"].size:,} superobs)', 'ADT (m)'),
        (axs[1, 1], io['longitude'], io['latitude'], io['err'], .6,
         dict(vmin=0.1, vmax=max(0.1001, float(io['err'].max())), cmap='viridis'),
         'IODA ObsError (inflated outside the window)', 'ObsError (m)'),
    ]
    for ax, lo, la, z, s, kw, title, label in panels:
        setup(ax)
        sc = scatter(ax, lo, la, z, s, **kw)
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=.8, pad=.01, label=label)
        if args.zoom:
            x0, x1, y0, y1 = args.zoom
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k-', lw=1, transform=PC)
    fig.suptitle(f'SWOT KaRIn -> IODA, window {wb:%Y-%m-%d %H:%M} to {we:%Y-%m-%d %H:%M} UTC, '
                 f'{len(inputs)} passes', fontsize=12)
    fig.savefig(outdir / f'{tag}_global.png', dpi=130)
    plt.close(fig)

    # Zoom: full-resolution input vs superobs
    if args.zoom:
        box = args.zoom
        extent = [box[0], box[1], box[2], box[3]]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6.2), subplot_kw=dict(projection=PC), constrained_layout=True)
        m_raw = in_box(raw['lon'], raw['lat'], box)
        m_qc = in_box(qcd['lon'], qcd['lat'], box)
        m_io = in_box(io['longitude'], io['latitude'], box)
        sel = m_qc & np.isfinite(qcd['adt'])
        zr = dict(ADT_RANGE)
        if sel.any():
            zr.update(vmin=np.percentile(qcd['adt'][sel], 1), vmax=np.percentile(qcd['adt'][sel], 99))
        for ax, lo, la, z, s, title in (
                (axs[0], raw['lon'][m_raw], raw['lat'][m_raw], raw['adt'][m_raw], 2,
                 f'Input, no QC ({m_raw.sum():,})'),
                (axs[1], qcd['lon'][m_qc], qcd['lat'][m_qc], qcd['adt'][m_qc], 2,
                 f'Input, converter QC ({m_qc.sum():,})'),
                (axs[2], io['longitude'][m_io], io['latitude'][m_io], io['adt'][m_io], 14,
                 f'IODA superobs ({m_io.sum():,})')):
            setup(ax, extent)
            sc = scatter(ax, lo, la, z, s, **zr)
            ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=axs, shrink=.8, pad=.01, label='ADT (m)')
        fig.savefig(outdir / f'{tag}_zoom.png', dpi=130)
        plt.close(fig)

    # Distributions: ADT and time
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    bins = np.linspace(-2, 2, 161)
    axs[0].hist(qcd['adt'], bins=bins, density=True, histtype='step', lw=1.5, label='input (QC, full res)')
    axs[0].hist(io['adt'], bins=bins, density=True, histtype='step', lw=1.5, label='IODA superobs')
    axs[0].set_xlabel('ADT (m)')
    axs[0].set_ylabel('density')
    axs[0].legend()
    axs[0].set_title('ADT distribution')

    to_dt = lambda s: np.array([EPOCH + timedelta(seconds=float(x)) for x in s], dtype='datetime64[s]')
    t_orig = to_dt(io['originalDateTime'])
    t0 = min(t_orig.min(), np.datetime64(wb.replace(tzinfo=None))) - np.timedelta64(1, 'h')
    t1 = max(t_orig.max(), np.datetime64(we.replace(tzinfo=None))) + np.timedelta64(1, 'h')
    step = np.timedelta64(10, 'm') if t1 - t0 <= np.timedelta64(12, 'h') else np.timedelta64(30, 'm')
    tb = np.arange(t0, t1, step)
    axs[1].hist(to_dt(io['originalDateTime']), bins=tb, histtype='step', lw=1.5, label='originalDateTime')
    axs[1].hist(to_dt(io['dateTime']), bins=tb, histtype='step', lw=1.5, label='dateTime (redated)')
    for t in (wb, we):
        axs[1].axvline(np.datetime64(t.replace(tzinfo=None)), color='k', ls='--', lw=1)
    axs[1].set_title('IODA obs time vs. DA window')
    axs[1].legend()
    axs[1].tick_params(axis='x', labelrotation=30)

    axs[2].scatter(to_dt(io['originalDateTime']), io['err'], s=1, rasterized=True)
    for t in (wb, we):
        axs[2].axvline(np.datetime64(t.replace(tzinfo=None)), color='k', ls='--', lw=1)
    axs[2].set_ylabel('ObsError (m)')
    axs[2].set_title('ObsError vs. original time')
    axs[2].tick_params(axis='x', labelrotation=30)
    fig.savefig(outdir / f'{tag}_stats.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
