#!/usr/bin/env python3
"""Sanity checks of the per-cycle SWOT IODA files written by make_swot_ioda.py.

@begin counts obs exactly at the window begin, which JEDI will not assimilate.

python check_swot_ioda.py [--out-root swot_ioda] [--error-ratio 1.0]
"""
import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
BASE_ERROR = 0.1


def seconds(t):
    return (t - EPOCH).total_seconds()


def check_file(path, cycle, error_ratio):
    with Dataset(path) as ds:
        md = ds['MetaData']
        t = md['dateTime'][:].astype(np.int64)
        t0 = md['originalDateTime'][:].astype(np.int64)
        err = ds['ObsError/absoluteDynamicTopography'][:]
        passes = np.unique(md['pass'][:])
        sources = list(ds.getncattr('obs_source_files'))
    wb, we = seconds(cycle - timedelta(hours=3)), seconds(cycle + timedelta(hours=3))
    c = seconds(cycle)
    dt_edge = np.where(t0 < wb, wb - t0, np.where(t0 > we, t0 - we, 0))
    expected = BASE_ERROR + error_ratio * dt_edge / 86400.0
    ids = [re.search(r'Expert_(\d{3})_(\d{3})_', s).groups() for s in sources]
    return dict(
        nlocs=t.size,
        # JEDI keeps obs with begin < dateTime <= end
        in_window=bool(((t >= wb) & (t <= we)).all()),
        at_begin=int((t == wb).sum()),
        span_h=((t0.min() - c) / 3600, (t0.max() - c) / 3600),
        err=(float(err.min()), float(err.max())),
        err_dev=float(np.abs(err - expected).max()),
        npass=passes.size,
        ngran=len(sources),
        dupes=len(ids) - len(set(ids)),
        size_mb=path.stat().st_size / 1e6,
    )


def consistency(path_a, path_b):
    """Obs shared by two cycles must only differ by dateTime and ObsError."""
    def load(p):
        with Dataset(p) as ds:
            md = ds['MetaData']
            keys = np.stack([md['originalDateTime'][:], md['latitude'][:].round(5), md['longitude'][:].round(5)], 1)
            return {tuple(k): v for k, v in zip(keys.tolist(), ds['ObsValue/absoluteDynamicTopography'][:].tolist())}
    a, b = load(path_a), load(path_b)
    shared = a.keys() & b.keys()
    return len(shared), max((abs(a[k] - b[k]) for k in shared), default=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', type=Path, default=ROOT / 'swot_ioda')
    p.add_argument('--error-ratio', type=float, default=1.0)
    args = p.parse_args()

    cycles = [datetime.strptime(s, '%Y%m%d%H').replace(tzinfo=timezone.utc)
              for s in (args.out_root / 'cycles.txt').read_text().split()]
    path = lambda c: args.out_root / f'gdas.{c:%Y%m%d}' / f'{c:%H}' / 'ocean' / 'adt' / f'gdas.t{c:%H}z.nasa_swk.nc'

    print(f"{'cycle':10s} {'nlocs':>8s} {'inWin':>5s} {'orig span (h)':>15s} {'ObsError':>13s} "
          f"{'errDev':>7s} {'gran':>4s} {'pass':>4s} {'dup':>3s} {'MB':>5s} {'@begin':>6s}")
    ok = True
    for c in cycles:
        f = path(c)
        if not f.exists():
            print(f'{c:%Y%m%d%H} MISSING')
            ok = False
            continue
        r = check_file(f, c, args.error_ratio)
        good = r['in_window'] and r['err_dev'] < 1e-3 and r['dupes'] == 0
        ok &= good
        print(f"{c:%Y%m%d%H} {r['nlocs']:8d} {str(r['in_window']):>5s} {r['span_h'][0]:7.2f}/{r['span_h'][1]:6.2f} "
              f"{r['err'][0]:6.3f}-{r['err'][1]:5.3f} {r['err_dev']:7.1e} {r['ngran']:4d} {r['npass']:4d} "
              f"{r['dupes']:3d} {r['size_mb']:5.1f} {r['at_begin']:6d}{'' if good else '  <-- FAILED'}")

    pairs = [(a, b) for a, b in zip(cycles, cycles[1:]) if path(a).exists() and path(b).exists()][:2]
    for a, b in pairs:
        n, dmax = consistency(path(a), path(b))
        print(f'shared obs {a:%Y%m%d%H} vs {b:%Y%m%d%H}: {n} with max |dObsValue| = {dmax:.2e}')
        ok &= n > 0 and dmax == 0.0
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
