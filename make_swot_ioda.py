#!/usr/bin/env python3
"""Make one SWOT KaRIn ADT IODA file per GDAS cycle from +/-N hours of SWOT L2 granules.

For each cycle, the granules overlapping [cycle - N h, cycle + N h] are given to the obsForge
Swot2Ioda converter (provider SWOT) with the 6 h DA window as its window, so that observations
outside of the DA window are re-dated to the window edges and their ObsError inflated by
"error ratio" (m/day), as done for RADS in obsForge's marine_prepobs task.

The converter runs on compute nodes through a Slurm job array (one element per cycle):

    python3 make_swot_ioda.py --start 2025090200 --end 2025090518 --dry-run
    python3 make_swot_ioda.py --start 2025090200 --end 2025090518 --submit

Output: <out-root>/gdas.YYYYMMDD/HH/ocean/adt/gdas.tHHz.nasa_swk.nc
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONVERTER = ROOT / "ioda_check" / "bin" / "obsforge_obsprovider2ioda.x"
GDAS_MODULEFILES = Path("/scratch3/NCEPDEV/da/Guillaume.Vernieres/sandboxes/gw-ee2-build-test-2/sorc/gdas.cd/modulefiles")
OBS_SPACE = "nasa_swk"
DEFAULT_SETUP = (f"export MACHINE_ID=ursa\n"
                 f"source {ROOT}/obsForge/ush/module-setup.sh\n"
                 f"module use {GDAS_MODULEFILES}\n"
                 f"module load GDAS/ursa.intel")
DA_HALF_WINDOW = timedelta(hours=3)
GAP_THRESHOLD = timedelta(minutes=10)
GRANULE_RE = re.compile(
    r"SWOT_L2_LR_SSH_Expert_(?P<cycle>\d{3})_(?P<pass>\d{3})_(?P<start>\d{8}T\d{6})_(?P<end>\d{8}T\d{6})"
    r"_(?P<crid>[A-Z0-9]+)_(?P<counter>\d{2})\.nc$"
)


@dataclass(frozen=True)
class Granule:
    path: Path
    cycle: int
    pass_number: int
    start: datetime
    end: datetime
    counter: int


@dataclass(frozen=True)
class DaCycle:
    center: datetime
    half_search: timedelta

    @property
    def window_begin(self) -> datetime:
        return self.center - DA_HALF_WINDOW

    @property
    def window_end(self) -> datetime:
        return self.center + DA_HALF_WINDOW

    @property
    def search_begin(self) -> datetime:
        return self.center - self.half_search

    @property
    def search_end(self) -> datetime:
        return self.center + self.half_search

    @property
    def tag(self) -> str:
        return self.center.strftime("%Y%m%d%H")


def utc(value: str, fmt: str) -> datetime:
    return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)


def iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def discover_granules(data_dir: Path) -> tuple[list[Granule], int]:
    """Completed downloads only (with a receipt), keeping the latest product counter of each pass."""
    latest: dict[tuple[int, int], Granule] = {}
    n_files = 0
    for path in sorted(data_dir.glob("SWOT_L2_LR_SSH_Expert_*.nc")):
        match = GRANULE_RE.match(path.name)
        if not match or not path.with_name(path.name + ".download.json").exists():
            continue
        n_files += 1
        granule = Granule(path=path.resolve(),
                          cycle=int(match["cycle"]),
                          pass_number=int(match["pass"]),
                          start=utc(match["start"], "%Y%m%dT%H%M%S"),
                          end=utc(match["end"], "%Y%m%dT%H%M%S"),
                          counter=int(match["counter"]))
        key = (granule.cycle, granule.pass_number)
        if key not in latest or granule.counter > latest[key].counter:
            latest[key] = granule
    return sorted(latest.values(), key=lambda g: g.start), n_files - len(latest)


def da_cycles(start: datetime, end: datetime, half_search: timedelta) -> list[DaCycle]:
    cycles = []
    center = start
    while center <= end:
        cycles.append(DaCycle(center, half_search))
        center += timedelta(hours=6)
    return cycles


def data_gaps(granules: list[Granule], begin: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    granules = [g for g in granules if g.end > g.start]  # zero-length granules hold no data
    gaps = []
    for a, b in zip(granules, granules[1:]):
        if b.start - a.end > GAP_THRESHOLD and b.start > begin and a.end < end:
            gaps.append((a.end, b.start))
    return gaps


def cycle_output(out_root: Path, cycle: DaCycle) -> Path:
    return (out_root / f"gdas.{cycle.center:%Y%m%d}" / f"{cycle.center:%H}" / "ocean" / "adt"
            / f"gdas.t{cycle.center:%H}z.{OBS_SPACE}.nc")


def write_converter_yaml(path: Path, cycle: DaCycle, granules: list[Granule], output: Path,
                         args: argparse.Namespace) -> None:
    lines = [
        "provider: SWOT",
        f"window begin: {iso(cycle.window_begin)}",
        f"window end: {iso(cycle.window_end)}",
        f"error ratio: {args.error_ratio}  # meters per day",
        f"ssha variable: {args.ssha_variable}",
        "binning:",
        f"  stride: {args.stride}",
        f"  min number of obs: {args.min_obs}",
        f"output file: {output}",
        "input files:",
        *[f"- {g.path}" for g in granules],
    ]
    path.write_text("\n".join(lines) + "\n")


def write_sbatch(path: Path, out_root: Path, n_cycles: int, args: argparse.Namespace) -> None:
    path.write_text(f"""#!/bin/bash
#SBATCH --job-name=swot2ioda
#SBATCH --account={args.account}
#SBATCH --qos={args.qos}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks={args.ntasks}
#SBATCH --time={args.walltime}
#SBATCH --array=1-{n_cycles}
#SBATCH --output={out_root}/logs/swot2ioda.%A_%a.log
set -eu
{args.setup}
CYCLE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {out_root}/cycles.txt)
YAML={out_root}/yaml/gdas.${{CYCLE}}.{OBS_SPACE}.yaml
echo "cycle ${{CYCLE}}: ${{YAML}}"
mkdir -p $(sed -n 's/^output file: //p' ${{YAML}} | xargs dirname)
cd {out_root}/logs
srun -n {args.ntasks} {args.converter} ${{YAML}}
""")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, type=lambda s: utc(s, "%Y%m%d%H"), help="First cycle, yyyymmddhh.")
    p.add_argument("--end", required=True, type=lambda s: utc(s, "%Y%m%d%H"), help="Last cycle, yyyymmddhh.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data", help="SWOT L2 Expert granules.")
    p.add_argument("--out-root", type=Path, default=ROOT / "swot_ioda", help="Output COMOUT-like root.")
    p.add_argument("--half-window-hours", type=float, default=24.0, help="Search window half length.")
    p.add_argument("--max-gap-hours", type=float, default=12.0,
                   help="Skip cycles with a longer data gap in the search window.")
    p.add_argument("--error-ratio", type=float, default=1.0, help="ObsError inflation, m/day.")
    p.add_argument("--ssha-variable", default="ssha_karin_2", choices=("ssha_karin_2", "ssha_karin"))
    p.add_argument("--stride", type=int, default=5, help="Superobing stride (swath pixels).")
    p.add_argument("--min-obs", type=int, default=5, help="Minimum number of pixels per superob.")
    p.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    p.add_argument("--setup", default=DEFAULT_SETUP,
                   help="Shell lines run in the batch job before the converter (environment setup).")
    p.add_argument("--account", default="marine-cpu")
    p.add_argument("--qos", default="batch")
    p.add_argument("--partition", default="u1-compute")
    p.add_argument("--ntasks", type=int, default=14)
    p.add_argument("--walltime", default="00:20:00")
    p.add_argument("--submit", action="store_true", help="Submit the Slurm array job.")
    p.add_argument("--dry-run", action="store_true", help="Only report what would be done.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_root = args.out_root.resolve()
    args.converter = args.converter.resolve()
    if not args.converter.exists():
        raise SystemExit(f"Converter not found: {args.converter}")

    granules, n_dupes = discover_granules(args.data_dir)
    if not granules:
        raise SystemExit(f"No completed SWOT granules in {args.data_dir}")
    first, last = granules[0].start, granules[-1].end
    print(f"{len(granules)} granules from {iso(first)} to {iso(last)} "
          f"({n_dupes} older product versions of the same passes ignored)")

    cycles = da_cycles(args.start, args.end, timedelta(hours=args.half_window_hours))
    max_gap = timedelta(hours=args.max_gap_hours)
    selected = []
    print(f"{'cycle':10s}  {'granules':>8s}  data gaps in the search window")
    for cycle in cycles:
        if cycle.search_begin < first or cycle.search_end > last:
            print(f"{cycle.tag}  skipped: search window {iso(cycle.search_begin)} to {iso(cycle.search_end)} "
                  "is not covered by the local data")
            continue
        inputs = [g for g in granules if g.start < cycle.search_end and g.end > cycle.search_begin]
        gaps = data_gaps(granules, cycle.search_begin, cycle.search_end)
        gap_str = ", ".join(f"{a:%m-%d %H:%M}->{b:%m-%d %H:%M}" for a, b in gaps) or "-"
        if not inputs or any(b - a > max_gap for a, b in gaps):
            print(f"{cycle.tag}  skipped: {len(inputs)} granules, gaps longer than {args.max_gap_hours:g} h: {gap_str}")
            continue
        print(f"{cycle.tag}  {len(inputs):8d}  {gap_str}")
        selected.append((cycle, inputs))

    if not selected:
        raise SystemExit("No cycle to process.")
    if args.dry_run:
        print(f"Dry run: {len(selected)} cycle(s) would be processed into {out_root}")
        return 0

    (out_root / "yaml").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    for cycle, inputs in selected:
        write_converter_yaml(out_root / "yaml" / f"gdas.{cycle.tag}.{OBS_SPACE}.yaml", cycle, inputs,
                             cycle_output(out_root, cycle), args)
    (out_root / "cycles.txt").write_text("".join(f"{cycle.tag}\n" for cycle, _ in selected))
    sbatch = out_root / "run_swot2ioda.sbatch"
    write_sbatch(sbatch, out_root, len(selected), args)
    print(f"Wrote {len(selected)} converter yaml(s) and {sbatch}")

    if args.submit:
        result = subprocess.run(["sbatch", str(sbatch)], capture_output=True, text=True, check=True)
        print(result.stdout.strip())
    else:
        print(f"Submit with: sbatch {sbatch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
