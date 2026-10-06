# Download SWOT Level-2 ocean observations

`fetch_swot_l2.py` uses NASA's public CMR catalog and authenticated PO.DAAC downloads. Requires Python 3.10+ and `curl`; no Python packages need installation. Defaults to the Version D **Expert** KaRIn ocean product used in this workspace.

## Examples

```bash
# Preview the newest three files, without authentication or downloads:
python3 fetch_swot_l2.py --latest 3 --dry-run

# Download them:
python3 fetch_swot_l2.py --latest 3 --output data/swot_l2

# Download observations intersecting a region and date range:
python3 fetch_swot_l2.py --start 2026-09-01 --end 2026-09-18 \
  --bbox=-80,20,-50,45 --output data/north_atlantic

# Limit a search while testing:
python3 fetch_swot_l2.py --start 2026-09-18 --end 2026-09-18 \
  --max-files 3 --dry-run
```

Date-only bounds include the entire UTC day. ISO timestamps are supported; timestamps without a timezone are interpreted as UTC. Bounding box order is west, south, east, north, using -180 to 180 longitude. Split dateline-crossing regions into two searches. Bounding boxes select intersecting **whole files**, not cropped observations; their footprint can include land or missing observations.

Optional `--product basic`, `--product unsmoothed`, or `--product windwave` selects other Version D products. Expert includes MDT for deriving ADT. Unsmoothed files are substantially larger. Date-range searches download all matches unless `--max-files` is given. Results are earliest first for date ranges and newest first with `--latest`.

## Authentication

Your existing Earthdata credentials are already configured on this machine. Elsewhere, register at https://urs.earthdata.nasa.gov and add an entry to `~/.netrc`:

```text
machine urs.earthdata.nasa.gov
  login YOUR_EARTHDATA_USERNAME
  password YOUR_EARTHDATA_PASSWORD
```

Restrict its permissions with `chmod 600 ~/.netrc`. Alternatively specify `--netrc /path/to/credentials`. The script stores login cookies only in a temporary directory, removed on normal exit. If NASA returns 401/403, verify the credentials and any required PO.DAAC application authorization in your Earthdata account.

## Re-running and validation

Each completed download gets a `.nc.download.json` receipt with its byte count and SHA-256. Re-runs verify that receipt and skip unchanged local files. Files without receipts, including the older sample downloads, are downloaded again. Transfers use retries and temporary files and replace the destination only after a successful transfer and NetCDF-4 signature check. Interrupted files are not treated as completed downloads; restarting downloads the file afresh rather than resuming a partial transfer.

The hash checks local integrity on later runs; it is not a comparison to NASA's checksum. The signature check rejects login HTML but does not validate every NetCDF variable. `download_manifest.json` records the latest search and per-file outcomes. Failed downloads cause a nonzero exit code. `--dry-run` writes nothing.

Data availability can lag observations by several days. Re-running an overlapping date range is appropriate for collecting newly arrived granules. Different processing revisions can have different filenames and are preserved separately.

References: [NASA CMR API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html), [PO.DAAC Expert Version D](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_LR_SSH_Expert_D).
