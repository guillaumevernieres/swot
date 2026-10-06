#!/usr/bin/env python3
"""Download SWOT KaRIn Level-2 ocean observations from NASA PO.DAAC.

Requires Python 3.10+ and curl. Uses ~/.netrc for Earthdata authentication.
Run with --help for examples. Geographic filters select entire swath files.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import netrc
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlencode, urlsplit

CMR = 'https://cmr.earthdata.nasa.gov/search/'
ARCHIVE = 'archive.swot.podaac.earthdata.nasa.gov'
MAGIC = b'\x89HDF\r\n\x1a\n'


def timestamp(value, end=False):
    """Date-only end bounds include the entire UTC day."""
    if len(value) == 10:
        value += 'T23:59:59.999999+00:00' if end else 'T00:00:00+00:00'
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def bounding_box(value):
    try:
        west, south, east, north = map(float, value.split(','))
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Use west,south,east,north in degrees, west < east and south < north. '
            'Split dateline-crossing regions into two searches.') from None
    return f'{west},{south},{east},{north}'


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def curl(url, output, auth=None, cookies=None):
    args = ['curl', '--location', '--fail', '--silent', '--show-error',
            '--proto', '=https', '--proto-redir', '=https',
            '--connect-timeout', '30', '--max-time', '1800',
            '--retry', '3', '--retry-delay', '2', '--output', str(output)]
    if auth:
        args += ['--netrc-file', str(auth), '--cookie', str(cookies),
                 '--cookie-jar', str(cookies)]
    result = subprocess.run(args + [url], capture_output=True, text=True)
    if result.returncode:
        # Avoid printing redirect URLs, cookies, or credentials.
        raise RuntimeError(f'curl failed (exit {result.returncode}) contacting '
                           f'{urlsplit(url).hostname}. Check network access and Earthdata login; '
                           'HTTP 401/403 may require authorizing the PO.DAAC application in Earthdata.')


def catalog(endpoint, params):
    url = CMR + endpoint + '.json?' + urlencode(params, doseq=True)
    with tempfile.TemporaryDirectory(prefix='swot-cmr-') as tmp:
        output = Path(tmp) / 'response.json'
        curl(url, output)
        return json.loads(output.read_text())['feed']['entry']


def search(args):
    short_name = f'SWOT_L2_LR_SSH_{args.product.upper()}_D'
    collections = catalog('collections', {'short_name': short_name, 'provider': 'POCLOUD'})
    if len(collections) != 1:
        raise RuntimeError(f'Expected one collection for {short_name}; found {len(collections)}.')
    params = {'collection_concept_id': collections[0]['id'], 'page_size': 2000,
              'sort_key': '-start_date' if args.latest else 'start_date'}
    if args.start:
        params['temporal'] = f'{args.start},{args.end}'
    if args.bbox:
        params['bounding_box'] = args.bbox
    limit = args.latest or args.max_files
    if limit:
        params['page_size'] = min(limit, 2000)
    files, seen = [], set()
    page = 1
    while True:
        entries = catalog('granules', dict(params, page_num=page))
        if not entries:
            break
        for entry in entries:
            for link in entry.get('links', []):
                url = link.get('href', '')
                parsed = urlsplit(url)
                if (parsed.scheme != 'https' or parsed.hostname != ARCHIVE
                        or not parsed.path.endswith('.nc') or link.get('inherited')):
                    continue
                name = Path(parsed.path).name
                if name in seen:
                    continue
                seen.add(name)
                files.append({'filename': name, 'url': url,
                              'time_start': entry.get('time_start'),
                              'time_end': entry.get('time_end'),
                              'estimated_mb': float(entry.get('granule_size', 0))})
                if limit and len(files) >= limit:
                    return short_name, files
        if len(entries) < params['page_size']:
            break
        page += 1
    return short_name, files


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def check_netcdf(path):
    with path.open('rb') as stream:
        if stream.read(8) != MAGIC:
            raise RuntimeError('Response is not NetCDF-4; Earthdata may have returned a login page.')


def save_json(path, content):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(content, stream, indent=2)
        stream.write('\n')
    temporary.replace(path)


def download(item, directory, auth, cookies):
    destination = directory / item['filename']
    receipt = destination.with_suffix('.nc.download.json')
    if destination.exists() and receipt.exists():
        try:
            saved = json.loads(receipt.read_text())
            if (saved['url'] == item['url'] and saved['bytes'] == destination.stat().st_size
                    and saved['sha256'] == digest(destination)):
                check_netcdf(destination)
                return 'skipped'
        except (ValueError, KeyError, RuntimeError):
            pass
    # A unique temporary file keeps interrupted or concurrent downloads separate.
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.part', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        curl(item['url'], temporary, auth, cookies)
        check_netcdf(temporary)
        saved = {'url': item['url'], 'bytes': temporary.stat().st_size,
                 'sha256': digest(temporary), 'downloaded_utc': datetime.now(timezone.utc).isoformat()}
        temporary.replace(destination)
        save_json(receipt, saved)
    finally:
        temporary.unlink(missing_ok=True)
    return 'downloaded'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', help='UTC start date or ISO timestamp')
    parser.add_argument('--end', help='UTC end date (inclusive day) or ISO timestamp')
    parser.add_argument('--latest', type=positive, metavar='N', help='Get N newest files instead of a date range')
    parser.add_argument('--max-files', type=positive, help='Limit a date-range search to this many files')
    parser.add_argument('--bbox', type=bounding_box, help='west,south,east,north; use --bbox=-80,20,-50,45')
    parser.add_argument('--product', choices=['expert', 'basic', 'unsmoothed', 'windwave'], default='expert')
    parser.add_argument('--output', type=Path, default=Path('data/swot_l2'))
    parser.add_argument('--netrc', type=Path, default=Path.home() / '.netrc', help='Earthdata credentials file')
    parser.add_argument('--dry-run', action='store_true', help='List matches without downloading or requiring login')
    args = parser.parse_args()
    if args.latest:
        if args.start or args.end or args.max_files:
            parser.error('--latest cannot be combined with --start, --end, or --max-files')
    elif not (args.start and args.end):
        parser.error('provide --start and --end, or --latest N')
    if args.start:
        try:
            args.start, args.end = timestamp(args.start), timestamp(args.end, end=True)
            if datetime.fromisoformat(args.start.replace('Z', '+00:00')) > datetime.fromisoformat(args.end.replace('Z', '+00:00')):
                raise ValueError('start is after end')
        except ValueError as error:
            parser.error(f'invalid date range: {error}')
    if not shutil.which('curl'):
        parser.error('curl is required')
    short_name, files = search(args)
    print(f'{short_name}: {len(files)} file(s), estimated {sum(f["estimated_mb"] for f in files):.1f} MB')
    for item in files:
        print(f'  {item["filename"]}')
    if args.dry_run or not files:
        return 0
    args.netrc = args.netrc.expanduser().resolve()
    try:
        credentials = netrc.netrc(str(args.netrc)).authenticators('urs.earthdata.nasa.gov')
        if not credentials or not credentials[0] or not credentials[2]:
            raise ValueError
    except (OSError, netrc.NetrcParseError, ValueError):
        raise RuntimeError('Add your Earthdata login for machine urs.earthdata.nasa.gov '
                           f'to {args.netrc}; see FETCH_SWOT.md.') from None
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'collection': short_name, 'searched_utc': datetime.now(timezone.utc).isoformat(),
                'start': args.start, 'end': args.end, 'bbox': args.bbox, 'files': files}
    failures = 0
    with tempfile.TemporaryDirectory(prefix='swot-login-') as tmp:
        cookies = Path(tmp) / 'cookies'
        for i, item in enumerate(files, 1):
            try:
                item['status'] = download(item, args.output, args.netrc, cookies)
            except (RuntimeError, OSError) as error:
                failures += 1
                item['status'] = 'failed'
                print(f'ERROR: {error}', file=sys.stderr)
            print(f'[{i}/{len(files)}] {item["status"]}: {item["filename"]}', flush=True)
            save_json(args.output / 'download_manifest.json', manifest)
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
