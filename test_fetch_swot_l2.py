import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fetch_swot_l2 as fetch


class DownloaderTests(unittest.TestCase):
    def test_utc_dates_and_box(self):
        self.assertEqual(fetch.timestamp('2026-09-18'), '2026-09-18T00:00:00Z')
        self.assertEqual(fetch.timestamp('2026-09-18', end=True), '2026-09-18T23:59:59.999999Z')
        self.assertEqual(fetch.timestamp('2026-09-18T01:00:00-04:00'), '2026-09-18T05:00:00Z')
        with self.assertRaises(argparse.ArgumentTypeError):
            fetch.bounding_box('170,-10,-170,10')

    def test_catalog_pagination_and_link_selection(self):
        def entry(i):
            return {'links': [{'href': f'https://{fetch.ARCHIVE}/data/file{i}.nc'},
                              {'href': 'https://example.com/unrelated.nc'}]}
        args = argparse.Namespace(product='expert', latest=None, max_files=2001,
                                  start='2026-09-01T00:00:00Z', end='2026-09-18T23:59:59Z', bbox=None)
        with patch.object(fetch, 'catalog', side_effect=[
            [{'id': 'collection'}], [entry(i) for i in range(2000)], [entry(2000)],
        ]) as request:
            _, items = fetch.search(args)
        self.assertEqual(len(items), 2001)
        self.assertEqual(request.call_args.args[1]['page_num'], 2)
        self.assertTrue(all('example.com' not in item['url'] for item in items))

    def test_download_skip_and_corruption_recovery(self):
        item = {'filename': 'sample.nc', 'url': f'https://{fetch.ARCHIVE}/sample.nc'}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            payload = fetch.MAGIC + b'test transfer bytes'
            def transfer(url, output, auth, cookies):
                output.write_bytes(payload)
            with patch.object(fetch, 'curl', side_effect=transfer) as curl:
                self.assertEqual(fetch.download(item, folder, 'netrc', 'cookies'), 'downloaded')
                self.assertEqual(fetch.download(item, folder, 'netrc', 'cookies'), 'skipped')
                (folder / 'sample.nc').write_bytes(fetch.MAGIC + b'corrupted')
                self.assertEqual(fetch.download(item, folder, 'netrc', 'cookies'), 'downloaded')
                self.assertEqual(curl.call_count, 2)

    def test_login_html_does_not_replace_existing_file(self):
        item = {'filename': 'sample.nc', 'url': f'https://{fetch.ARCHIVE}/sample.nc'}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            existing = folder / 'sample.nc'
            existing.write_bytes(fetch.MAGIC + b'original')
            def transfer(url, output, auth, cookies):
                output.write_text('<html>Earthdata Login</html>')
            with patch.object(fetch, 'curl', side_effect=transfer):
                with self.assertRaises(RuntimeError):
                    fetch.download(item, folder, 'netrc', 'cookies')
            self.assertEqual(existing.read_bytes(), fetch.MAGIC + b'original')
            self.assertEqual(list(folder.glob('*.part')), [])


if __name__ == '__main__':
    unittest.main()
