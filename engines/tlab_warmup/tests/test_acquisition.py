import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tlab_warmup.data import acquire


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class AcquisitionTests(unittest.TestCase):
    def config(self, payload):
        return {"source": {"repository": "fixture", "revision": "pinned", "file": "x",
                           "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
                "maximum_download_bytes": 100, "acquisition_seconds": 900}

    def test_download_and_verified_reuse(self):
        payload = b"synthetic-download"
        c = self.config(payload)
        with tempfile.TemporaryDirectory() as d:
            with patch("urllib.request.urlopen", return_value=Response(payload)) as urlopen:
                p = acquire(c, Path(d), None, lambda _: None)
                self.assertEqual(p.read_bytes(), payload)
                self.assertEqual(urlopen.call_count, 1)
            with patch("urllib.request.urlopen") as urlopen:
                self.assertEqual(acquire(c, Path(d), None, lambda _: None), p)
                urlopen.assert_not_called()

    def test_resume_range_header_and_integrity(self):
        payload = b"synthetic-download"
        c = self.config(payload)
        with tempfile.TemporaryDirectory() as d:
            Path(d, "multi_event.parquet.part").write_bytes(payload[:5])
            with patch("urllib.request.urlopen", return_value=Response(payload[5:], 206,
                       {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"})) as urlopen:
                p = acquire(c, Path(d), None, lambda _: None)
                self.assertEqual(p.read_bytes(), payload)
                self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=5-")

    def test_refused_resume_preserves_partial(self):
        payload = b"synthetic-download"
        with tempfile.TemporaryDirectory() as d:
            partial = Path(d, "multi_event.parquet.part")
            partial.write_bytes(payload[:5])
            with patch("urllib.request.urlopen", return_value=Response(payload)):
                with self.assertRaisesRegex(RuntimeError, "refused resume"):
                    acquire(self.config(payload), Path(d), None, lambda _: None)
            self.assertEqual(partial.read_bytes(), payload[:5])

    def test_deadline_preserves_partial(self):
        payload = b"synthetic-download"
        with tempfile.TemporaryDirectory() as d:
            with patch("urllib.request.urlopen", return_value=Response(payload)):
                with patch("time.monotonic", side_effect=[0, 901]):
                    with self.assertRaisesRegex(TimeoutError, "budget"):
                        acquire(self.config(payload), Path(d), None, lambda _: None)
            self.assertTrue(Path(d, "multi_event.parquet.part").exists())

    def test_byte_limit_checked_before_access(self):
        payload = b"synthetic-download"
        c = self.config(payload)
        c["maximum_download_bytes"] = 1
        with tempfile.TemporaryDirectory() as d, patch("urllib.request.urlopen") as urlopen:
            with self.assertRaisesRegex(ValueError, "byte limit"):
                acquire(c, Path(d), None, lambda _: None)
            urlopen.assert_not_called()
