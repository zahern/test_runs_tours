import gzip
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "54_extract_parallel.py")
_spec = importlib.util.spec_from_file_location("extract54", _TARGET)
assert _spec is not None and _spec.loader is not None
_e54 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e54)


def _gz_bytes(payload):
    return gzip.compress(payload)


def _write(path, data):
    with open(path, "wb") as handle:
        handle.write(data)
    return path


_RAW_HEADER = ("StartDate,SnappedPath,TimestampPath,SpeedPath_kph,"
               "XAccPath,YAccPath,VehicleType,VehicleID,TripID\n")
_RAW_ROW = ('2025-03-01,"153.0 -27.45,153.01 -27.46,153.02 -27.47",'
            '"2025-03-01T10:00:00,2025-03-01T10:00:05,2025-03-01T10:00:10",'
            '"60,80,120","-0.5,-0.35,0.35","0.1,0.4,0.45",CAR,V1,T1\n')


class IsGzipValid(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = self._tmp.name

    def _path(self, name):
        return os.path.join(self.base, name)

    def test_valid_single_member(self):
        path = self._path("ok.csv.gz")
        _write(path, _gz_bytes(b"col\nval\n"))
        self.assertTrue(_e54.is_valid_gzip(path))

    def test_valid_concatenated_members(self):
        path = self._path("concat.csv.gz")
        _write(path, _gz_bytes(b"col\nval1\n") + _gz_bytes(b"col\nval2\n"))
        self.assertTrue(_e54.is_valid_gzip(path))

    def test_not_gzip(self):
        path = self._path("plain.csv")
        _write(path, b"col\nval\n")
        self.assertFalse(_e54.is_valid_gzip(path))

    def test_empty_file(self):
        path = self._path("empty.csv.gz")
        open(path, "wb").close()
        self.assertFalse(_e54.is_valid_gzip(path))

    def test_truncated_first_member(self):
        good = _gz_bytes(b"col\nval\n")
        path = self._path("trunc1.csv.gz")
        _write(path, good[: len(good) // 2])
        self.assertFalse(_e54.is_valid_gzip(path))

    def test_valid_first_truncated_second(self):
        good = _gz_bytes(b"col\nval\n")
        path = self._path("trunc2.csv.gz")
        _write(path, good + good[: len(good) // 2])
        self.assertFalse(_e54.is_valid_gzip(path))

    def test_corrupt_crc(self):
        good = bytearray(_gz_bytes(b"col\nval\n"))
        good[-5] ^= 0xFF
        path = self._path("badcrc.csv.gz")
        _write(path, bytes(good))
        self.assertFalse(_e54.is_valid_gzip(path))


class ProcessFileContract(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = self._tmp.name
        self.out_base = os.path.join(self.base, "out", "traffic_points.csv")
        os.makedirs(os.path.dirname(self.out_base), exist_ok=True)
        _e54._init(self.out_base, True)

    def _fixture(self, name, data):
        return _write(os.path.join(self.base, name), data)

    def test_valid_outside_bbox_returns_zero_no_shard(self):
        csv = (_RAW_HEADER +
               '2025-03-01,"140.0 -20.0",'
               '"2025-03-01T10:00:00","60","-0.5","0.1",CAR,V1,T1\n')
        path = self._fixture("outside.csv.gz", _gz_bytes(csv.encode()))
        self.assertEqual(_e54.process_file(path), 0)
        self.assertEqual(os.listdir(os.path.dirname(self.out_base)), [])

    def test_corrupt_gzip_returns_minus_one_no_shard_writes_error_log(self):
        path = self._fixture("corrupt.csv.gz", b"\x1f\x8b" + b"garbage" * 100)
        self.assertEqual(_e54.process_file(path), -1)
        self.assertFalse(any(name.endswith(".gz")
                             for name in os.listdir(os.path.dirname(self.out_base))))
        errors = os.path.join(os.path.dirname(self.out_base),
                              "traffic_points.csv.pid%d.errors.jsonl" % os.getpid())
        self.assertTrue(os.path.exists(errors))
        with open(errors, encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        self.assertEqual(record["source"], "corrupt.csv.gz")
        self.assertIn("error_type", record)

    def test_value_error_from_read_points_propagates(self):
        path = self._fixture("valid.csv.gz", _gz_bytes(b"col\nval\n"))
        with mock.patch.object(_e54, "read_points", side_effect=ValueError("bad")):
            with self.assertRaises(ValueError):
                _e54.process_file(path)

    def test_os_error_missing_file_propagates(self):
        with self.assertRaises(OSError):
            _e54.process_file(os.path.join(self.base, "missing.csv.gz"))


class MainCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _TARGET
        cls.python = sys.executable

    def _run(self, raw_dir, out_path):
        return subprocess.run(
            [self.python, self.script, "--raw-dir", raw_dir, "--out", out_path,
             "--workers", "1", "--all-points"],
            capture_output=True, text=True)

    def _write_raw(self, raw_dir, name, payload):
        os.makedirs(raw_dir, exist_ok=True)
        return _write(os.path.join(raw_dir, name), payload)

    def test_success_writes_complete_marker(self):
        with tempfile.TemporaryDirectory() as base:
            raw_dir = os.path.join(base, "raw")
            out_dir = os.path.join(base, "out")
            out_path = os.path.join(out_dir, "traffic_points.csv.gz")
            self._write_raw(raw_dir, "trips-a.csv.gz",
                            _gz_bytes((_RAW_HEADER + _RAW_ROW).encode()))
            result = self._run(raw_dir, out_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            marker = out_path + ".complete.json"
            self.assertTrue(os.path.exists(marker))
            with open(marker, encoding="utf-8") as handle:
                status = json.load(handle)
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["input_files"], 1)
            self.assertEqual(status["skipped_files"], 0)
            self.assertEqual(status["point_rows"], 3)
            self.assertTrue(os.path.exists(out_path))

    def test_corrupt_plus_valid_partial(self):
        with tempfile.TemporaryDirectory() as base:
            raw_dir = os.path.join(base, "raw")
            out_dir = os.path.join(base, "out")
            out_path = os.path.join(out_dir, "traffic_points.csv.gz")
            self._write_raw(raw_dir, "trips-bad.csv.gz",
                            b"\x1f\x8b" + b"junkjunkjunk")
            self._write_raw(raw_dir, "trips-good.csv.gz",
                            _gz_bytes((_RAW_HEADER + _RAW_ROW).encode()))
            result = self._run(raw_dir, out_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            marker = out_path + ".complete.json"
            self.assertTrue(os.path.exists(marker))
            with open(marker, encoding="utf-8") as handle:
                status = json.load(handle)
            self.assertEqual(status["status"], "partial")
            self.assertEqual(status["skipped_files"], 1)
            self.assertEqual(status["point_rows"], 3)
            self.assertEqual(status["input_files"], 2)

    def test_no_valid_points_fails_no_marker(self):
        with tempfile.TemporaryDirectory() as base:
            raw_dir = os.path.join(base, "raw")
            out_dir = os.path.join(base, "out")
            out_path = os.path.join(out_dir, "traffic_points.csv.gz")
            self._write_raw(raw_dir, "trips-bad.csv.gz",
                            b"\x1f\x8b" + b"junkjunkjunk")
            result = self._run(raw_dir, out_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(os.path.exists(out_path + ".complete.json"))

    def test_rerun_no_inputs_clears_marker_and_fails(self):
        with tempfile.TemporaryDirectory() as base:
            raw_dir = os.path.join(base, "raw")
            out_dir = os.path.join(base, "out")
            out_path = os.path.join(out_dir, "traffic_points.csv.gz")
            marker = out_path + ".complete.json"
            os.makedirs(out_dir, exist_ok=True)
            with open(marker, "w", encoding="utf-8") as handle:
                json.dump({"status": "complete"}, handle)
            os.makedirs(raw_dir, exist_ok=True)
            result = self._run(raw_dir, out_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()
