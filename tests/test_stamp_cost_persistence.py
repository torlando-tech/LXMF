import contextlib
import importlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


router_module = importlib.import_module("LXMF.LXMRouter")
msgpack = router_module.msgpack
RealThread = threading.Thread


class ImmediateThread:
    def __init__(self, target, daemon=None):
        self.target = target

    def setDaemon(self, daemon):
        pass

    def start(self):
        self.target()


class TestStampCostPersistence(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.wall = 2_000_000_000.0
        self.monotonic = 100.0
        for patch in (
            mock.patch.object(router_module.time, "time", side_effect=lambda: self.wall),
            mock.patch.object(router_module.time, "monotonic", side_effect=lambda: self.monotonic),
            mock.patch.object(router_module.RNS, "Destination"),
            mock.patch.object(router_module.RNS.Transport, "register_announce_handler"),
            mock.patch.object(router_module.signal, "signal"),
            mock.patch.object(router_module.atexit, "register"),
            mock.patch.object(router_module.LXMRouter, "jobloop"),
            mock.patch.object(router_module.threading, "Thread", ImmediateThread),
        ):
            self.stack.enter_context(patch)
        self.router = router_module.LXMRouter(identity=mock.Mock(), storagepath=self.directory)
        self.path = Path(self.router.storagepath) / "outbound_stamp_costs"

    def read_cache(self):
        return msgpack.unpackb(self.path.read_bytes())

    def test_burst_preserves_latest_values_and_timestamps_with_one_periodic_write(self):
        first = b"a" * 16
        second = b"b" * 16
        expected = {}
        with mock.patch.object(router_module.os, "replace", wraps=router_module.os.replace) as replace:
            for destination, cost, timestamp in (
                (first, 8, 2_000_000_001.0),
                (second, None, 2_000_000_002.0),
                (first, 9, 2_000_000_003.0),
                (first, 9, 2_000_000_004.0),
            ):
                self.wall = timestamp
                self.router.update_stamp_cost(destination, cost)
                expected[destination] = [timestamp, cost]
                self.assertEqual(cost, self.router.get_outbound_stamp_cost(destination))
            self.assertFalse(self.path.exists(), "announces must not rewrite the cache individually")
            self.monotonic = 159.0
            self.router.jobs()
            self.assertFalse(self.path.exists())
            self.monotonic = 161.0
            self.router.jobs()
            self.assertEqual(expected, self.read_cache())
            self.assertEqual(1, replace.call_count)
            self.monotonic = 300.0
            self.router.jobs()
            self.assertEqual(1, replace.call_count, "an idle cache must not be rewritten")

    def test_shutdown_flushes_pending_costs_and_closes_updates(self):
        destination = b"a" * 16
        self.wall = 2_000_000_005.0
        self.router.update_stamp_cost(destination, 11)
        self.router.exit_handler()
        expected = {destination: [2_000_000_005.0, 11]}
        self.assertEqual(expected, self.read_cache())
        self.router.update_stamp_cost(b"b" * 16, 19)
        self.assertIsNone(self.router.get_outbound_stamp_cost(b"b" * 16))
        self.monotonic = 300.0
        self.router.jobs()
        self.router.exit_handler()
        self.assertEqual(expected, self.read_cache())
        reloaded = router_module.LXMRouter(identity=mock.Mock(), storagepath=self.directory)
        self.assertEqual(11, reloaded.get_outbound_stamp_cost(destination))
        self.assertEqual(expected, reloaded.outbound_stamp_costs)

    def test_failed_replace_preserves_old_cache_cleans_temp_and_retries(self):
        destination = b"a" * 16
        self.router.update_stamp_cost(destination, 8)
        self.monotonic = 161.0
        self.router.jobs()
        original = self.path.read_bytes()
        self.wall = 2_000_000_020.0
        self.router.update_stamp_cost(destination, 12)
        self.monotonic = 222.0
        with mock.patch.object(router_module.os, "replace", side_effect=OSError("write failed")):
            self.router.jobs()
        self.assertEqual(original, self.path.read_bytes())
        self.assertEqual([], list(self.path.parent.glob("outbound_stamp_costs.tmp.*")))
        self.monotonic = 226.0
        self.router.jobs()
        self.assertEqual({destination: [2_000_000_020.0, 12]}, self.read_cache())

    def test_wall_clock_changes_do_not_delay_periodic_persistence(self):
        destination = b"a" * 16
        self.router.update_stamp_cost(destination, 7)
        self.wall -= 3600
        self.monotonic = 160.0
        self.router.jobs()
        self.assertEqual({destination: [2_000_000_000.0, 7]}, self.read_cache())
        self.wall = 2_000_100_000.0
        self.router.update_stamp_cost(destination, None)
        self.monotonic = 219.0
        self.router.jobs()
        self.assertEqual(7, self.read_cache()[destination][1])
        self.monotonic = 220.0
        self.router.jobs()
        self.assertEqual({destination: [2_000_100_000.0, None]}, self.read_cache())

    def test_update_during_flush_is_preserved_by_the_next_flush(self):
        destination = b"a" * 16
        self.router.update_stamp_cost(destination, 8)
        self.monotonic = 161.0
        entered = threading.Event()
        release = threading.Event()
        attempted = threading.Event()
        errors = []
        original_replace = router_module.os.replace

        def paused_replace(source, target):
            entered.set()
            if not release.wait(5): raise TimeoutError("flush not released")
            return original_replace(source, target)

        def flush():
            try: self.router.jobs()
            except Exception as error: errors.append(error)

        def update():
            attempted.set()
            try: self.router.update_stamp_cost(destination, 12)
            except Exception as error: errors.append(error)

        with mock.patch.object(router_module.os, "replace", side_effect=paused_replace):
            writer = RealThread(target=flush, daemon=True)
            updater = RealThread(target=update, daemon=True)
            writer.start()
            try:
                self.assertTrue(entered.wait(5))
                self.wall = 2_000_000_020.0
                updater.start()
                self.assertTrue(attempted.wait(5))
            finally:
                release.set()
                writer.join(5)
                if updater.ident is not None: updater.join(5)
            self.assertFalse(writer.is_alive())
            self.assertFalse(updater.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(12, self.router.get_outbound_stamp_cost(destination))
        self.assertEqual({destination: [2_000_000_000.0, 8]}, self.read_cache())
        self.monotonic = 221.0
        self.router.jobs()
        self.assertEqual({destination: [2_000_000_020.0, 12]}, self.read_cache())

    def test_continuous_updates_do_not_postpone_the_flush_deadline(self):
        destination = b"a" * 16
        for second in range(60):
            self.monotonic = 100.0 + second
            self.wall = 2_000_000_000.0 + second
            self.router.update_stamp_cost(destination, second)
            self.router.jobs()
        self.assertFalse(self.path.exists())
        self.monotonic = 160.0
        self.router.jobs()
        self.assertEqual({destination: [2_000_000_059.0, 59]}, self.read_cache())
