import concurrent.futures
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from LXMF.Storage import LXMFStore


class StorageThreadingTests(unittest.TestCase):
    def test_multithread_and_serialized_builds_preserve_costs_and_atomic_claims(self):
        for threadsafety in (1, 3):
            with self.subTest(threadsafety=threadsafety), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "state.db"
                with mock.patch.object(sqlite3, "threadsafety", threadsafety):
                    store = LXMFStore(str(database))
                destinations = [index.to_bytes(16, "big") for index in range(16)]
                transient_id = bytes.fromhex("ab" * 32)
                received = time.time()

                def update(index):
                    cost = 8 + index / 256
                    self.assertTrue(store.update_stamp_cost(destinations[index], cost))
                    self.assertEqual(cost, store.get_outbound_stamp_cost(destinations[index]))
                    return store.conditionally_claim_delivered(transient_id, received)

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        claims = list(executor.map(update, range(len(destinations))))
                    self.assertEqual(1, sum(claims))
                    self.assertEqual({transient_id}, store.known_delivered([transient_id]))
                finally:
                    store.close()

                reopened = LXMFStore(str(database))
                try:
                    self.assertEqual(
                        [8 + index / 256 for index in range(len(destinations))],
                        [reopened.get_outbound_stamp_cost(destination) for destination in destinations],
                    )
                    self.assertTrue(reopened.has_delivered(transient_id))
                finally:
                    reopened.close()

    def test_single_thread_build_is_rejected_before_creating_a_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.db"
            with mock.patch.object(sqlite3, "threadsafety", 0):
                with self.assertRaises(RuntimeError):
                    LXMFStore(str(database))
            self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
