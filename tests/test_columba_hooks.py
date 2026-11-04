import importlib
import threading
import types
import unittest
from unittest import mock

import RNS


router_module = importlib.import_module("LXMF.LXMRouter")
stamper = importlib.import_module("LXMF.LXStamper")


class ImmediateThread:
    def __init__(self, target, daemon=None):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class DeferredThread(ImmediateThread):
    instances = []

    def __init__(self, target, daemon=None):
        super().__init__(target, daemon)
        self.__class__.instances.append(self)

    def start(self):
        pass

    def run(self):
        self.target()


class TestColumbaHooks(unittest.TestCase):
    def tearDown(self):
        stamper.set_external_generator(None)
        self.assertEqual({}, stamper.active_jobs)

    @staticmethod
    def stamp_for_cost(workblock, cost):
        for candidate in range(100000):
            stamp = candidate.to_bytes(stamper.STAMP_SIZE, byteorder="big")
            if stamper.stamp_valid(stamp, cost, workblock):
                return stamp
        raise AssertionError("could not find inexpensive test stamp")

    def test_external_stamp_generator_takes_precedence_on_android(self):
        calls = []
        expected_stamp = self.stamp_for_cost(b"message-id", 1)

        def external_generator(workblock, stamp_cost):
            calls.append((workblock, stamp_cost))
            return expected_stamp, 7

        stamper.set_external_generator(external_generator)
        with (
            mock.patch.object(stamper.RNS.vendor.platformutils, "is_android", return_value=True),
            mock.patch.object(stamper.RNS.Stamps, "generate_stamp", side_effect=AssertionError("upstream workers selected")),
        ):
            stamp, value = stamper.generate_stamp(b"message-id", 1, expand_rounds=0)

        self.assertEqual(expected_stamp, stamp)
        self.assertGreaterEqual(value, 1)
        self.assertEqual([(b"message-id", 1)], calls)

    def test_external_generator_accepts_valid_nonzero_cost_stamp(self):
        expected_stamp = self.stamp_for_cost(b"message-id", 3)
        stamper.set_external_generator(lambda _workblock, _cost: (expected_stamp, 42))

        stamp, value = stamper.generate_stamp(b"message-id", 3, expand_rounds=0)

        self.assertEqual(expected_stamp, stamp)
        self.assertTrue(stamper.stamp_valid(stamp, 3, b"message-id"))

    def test_external_generator_accepts_canonical_exact_target_boundary(self):
        cost = 3
        boundary_digest = (1 << (256-cost)).to_bytes(32, byteorder="big")
        expected_stamp = bytes(stamper.STAMP_SIZE)
        stamper.set_external_generator(lambda _workblock, _cost: (expected_stamp, 1))

        with mock.patch.object(stamper.RNS.Identity, "full_hash", return_value=boundary_digest):
            stamp, value = stamper.generate_stamp(b"message-id", cost, expand_rounds=0)

        self.assertEqual(expected_stamp, stamp)
        self.assertEqual(cost - 1/256, value)

    def test_external_generator_rejects_malformed_results(self):
        malformed_results = [
            None,
            b"not-a-tuple",
            (b"only-one-item",),
            (b"one", 1, "extra"),
            [bytes(stamper.STAMP_SIZE), 1],
            ("not-bytes", 1),
            (bytearray(stamper.STAMP_SIZE), 1),
            (bytes(stamper.STAMP_SIZE), "not-an-int"),
            (bytes(stamper.STAMP_SIZE), True),
            (bytes(stamper.STAMP_SIZE), -1),
            (bytes(stamper.STAMP_SIZE), 10**10000),
            (bytes(stamper.STAMP_SIZE - 1), 1),
            (bytes(stamper.STAMP_SIZE + 1), 1),
        ]
        for index, result in enumerate(malformed_results):
            with self.subTest(case=index):
                def external_generator(_workblock, _cost):
                    return result

                stamper.set_external_generator(external_generator)
                with mock.patch.object(stamper.RNS, "log") as log:
                    stamp, value = stamper.generate_stamp(b"message-id", 0, expand_rounds=0)
                self.assertIsNone(stamp)
                self.assertEqual(0, value)
                self.assertTrue(log.called)
                self.assertEqual({}, stamper.active_jobs)

    def test_external_generator_rejects_insufficient_stamp_value(self):
        insufficient_stamp = next(
            candidate.to_bytes(stamper.STAMP_SIZE, byteorder="big")
            for candidate in range(100)
            if not stamper.stamp_valid(candidate.to_bytes(stamper.STAMP_SIZE, byteorder="big"), 24, b"message-id")
        )
        self.assertFalse(stamper.stamp_valid(insufficient_stamp, 24, b"message-id"))
        stamper.set_external_generator(lambda _workblock, _cost: (insufficient_stamp, 1))

        stamp, value = stamper.generate_stamp(b"message-id", 24, expand_rounds=0)

        self.assertIsNone(stamp)
        self.assertEqual(0, value)

    def test_external_generator_exception_fails_closed_and_cleans_up(self):
        def exploding_generator(_workblock, _cost):
            raise RuntimeError("native bridge failed")

        stamper.set_external_generator(exploding_generator)
        with (
            mock.patch.object(stamper.RNS, "log") as log,
            mock.patch.object(stamper.RNS, "trace_exception") as trace_exception,
        ):
            stamp, value = stamper.generate_stamp(b"message-id", 0, expand_rounds=0)

        self.assertIsNone(stamp)
        self.assertEqual(0, value)
        self.assertEqual({}, stamper.active_jobs)
        self.assertTrue(any("external stamp generator" in str(call).lower() for call in log.call_args_list))
        trace_exception.assert_called_once()

    def test_cancel_work_reaches_cooperative_external_job_and_discards_result(self):
        started = threading.Event()
        release = threading.Event()
        cancel_calls = []
        result = []

        def external_generator(_workblock, _cost, cancellation_token):
            started.set()
            release.wait(2)
            return b"", 8

        def cancel_external(cancellation_token):
            cancel_calls.append(cancellation_token.message_id)
            self.assertTrue(cancellation_token.cancelled)
            release.set()

        message_id = b"cancelled-message"
        stamper.set_external_generator(
            external_generator, cancel_external, pass_cancellation_token=True
        )
        worker = threading.Thread(
            target=lambda: result.append(stamper.generate_stamp(message_id, 1, expand_rounds=0))
        )
        worker.start()
        self.assertTrue(started.wait(1))
        self.assertIn(message_id, stamper.active_jobs)

        with mock.patch.object(stamper.RNS, "trace_exception") as trace_exception:
            stamper.cancel_work(message_id)
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual([message_id], cancel_calls)
        self.assertEqual([(None, 0)], result)
        self.assertNotIn(message_id, stamper.active_jobs)
        trace_exception.assert_not_called()

    def test_cancelled_noncooperative_external_job_discards_late_result(self):
        started = threading.Event()
        release = threading.Event()
        expected_stamp = self.stamp_for_cost(b"message-id", 1)
        result = []

        def external_generator(_workblock, _cost):
            started.set()
            release.wait(2)
            return expected_stamp, 5

        message_id = b"noncooperative-message"
        stamper.set_external_generator(external_generator)
        worker = threading.Thread(
            target=lambda: result.append(stamper.generate_stamp(message_id, 1, expand_rounds=0))
        )
        worker.start()
        self.assertTrue(started.wait(1))
        stamper.cancel_work(message_id)
        release.set()
        worker.join(2)

        self.assertEqual([(None, 0)], result)
        self.assertNotIn(message_id, stamper.active_jobs)

    def test_reset_cancels_active_generator_and_prevents_stale_result(self):
        started = threading.Event()
        release = threading.Event()
        cancel_calls = []
        expected_stamp = self.stamp_for_cost(b"message-id", 1)
        result = []

        def external_generator(_workblock, _cost, _token):
            started.set()
            release.wait(2)
            return expected_stamp, 3

        def cancel_external(token):
            cancel_calls.append(token.message_id)
            release.set()

        message_id = b"stale-generator-message"
        stamper.set_external_generator(external_generator, cancel_external)
        worker = threading.Thread(
            target=lambda: result.append(stamper.generate_stamp(message_id, 1, expand_rounds=0))
        )
        worker.start()
        self.assertTrue(started.wait(1))

        stamper.set_external_generator(None)
        worker.join(2)

        self.assertEqual([message_id], cancel_calls)
        self.assertEqual([(None, 0)], result)
        self.assertEqual({}, stamper.active_jobs)

    def test_replacement_cancels_active_generator_and_new_generator_is_used(self):
        started = threading.Event()
        release = threading.Event()
        cancel_calls = []
        stale_stamp = self.stamp_for_cost(b"message-id", 1)
        fresh_stamp = self.stamp_for_cost(b"fresh-message", 2)
        stale_result = []

        def stale_generator(_workblock, _cost, _token):
            started.set()
            release.wait(2)
            return stale_stamp, 3

        def cancel_stale(token):
            cancel_calls.append(token.message_id)
            release.set()

        message_id = b"replaced-generator-message"
        stamper.set_external_generator(stale_generator, cancel_stale)
        worker = threading.Thread(
            target=lambda: stale_result.append(stamper.generate_stamp(message_id, 1, expand_rounds=0))
        )
        worker.start()
        self.assertTrue(started.wait(1))

        stamper.set_external_generator(lambda _workblock, _cost: (fresh_stamp, 2))
        worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual([message_id], cancel_calls)
        self.assertEqual([(None, 0)], stale_result)
        stamp, _value = stamper.generate_stamp(b"fresh-message", 2, expand_rounds=0)
        self.assertEqual(fresh_stamp, stamp)
        self.assertEqual({}, stamper.active_jobs)

    def test_fractional_cost_and_no_expand_reach_external_generator_unchanged(self):
        material = b"fractional-material"
        cost = 2.5
        expected = self.stamp_for_cost(material, cost)
        calls = []
        def generator(workblock, target_cost):
            calls.append((workblock, target_cost))
            return expected, 0x7FFFFFFFFFFFFFFF
        stamper.set_external_generator(generator)

        stamp, value = stamper.generate_stamp(material, cost, expand_rounds=99, no_expand=True)

        self.assertEqual([(material, cost)], calls)
        self.assertEqual(expected, stamp)
        self.assertEqual(stamper.stamp_value(material, expected), value)

    def test_fractional_validation_uses_canonical_target_not_reported_value(self):
        material = b"fractional-boundary"
        cost = 2.5
        digest = RNS.Stamps.target_from_fp(RNS.Stamps.cost_to_fp(cost)).to_bytes(32, "big")
        expected = bytes(stamper.STAMP_SIZE)
        stamper.set_external_generator(lambda _workblock, _cost: (expected, 1))

        with mock.patch.object(RNS.Identity, "full_hash", return_value=digest):
            stamp, value = stamper.generate_stamp(material, cost, no_expand=True)
            expected_value = stamper.stamp_value(material, expected)

        self.assertEqual(expected, stamp)
        self.assertEqual(expected_value, value)
        self.assertLess(value, cost)

    def test_fallback_generates_canonical_stamp_with_unexpanded_material(self):
        material = b"fallback"
        stamper.set_external_generator(None)
        stamp, value = stamper.generate_stamp(material, 0, expand_rounds=17, no_expand=True)

        self.assertIsInstance(stamp, bytes)
        self.assertEqual(32, len(stamp))
        self.assertTrue(stamper.stamp_valid(stamp, 0, material))
        self.assertEqual(stamper.stamp_value(material, stamp), value)

    def test_external_job_does_not_replace_existing_upstream_job(self):
        message_id = b"upstream-job"
        existing_job = object()
        generator = mock.Mock(return_value=(bytes(stamper.STAMP_SIZE), 1))
        stamper.set_external_generator(generator)
        with mock.patch.dict(RNS.Stamps.active_jobs, {message_id: existing_job}):
            self.assertEqual((None, 0), stamper.generate_stamp(message_id, 0, no_expand=True))
            self.assertIs(existing_job, RNS.Stamps.active_jobs[message_id])
        generator.assert_not_called()

    def test_replacement_does_not_orphan_duplicate_external_job(self):
        message_id = b"duplicate-job"
        started = threading.Event()
        release = threading.Event()
        results = []

        def first(_workblock, _cost, token):
            started.set()
            release.wait(2)
            self.assertTrue(token.cancelled)
            return None

        stamper.set_external_generator(first)
        worker = threading.Thread(target=lambda: results.append(
            stamper.generate_stamp(message_id, 0, no_expand=True)
        ))
        worker.start()
        self.assertTrue(started.wait(1))
        replacement = mock.Mock(return_value=(bytes(stamper.STAMP_SIZE), 1))
        stamper.set_external_generator(replacement, pass_cancellation_token=False)
        try:
            self.assertEqual((None, 0), stamper.generate_stamp(message_id, 0, no_expand=True))
            replacement.assert_not_called()
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual([(None, 0)], results)

    def test_stock_fallback_cancellation_survives_generator_registration(self):
        for register_external in (False, True):
            with self.subTest(register_external=register_external):
                message_id = b"fallback-running"
                started = threading.Event()
                release = threading.Event()
                results = []

                def nonmatching_hash(_material):
                    started.set()
                    release.wait(2)
                    return b"\xff" * 32

                stamper.set_external_generator(None)
                with mock.patch.object(RNS.Identity, "full_hash", side_effect=nonmatching_hash):
                    worker = threading.Thread(target=lambda: results.append(
                        stamper.generate_stamp(message_id, 1, no_expand=True)
                    ))
                    worker.start()
                    self.assertTrue(started.wait(1))
                    if register_external:
                        replacement = mock.Mock(return_value=(bytes(stamper.STAMP_SIZE), 1))
                        stamper.set_external_generator(replacement, pass_cancellation_token=False)
                        self.assertEqual((None, 0), stamper.generate_stamp(message_id, 0, no_expand=True))
                        replacement.assert_not_called()
                    try:
                        stamper.cancel_work(message_id)
                    finally:
                        release.set()
                        worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual([(None, None)], results)
                self.assertNotIn(message_id, RNS.Stamps.active_jobs)


    def test_replacement_during_workblock_creation_never_calls_stale_generator(self):
        stale = mock.Mock(return_value=(bytes(stamper.STAMP_SIZE), 1))
        fresh = mock.Mock(return_value=(bytes(stamper.STAMP_SIZE), 1))
        stamper.set_external_generator(stale, pass_cancellation_token=False)

        def replace(*_args, **_kwargs):
            stamper.set_external_generator(fresh, pass_cancellation_token=False)
            return b"workblock"

        with mock.patch.object(stamper, "stamp_workblock", side_effect=replace):
            self.assertEqual((None, 0), stamper.generate_stamp(b"stale-start", 0))
        stale.assert_not_called()
        fresh.assert_not_called()

    def test_native_cancellation_runs_once_outside_jobs_lock(self):
        message_id = b"cancel-once"
        started = threading.Event()
        release = threading.Event()
        acquired = threading.Event()
        results = []
        cancel_calls = []

        def generator(_workblock, _cost, token):
            started.set()
            release.wait(2)
            return None

        def cancel(token):
            cancel_calls.append(token.message_id)
            def acquire():
                with stamper.active_jobs_lock:
                    acquired.set()
            observer = threading.Thread(target=acquire)
            observer.start()
            self.assertTrue(acquired.wait(1))
            observer.join(1)

        stamper.set_external_generator(generator, cancel)
        worker = threading.Thread(target=lambda: results.append(
            stamper.generate_stamp(message_id, 0, no_expand=True)
        ))
        worker.start()
        self.assertTrue(started.wait(1))
        try:
            stamper.cancel_work(message_id)
            stamper.cancel_work(message_id)
            stamper.set_external_generator(None)
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual([message_id], cancel_calls)
        self.assertEqual([(None, 0)], results)

    def test_lxmf_delivery_exposes_receiving_metadata(self):
        delivered = []
        message = types.SimpleNamespace(
            source_blackholed=False,
            ratchet_id=None,
            signature_validated=False,
            fields={},
            destination_hash=b"destination",
            source_hash=b"source",
            hash=b"message",
        )
        router = object.__new__(router_module.LXMRouter)
        router.delivery_destinations = {
            message.destination_hash: types.SimpleNamespace(stamp_cost=None)
        }
        router.store = None
        router.ignored_list = []
        router.delivered_transient_ids_lock = threading.Lock()
        router.locally_delivered_transient_ids = {}
        router.has_message = lambda _: False
        router._LXMRouter__delivery_callback = delivered.append

        with mock.patch.object(router_module.LXMessage, "unpack_from_bytes", return_value=message):
            accepted = router.lxmf_delivery(
                b"encoded-message",
                method=router_module.LXMessage.OPPORTUNISTIC,
                receiving_interface="AutoInterface[peer]",
                receiving_hops=3,
            )

        self.assertTrue(accepted)
        self.assertEqual("AutoInterface[peer]", message.receiving_interface)
        self.assertEqual(3, message.receiving_hops)
        self.assertEqual([message], delivered)

    def test_opportunistic_packet_captures_metadata_before_async_handoff(self):
        DeferredThread.instances = []
        delivered = []
        destination_hash = b"d" * 16
        source_hash = b"s" * 16
        payload = RNS.vendor.umsgpack.packb([1234.5, b"title", b"message content", {}])
        wire_data = source_hash + bytes(64) + payload
        router = object.__new__(router_module.LXMRouter)
        router.store = None
        router.delivery_destinations = {
            destination_hash: types.SimpleNamespace(stamp_cost=None)
        }
        router.ignored_list = []
        router.delivered_transient_ids_lock = threading.Lock()
        router.locally_delivered_transient_ids = {}
        router.has_message = lambda _: False
        router.register_delivery_callback(delivered.append)
        packet = types.SimpleNamespace(
            destination_type=RNS.Destination.SINGLE,
            destination=types.SimpleNamespace(hash=destination_hash),
            rssi=-72,
            snr=7.5,
            q=81,
            packet_hash=b"packet-hash",
            ratchet_id=b"ratchet-id",
            receiving_interface="TCPInterface[peer]",
            hops=4,
            prove=lambda: None,
        )

        with mock.patch.object(router_module.threading, "Thread", DeferredThread):
            router.delivery_packet(wire_data, packet)

        packet.receiving_interface = "mutated-interface"
        packet.hops = 99
        DeferredThread.instances[0].run()

        self.assertEqual(1, len(delivered))
        message = delivered[0]
        self.assertIsInstance(message, router_module.LXMessage)
        self.assertEqual(b"message content", message.content)
        self.assertEqual(destination_hash, message.destination_hash)
        self.assertEqual(source_hash, message.source_hash)
        self.assertEqual("TCPInterface[peer]", message.receiving_interface)
        self.assertEqual(4, message.receiving_hops)
        self.assertEqual(-72, message.rssi)
        self.assertEqual(7.5, message.snr)
        self.assertEqual(81, message.q)
        self.assertEqual(router_module.LXMessage.OPPORTUNISTIC, message.method)



if __name__ == "__main__":
    unittest.main()
