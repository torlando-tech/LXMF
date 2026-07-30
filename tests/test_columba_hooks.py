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
        expected_stamp = self.stamp_for_cost(b"", 1)

        def external_generator(workblock, stamp_cost):
            calls.append((workblock, stamp_cost))
            return expected_stamp, 7

        stamper.set_external_generator(external_generator)
        with (
            mock.patch.object(stamper.RNS.vendor.platformutils, "is_android", return_value=True),
            mock.patch.object(stamper, "job_android", side_effect=AssertionError("native Android workers selected")),
        ):
            stamp, value = stamper.generate_stamp(b"message-id", 1, expand_rounds=0)

        self.assertEqual(expected_stamp, stamp)
        self.assertGreaterEqual(value, 1)
        self.assertEqual([(b"", 1)], calls)

    def test_external_generator_accepts_valid_nonzero_cost_stamp(self):
        expected_stamp = self.stamp_for_cost(b"", 3)
        stamper.set_external_generator(lambda _workblock, _cost: (expected_stamp, 42))

        stamp, value = stamper.generate_stamp(b"message-id", 3, expand_rounds=0)

        self.assertEqual(expected_stamp, stamp)
        self.assertTrue(stamper.stamp_valid(stamp, 3, b""))

    def test_external_generator_accepts_canonical_exact_target_boundary(self):
        cost = 3
        boundary_digest = (1 << (256-cost)).to_bytes(32, byteorder="big")
        expected_stamp = bytes(stamper.STAMP_SIZE)
        stamper.set_external_generator(lambda _workblock, _cost: (expected_stamp, 1))

        with mock.patch.object(stamper.RNS.Identity, "full_hash", return_value=boundary_digest):
            stamp, value = stamper.generate_stamp(b"message-id", cost, expand_rounds=0)

        self.assertEqual(expected_stamp, stamp)
        self.assertEqual(cost-1, value)

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
        for result in malformed_results:
            with self.subTest(result=result):
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
            if stamper.stamp_value(b"", candidate.to_bytes(stamper.STAMP_SIZE, byteorder="big")) == 0
        )
        self.assertEqual(0, stamper.stamp_value(b"", insufficient_stamp))
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
        expected_stamp = self.stamp_for_cost(b"", 1)
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
        expected_stamp = self.stamp_for_cost(b"", 1)
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
        stale_stamp = self.stamp_for_cost(b"", 1)
        fresh_stamp = self.stamp_for_cost(b"", 2)
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
        router = object.__new__(router_module.LXMRouter)
        router.lxmf_delivery = mock.Mock(return_value=True)
        packet = types.SimpleNamespace(
            destination_type=RNS.Destination.SINGLE,
            destination=types.SimpleNamespace(hash=b"destination-hash"),
            rssi=-72,
            snr=7.5,
            q=81,
            packet_hash=b"packet-hash",
            ratchet_id=b"ratchet-id",
            receiving_interface="TCPInterface[peer]",
            hops=4,
            prove=mock.Mock(),
        )

        with (
            mock.patch.object(router_module.RNS.Reticulum, "get_instance", return_value=mock.Mock()),
            mock.patch.object(router_module.threading, "Thread", DeferredThread),
        ):
            router.delivery_packet(b"payload", packet)

        packet.receiving_interface = "mutated-interface"
        packet.hops = 99
        DeferredThread.instances[0].run()

        packet.prove.assert_called_once_with()
        router.lxmf_delivery.assert_called_once_with(
            b"destination-hashpayload",
            RNS.Destination.SINGLE,
            phy_stats={"rssi": -72, "snr": 7.5, "q": 81},
            ratchet_id=b"ratchet-id",
            method=router_module.LXMessage.OPPORTUNISTIC,
            receiving_interface="TCPInterface[peer]",
            receiving_hops=4,
        )


if __name__ == "__main__":
    unittest.main()
