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


class TestColumbaHooks(unittest.TestCase):
    def tearDown(self):
        stamper.set_external_generator(None)

    def test_external_stamp_generator_takes_precedence(self):
        calls = []
        expected_stamp = bytes(range(stamper.STAMP_SIZE))

        def external_generator(workblock, stamp_cost):
            calls.append((workblock, stamp_cost))
            return expected_stamp, 1

        stamper.set_external_generator(external_generator)
        stamp, value = stamper.generate_stamp(b"message-id", 0, expand_rounds=0)

        self.assertEqual(expected_stamp, stamp)
        self.assertEqual(stamper.stamp_value(b"", expected_stamp), value)
        self.assertEqual([(b"", 0)], calls)

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

    def test_opportunistic_packet_forwards_metadata_through_worker(self):
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
            mock.patch.object(router_module.threading, "Thread", ImmediateThread),
        ):
            router.delivery_packet(b"payload", packet)

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
