import importlib
import threading
import types
import unittest
from unittest import mock

import RNS

router_module = importlib.import_module("LXMF.LXMRouter")


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



class ReceivingMetadataTests(unittest.TestCase):
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
