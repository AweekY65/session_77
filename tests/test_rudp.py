"""Terminal-runnable tests for the reliable transport protocol.

Run with:  python3 -m unittest discover -s tests -v
"""

import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rudp import packet as pkt
from rudp.protocol import ReliableConnection
from rudp.simnet import LinkConfig, SimNet

A, B = "A", "B"


def make_pair(seed=0, window=8, mss=256, rto=10.0, max_retries=5,
              link_ab=LinkConfig(), link_ba=LinkConfig()):
    net = SimNet(seed=seed)
    net.set_link(A, B, link_ab)
    net.set_link(B, A, link_ba)
    ca = ReliableConnection(net, A, B, window_size=window, mss=mss,
                            rto=rto, max_retries=max_retries)
    cb = ReliableConnection(net, B, A, window_size=window, mss=mss,
                            rto=rto, max_retries=max_retries)
    return net, ca, cb


def transfer(testcase, data, seed=0, **kwargs):
    net, ca, cb = make_pair(seed=seed, **kwargs)
    ca.send(data)
    net.run([ca, cb])
    testcase.assertEqual(ca.state, ReliableConnection.OPEN, ca.error)
    testcase.assertEqual(cb.state, ReliableConnection.OPEN, cb.error)
    testcase.assertTrue(ca.idle())
    # in-order, gap-free delivery: delivered seqs must be 0..n-1 ascending
    testcase.assertEqual(cb.delivered_seqs, list(range(len(cb.delivered_seqs))))
    testcase.assertEqual(hashlib.sha256(bytes(cb.received)).hexdigest(),
                         hashlib.sha256(data).hexdigest())
    testcase.assertEqual(len(cb.received), len(data))
    return net, ca, cb


def payload(size=64 * 1024, seed=1234):
    import random
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


class TestPacketCodec(unittest.TestCase):
    def test_roundtrip(self):
        p = pkt.Packet(seq=42, flags=pkt.FLAG_DATA, payload=b"hello")
        self.assertEqual(pkt.decode(pkt.encode(p)), p)

    def test_corruption_detected(self):
        raw = bytearray(pkt.encode(pkt.Packet(seq=1, flags=pkt.FLAG_DATA,
                                              payload=b"abcdef")))
        raw[-1] ^= 0xFF
        with self.assertRaises(pkt.CorruptPacketError):
            pkt.decode(bytes(raw))

    def test_truncated_detected(self):
        raw = pkt.encode(pkt.Packet(seq=1, flags=pkt.FLAG_DATA, payload=b"abc"))
        with self.assertRaises(pkt.CorruptPacketError):
            pkt.decode(raw[:-2])


class TestReliableTransfer(unittest.TestCase):
    def test_no_loss(self):
        transfer(self, payload())

    def test_empty_and_small_payloads(self):
        transfer(self, b"")
        transfer(self, b"x")
        transfer(self, b"y" * 255)  # smaller than one MSS

    def test_random_loss_both_directions(self):
        link = LinkConfig(loss=0.15, delay=1.0, jitter=2.0)
        transfer(self, payload(), seed=7, link_ab=link, link_ba=link)

    def test_severe_reorder(self):
        link = LinkConfig(delay=1.0, jitter=30.0, reorder=0.4,
                          reorder_delay=50.0)
        # rto must exceed worst-case RTT (~2 * (1 + 30 + 50))
        transfer(self, payload(), seed=11, rto=200.0,
                 link_ab=link, link_ba=link)

    def test_heavy_duplication(self):
        link = LinkConfig(duplicate=0.5, delay=1.0, jitter=2.0)
        net, ca, cb = transfer(self, payload(), seed=13,
                               link_ab=link, link_ba=link)
        self.assertGreater(net.stats[(A, B)].duplicated, 0)

    def test_ack_loss(self):
        # DATA path clean, ACK path very lossy: data retransmission +
        # receiver de-dup must keep the stream exact.
        transfer(self, payload(), seed=17,
                 link_ab=LinkConfig(delay=1.0),
                 link_ba=LinkConfig(loss=0.4, delay=1.0))

    def test_corruption_recovered(self):
        link = LinkConfig(corrupt=0.2, delay=1.0, jitter=2.0)
        net, ca, cb = transfer(self, payload(), seed=19,
                               link_ab=link, link_ba=link)
        self.assertGreater(net.stats[(A, B)].corrupted, 0)

    def test_combined_adversarial(self):
        link = LinkConfig(loss=0.1, duplicate=0.2, corrupt=0.05,
                          delay=1.0, jitter=10.0, reorder=0.2,
                          reorder_delay=30.0)
        transfer(self, payload(), seed=23, rto=100.0,
                 link_ab=link, link_ba=link)

    def test_window_allows_multiple_in_flight(self):
        # window > 1 must actually be used (more than one packet in flight)
        net, ca, cb = make_pair(seed=3, window=8, mss=64,
                                rto=300.0,
                                link_ab=LinkConfig(delay=100.0))
        data = payload(size=64 * 8)
        ca.send(data)
        # long link delay: nothing ACKed yet, window fully occupied
        self.assertEqual(len(ca._unacked), 8)
        net.run([ca, cb])
        self.assertEqual(hashlib.sha256(bytes(cb.received)).hexdigest(),
                         hashlib.sha256(data).hexdigest())

    def test_permanent_loss_gives_error_not_hang(self):
        net, ca, cb = make_pair(seed=29, max_retries=3, rto=5.0,
                                link_ab=LinkConfig(loss=1.0))
        ca.send(payload(size=4096))
        net.run([ca, cb], max_time=100_000)
        self.assertEqual(ca.state, ReliableConnection.ERROR)
        self.assertIn("max_retries", ca.error)
        # bounded: 3 retries * rto per packet, far below max_time
        self.assertLess(net.now, 1_000)

    def test_bidirectional(self):
        net, ca, cb = make_pair(seed=31,
                                link_ab=LinkConfig(loss=0.1, jitter=5.0),
                                link_ba=LinkConfig(loss=0.1, jitter=5.0))
        data_ab, data_ba = payload(seed=1), payload(seed=2)
        ca.send(data_ab)
        cb.send(data_ba)
        net.run([ca, cb])
        self.assertEqual(hashlib.sha256(bytes(cb.received)).hexdigest(),
                         hashlib.sha256(data_ab).hexdigest())
        self.assertEqual(hashlib.sha256(bytes(ca.received)).hexdigest(),
                         hashlib.sha256(data_ba).hexdigest())


if __name__ == "__main__":
    unittest.main()
