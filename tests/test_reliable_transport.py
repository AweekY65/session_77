"""可靠传输协议端到端测试：全部基于纯内存模拟网络，终端执行。

运行：python3 -m unittest discover -s tests -v
"""

import hashlib
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtp import (ChannelConfig, ConnectionError_, Receiver, Sender,
                 SimulatedNetwork, run, wire)


def make_data(size: int, seed: int = 1234) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


def transfer(data: bytes, data_cfg: ChannelConfig, ack_cfg: ChannelConfig,
             seed: int = 0, **sender_kwargs):
    """搭建双向模拟网络并跑一次完整传输，返回 (sender, receiver, net)。"""
    net = SimulatedNetwork(seed=seed)
    net.add_channel("data", data_cfg)
    net.add_channel("ack", ack_cfg)
    sender = Sender(data, **sender_kwargs)
    receiver = Receiver()
    wire(net, sender, receiver)
    run(net, sender, receiver)
    return sender, receiver, net


class ReliableTransportTest(unittest.TestCase):
    def assert_hash_matches(self, data, receiver):
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         hashlib.sha256(bytes(receiver.delivered)).hexdigest())
        self.assertEqual(len(data), len(receiver.delivered))

    def test_no_loss(self):
        data = make_data(64 * 1024)
        cfg = ChannelConfig()
        sender, receiver, net = transfer(data, cfg, ChannelConfig(), seed=1)
        self.assert_hash_matches(data, receiver)
        self.assertTrue(receiver.fin_received)
        self.assertEqual(net.stats["data.dropped"], 0)

    def test_random_loss_both_directions(self):
        data = make_data(128 * 1024)
        data_cfg = ChannelConfig(loss_rate=0.2)
        ack_cfg = ChannelConfig(loss_rate=0.2)
        sender, receiver, net = transfer(data, data_cfg, ack_cfg, seed=2,
                                         timeout=30.0)
        self.assert_hash_matches(data, receiver)
        self.assertGreater(net.stats["data.dropped"], 0)
        self.assertGreater(net.stats["ack.dropped"], 0)

    def test_severe_reorder(self):
        data = make_data(64 * 1024)
        # 高乱序概率 + 大额外延迟，制造严重乱序到达
        data_cfg = ChannelConfig(reorder_rate=0.8, reorder_extra=60.0,
                                 min_delay=1.0, max_delay=2.0)
        sender, receiver, net = transfer(data, data_cfg, ChannelConfig(),
                                         seed=3, window=16, timeout=200.0)
        self.assert_hash_matches(data, receiver)
        self.assertGreater(net.stats["data.reordered"], 0)

    def test_duplicate_packets(self):
        data = make_data(64 * 1024)
        data_cfg = ChannelConfig(dup_rate=0.5)
        ack_cfg = ChannelConfig(dup_rate=0.5)
        sender, receiver, net = transfer(data, data_cfg, ack_cfg, seed=4)
        self.assert_hash_matches(data, receiver)
        self.assertGreater(net.stats["data.duplicated"], 0)
        self.assertGreater(receiver.duplicates_dropped, 0)  # 去重逻辑确实生效

    def test_ack_loss(self):
        data = make_data(64 * 1024)
        ack_cfg = ChannelConfig(loss_rate=0.4)  # 只丢 ACK
        sender, receiver, net = transfer(data, ChannelConfig(), ack_cfg,
                                         seed=5, timeout=25.0)
        self.assert_hash_matches(data, receiver)
        self.assertGreater(net.stats["ack.dropped"], 0)

    def test_corruption_detected_by_crc(self):
        data = make_data(32 * 1024)
        data_cfg = ChannelConfig(corrupt_rate=0.1)
        sender, receiver, net = transfer(data, data_cfg, ChannelConfig(),
                                         seed=6, timeout=25.0)
        self.assert_hash_matches(data, receiver)
        self.assertGreater(net.stats["data.corrupted"], 0)

    def test_max_retries_exceeded_on_permanent_loss(self):
        data = make_data(16 * 1024)
        # 数据方向 100% 丢包：连接必须在有限重试后报错，不得无限等待
        data_cfg = ChannelConfig(loss_rate=1.0)
        net = SimulatedNetwork(seed=7)
        net.add_channel("data", data_cfg)
        net.add_channel("ack", ChannelConfig())
        sender = Sender(data, chunk_size=256, window=4, timeout=10.0,
                        max_retries=5)
        receiver = Receiver()
        wire(net, sender, receiver)
        with self.assertRaises(ConnectionError_):
            run(net, sender, receiver)
        self.assertIsNotNone(sender.error)
        self.assertFalse(sender.finished)
        # 虚拟时间有界：窗口内首个包重试 5 次后即失败
        self.assertLessEqual(net.now, 10.0 * (5 + 1) + 1)

    def test_mid_connection_blackout_then_error(self):
        # 前若干包正常，随后链路永久中断 -> 必须报错而非挂起
        data = make_data(8 * 1024)
        net = SimulatedNetwork(seed=8)
        net.add_channel("data", ChannelConfig())
        net.add_channel("ack", ChannelConfig())
        sender = Sender(data, chunk_size=256, window=4, timeout=10.0,
                        max_retries=3)
        receiver = Receiver()
        wire(net, sender, receiver)

        orig_send = net.send
        state = {"count": 0}

        def flaky(channel, payload, deliver):
            state["count"] += 1
            if channel == "data" and state["count"] > 6:
                return  # 静默丢弃，模拟永久中断
            orig_send(channel, payload, deliver)

        net.send = flaky
        with self.assertRaises(ConnectionError_):
            run(net, sender, receiver)

    def test_mixed_hostile_network(self):
        # 丢包 + 重复 + 乱序 + 损坏同时存在
        data = make_data(96 * 1024)
        data_cfg = ChannelConfig(loss_rate=0.1, dup_rate=0.2, corrupt_rate=0.02,
                                 reorder_rate=0.4, reorder_extra=30.0)
        ack_cfg = ChannelConfig(loss_rate=0.1, dup_rate=0.1)
        sender, receiver, net = transfer(data, data_cfg, ack_cfg, seed=9,
                                         window=12, timeout=120.0)
        self.assert_hash_matches(data, receiver)

    def test_empty_and_small_payloads(self):
        for size in (0, 1, 100):
            data = make_data(size)
            sender, receiver, net = transfer(data, ChannelConfig(),
                                             ChannelConfig(), seed=10,
                                             chunk_size=64)
            self.assert_hash_matches(data, receiver)


if __name__ == "__main__":
    unittest.main()
