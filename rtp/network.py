"""纯内存网络模拟器：虚拟时钟 + 事件队列，无真实 socket、无外部连接。

支持按方向独立配置：
    loss_rate    丢包概率
    dup_rate     重复包概率（额外投递一份延迟副本）
    corrupt_rate 比特损坏概率（翻转负载中一个字节，触发 CRC 校验失败）
    min_delay / max_delay   基础单向延迟（虚拟时间单位）
    reorder_rate 乱序概率（命中时追加随机额外延迟，从而打乱到达顺序）
所有随机性由种子控制，测试完全确定、可复现。
"""

from __future__ import annotations

import heapq
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

DeliverFn = Callable[[bytes, float], None]


@dataclass
class ChannelConfig:
    loss_rate: float = 0.0
    dup_rate: float = 0.0
    corrupt_rate: float = 0.0
    min_delay: float = 1.0
    max_delay: float = 5.0
    reorder_rate: float = 0.0
    reorder_extra: float = 20.0  # 乱序时追加的最大额外延迟


@dataclass
class SimulatedNetwork:
    seed: int = 0
    channels: dict = field(default_factory=dict)

    def __post_init__(self):
        self.now = 0.0
        self.rng = random.Random(self.seed)
        self._events: list = []  # (time, counter, payload, deliver_fn)
        self._counter = 0
        self.stats = Counter()

    def add_channel(self, name: str, config: ChannelConfig | None = None, **kwargs):
        self.channels[name] = config if config is not None else ChannelConfig(**kwargs)

    def send(self, channel: str, payload: bytes, deliver: DeliverFn):
        cfg = self.channels[channel]
        self.stats[f"{channel}.sent"] += 1
        if self.rng.random() < cfg.loss_rate:
            self.stats[f"{channel}.dropped"] += 1
            return
        copies = 1
        if self.rng.random() < cfg.dup_rate:
            copies += 1
            self.stats[f"{channel}.duplicated"] += 1
        for i in range(copies):
            data = payload
            if self.rng.random() < cfg.corrupt_rate and len(data) > 0:
                data = self._corrupt(data)
                self.stats[f"{channel}.corrupted"] += 1
            delay = self.rng.uniform(cfg.min_delay, cfg.max_delay)
            if self.rng.random() < cfg.reorder_rate:
                delay += self.rng.uniform(0, cfg.reorder_extra)
                self.stats[f"{channel}.reordered"] += 1
            if i > 0:
                delay += self.rng.uniform(0, cfg.max_delay)  # 副本再错开一点
            self._schedule(self.now + delay, data, deliver)

    def _corrupt(self, data: bytes) -> bytes:
        buf = bytearray(data)
        idx = self.rng.randrange(len(buf))
        buf[idx] ^= 1 << self.rng.randrange(8)
        return bytes(buf)

    def _schedule(self, time: float, payload: bytes, deliver: DeliverFn):
        self._counter += 1
        heapq.heappush(self._events, (time, self._counter, payload, deliver))

    def next_event_time(self):
        return self._events[0][0] if self._events else None

    def advance_to(self, target: float):
        """推进虚拟时钟到 target，并投递所有到期的包。"""
        while self._events and self._events[0][0] <= target:
            time, _, payload, deliver = heapq.heappop(self._events)
            self.now = max(self.now, time)
            self.stats["delivered"] += 1
            deliver(payload, self.now)
        self.now = max(self.now, target)
