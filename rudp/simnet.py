"""Deterministic in-memory network simulator.

No real sockets, no internet, no external systems: packets are byte strings
moved between registered endpoints through a virtual-time event queue.

Each directed link (src -> dst) can be configured with:
  - loss probability        (packet is dropped)
  - duplicate probability   (packet is delivered twice)
  - corruption probability  (payload bits flipped; caught by CRC-32)
  - base delay + jitter     (virtual time units; jitter causes reordering)
  - reorder probability     (packet is held back for an extra long delay)

Everything is driven by a seeded random generator, so tests are fully
reproducible.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LinkConfig:
    loss: float = 0.0
    duplicate: float = 0.0
    corrupt: float = 0.0
    delay: float = 1.0
    jitter: float = 0.0
    reorder: float = 0.0
    reorder_delay: float = 20.0


@dataclass
class LinkStats:
    sent: int = 0
    dropped: int = 0
    duplicated: int = 0
    corrupted: int = 0
    delivered: int = 0


class SimNet:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self.now: float = 0.0
        self._events: List[Tuple[float, int, str, bytes]] = []
        self._counter = 0
        self._endpoints: Dict[str, Callable[[str, bytes], None]] = {}
        self._links: Dict[Tuple[str, str], LinkConfig] = {}
        self.stats: Dict[Tuple[str, str], LinkStats] = {}

    # -- topology ---------------------------------------------------------
    def add_endpoint(self, address: str, handler: Callable[[str, bytes], None]) -> None:
        self._endpoints[address] = handler

    def set_link(self, src: str, dst: str, config: LinkConfig) -> None:
        self._links[(src, dst)] = config
        self.stats.setdefault((src, dst), LinkStats())

    def _link(self, src: str, dst: str) -> LinkConfig:
        return self._links.get((src, dst), LinkConfig())

    # -- sending ----------------------------------------------------------
    def send(self, src: str, dst: str, payload: bytes) -> None:
        link = self._link(src, dst)
        st = self.stats.setdefault((src, dst), LinkStats())
        st.sent += 1
        if self._rng.random() < link.loss:
            st.dropped += 1
            return
        copies = 1
        if self._rng.random() < link.duplicate:
            copies += 1
            st.duplicated += 1
        for _ in range(copies):
            data = payload
            if self._rng.random() < link.corrupt and data:
                st.corrupted += 1
                pos = self._rng.randrange(len(data))
                data = data[:pos] + bytes([data[pos] ^ 0xFF]) + data[pos + 1:]
            delay = link.delay + self._rng.uniform(0.0, link.jitter)
            if self._rng.random() < link.reorder:
                delay += link.reorder_delay
            self._counter += 1
            heapq.heappush(self._events, (self.now + delay, self._counter, dst, data))

    # -- simulation loop --------------------------------------------------
    def _deliver_due(self) -> None:
        while self._events and self._events[0][0] <= self.now:
            _, _, dst, data = heapq.heappop(self._events)
            handler = self._endpoints.get(dst)
            if handler is not None:
                st = self.stats.setdefault(("*", dst), LinkStats())
                st.delivered += 1
                handler(dst, data)

    def run(self, connections, max_time: float = 1_000_000.0) -> None:
        """Advance virtual time until all connections are idle or one errors."""
        conns = list(connections)
        while True:
            if any(c.state == c.ERROR for c in conns):
                return
            self._deliver_due()
            for c in conns:
                c.tick(self.now)
            if not self._events and all(c.idle() for c in conns):
                return
            next_event = self._events[0][0] if self._events else float("inf")
            next_timeout = min(
                (c.next_timeout() for c in conns), default=float("inf")
            )
            nxt = min(next_event, next_timeout)
            if nxt == float("inf"):
                return
            self.now = max(self.now, nxt)
            if self.now > max_time:
                raise TimeoutError(
                    f"simulation exceeded max_time={max_time} (possible livelock)"
                )
