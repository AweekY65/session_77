"""Reliable sliding-window transport on top of SimNet.

Sender:
  - outgoing byte stream is chunked into MSS-sized DATA packets, each with
    its own sequence number (0, 1, 2, ...)
  - at most `window_size` unacknowledged packets may be in flight
  - every unacked packet is retransmitted after `rto` virtual time units
  - a packet is retransmitted at most `max_retries` times; beyond that the
    connection enters the ERROR state instead of waiting forever

Receiver:
  - per-packet ACKs
  - out-of-order packets are buffered; only the contiguous prefix starting
    at the expected sequence number is delivered upward, so the application
    never sees duplicated, reordered or partially-missing data
  - duplicate packets are de-duplicated but still re-ACKed (the ACK may
    have been lost)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from . import packet as pkt
from .simnet import SimNet


class ConnectionError(RuntimeError):
    pass


class _Unacked:
    __slots__ = ("data", "last_sent", "retries")

    def __init__(self, data: bytes, last_sent: float):
        self.data = data
        self.last_sent = last_sent
        self.retries = 0


class ReliableConnection:
    OPEN = "OPEN"
    ERROR = "ERROR"

    def __init__(
        self,
        net: SimNet,
        local_addr: str,
        peer_addr: str,
        window_size: int = 8,
        mss: int = 1024,
        rto: float = 10.0,
        max_retries: int = 5,
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if mss < 1 or mss > pkt.MAX_PAYLOAD:
            raise ValueError("invalid mss")
        self.net = net
        self.local_addr = local_addr
        self.peer_addr = peer_addr
        self.window_size = window_size
        self.mss = mss
        self.rto = rto
        self.max_retries = max_retries

        self.state = self.OPEN
        self.error: Optional[str] = None

        # sender side
        self._send_queue = bytearray()
        self._next_seq = 0
        self._unacked: Dict[int, _Unacked] = {}

        # receiver side
        self._expect_seq = 0
        self._recv_buffer: Dict[int, bytes] = {}
        self.received = bytearray()
        self.delivered_seqs: List[int] = []  # test hook: order of delivery

        net.add_endpoint(local_addr, self._on_wire_bytes)

    # -- application API --------------------------------------------------
    def send(self, data: bytes) -> None:
        self._ensure_open()
        self._send_queue.extend(data)
        self._pump()

    def idle(self) -> bool:
        return not self._unacked and not self._send_queue

    def next_timeout(self) -> float:
        if not self._unacked:
            return float("inf")
        return min(u.last_sent + self.rto for u in self._unacked.values())

    # -- sender internals -------------------------------------------------
    def _pump(self) -> None:
        while (
            self.state == self.OPEN
            and self._send_queue
            and len(self._unacked) < self.window_size
        ):
            chunk = bytes(self._send_queue[: self.mss])
            del self._send_queue[: self.mss]
            seq = self._next_seq
            self._next_seq += 1
            raw = pkt.encode(pkt.Packet(seq=seq, flags=pkt.FLAG_DATA, payload=chunk))
            self._unacked[seq] = _Unacked(raw, self.net.now)
            self.net.send(self.local_addr, self.peer_addr, raw)

    def _on_ack(self, seq: int) -> None:
        if self._unacked.pop(seq, None) is not None:
            self._pump()

    def tick(self, now: float) -> None:
        if self.state != self.OPEN:
            return
        for seq, entry in list(self._unacked.items()):
            if entry.last_sent + self.rto <= now:
                entry.retries += 1
                if entry.retries > self.max_retries:
                    self.state = self.ERROR
                    self.error = (
                        f"packet seq={seq} exceeded max_retries="
                        f"{self.max_retries}; link appears permanently down"
                    )
                    return
                entry.last_sent = now
                self.net.send(self.local_addr, self.peer_addr, entry.data)

    # -- receiver internals -----------------------------------------------
    def _on_wire_bytes(self, _addr: str, raw: bytes) -> None:
        if self.state != self.OPEN:
            return
        try:
            packet = pkt.decode(raw)
        except pkt.CorruptPacketError:
            return  # corrupted packet: drop silently, sender will retransmit
        if packet.is_ack:
            self._on_ack(packet.seq)
        if packet.is_data:
            self._on_data(packet)

    def _on_data(self, packet: pkt.Packet) -> None:
        seq = packet.seq
        if seq >= self._expect_seq and seq not in self._recv_buffer:
            self._recv_buffer[seq] = packet.payload
            while self._expect_seq in self._recv_buffer:
                chunk = self._recv_buffer.pop(self._expect_seq)
                self.received.extend(chunk)
                self.delivered_seqs.append(self._expect_seq)
                self._expect_seq += 1
        # Always (re-)ACK, even for duplicates/old packets: the previous
        # ACK may have been lost, and the sender relies on it.
        ack = pkt.encode(pkt.Packet(seq=seq, flags=pkt.FLAG_ACK))
        self.net.send(self.local_addr, self.peer_addr, ack)

    def _ensure_open(self) -> None:
        if self.state != self.OPEN:
            raise ConnectionError(f"connection is {self.state}: {self.error}")
