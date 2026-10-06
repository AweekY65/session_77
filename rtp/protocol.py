"""可靠传输协议：滑动窗口 + 累积 ACK + 超时重传 + 去重/乱序缓存。

发送端：
    - 字节流按 chunk_size 切分为定长上限的 DATA 包，最后一个包带 FIN 标志。
    - 滑动窗口允许最多 window 个未确认包同时在途。
    - 每个在途包独立计时，超时未收到累积 ACK 则重传。
    - 单包重传次数超过 max_retries 时连接进入 error 状态，绝不无限等待。
接收端：
    - 维护期望序号 expected，乱序包进入缓存，重复包直接丢弃。
    - 只有连续片段才向上层（delivered 缓冲）交付，缺口补齐前不交付后续数据。
    - 每收到一个 DATA 包都回复累积 ACK（下一个期望序号）。
"""

from __future__ import annotations

from .network import SimulatedNetwork
from .packet import FLAG_ACK, FLAG_DATA, FLAG_FIN, Packet


class ConnectionError_(Exception):
    """连接进入永久错误状态（重传超过上限）。"""


class Sender:
    def __init__(self, data: bytes, chunk_size: int = 512, window: int = 8,
                 timeout: float = 30.0, max_retries: int = 10):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]
        self.total = len(self.chunks)
        self.window = window
        self.timeout = timeout
        self.max_retries = max_retries

        self.send_base = 0       # 窗口左沿：最小未确认序号
        self.next_seq = 0        # 下一个待发送序号
        self.unacked: dict[int, list] = {}  # seq -> [packet_bytes, last_sent_time, retries]
        self.finished = False    # 所有包（含 FIN）均已被确认
        self.error: str | None = None
        self.emit = None         # 由 wire() 注入：bytes -> 网络

    def tick(self, now: float):
        if self.error or self.finished:
            return
        # 1) 超时重传
        for seq in sorted(self.unacked):
            pkt_bytes, last_sent, retries = self.unacked[seq]
            if now - last_sent >= self.timeout - 1e-9:
                if retries >= self.max_retries:
                    self.error = (f"max retries ({self.max_retries}) exceeded "
                                  f"for seq={seq}")
                    return
                self.emit(pkt_bytes)
                self.unacked[seq] = [pkt_bytes, now, retries + 1]
        # 2) 窗口允许时发送新包
        while self.next_seq - self.send_base < self.window and self.next_seq < self.total:
            seq = self.next_seq
            flags = FLAG_DATA | (FLAG_FIN if seq == self.total - 1 else 0)
            pkt_bytes = Packet(seq=seq, flags=flags, payload=self.chunks[seq]).encode()
            self.emit(pkt_bytes)
            self.unacked[seq] = [pkt_bytes, now, 0]
            self.next_seq += 1

    def on_receive(self, data: bytes, now: float):
        pkt = Packet.decode(data)
        if pkt is None or not (pkt.flags & FLAG_ACK):
            return
        ack = pkt.ack  # 累积确认：对方下一个期望的序号
        if self.send_base < ack <= self.total:
            for seq in range(self.send_base, ack):
                self.unacked.pop(seq, None)
            self.send_base = ack
        if self.send_base == self.total:
            self.finished = True

    def next_timeout(self):
        if self.error or self.finished or not self.unacked:
            return None
        return min(last_sent + self.timeout
                   for _, last_sent, _ in self.unacked.values())


class Receiver:
    def __init__(self):
        self.expected = 0                 # 下一个期望按序收到的序号
        self.buffered: dict[int, tuple] = {}  # 乱序缓存 seq -> (payload, fin)
        self.delivered = bytearray()      # 仅包含连续、无重复的字节流
        self.fin_received = False
        self.duplicates_dropped = 0
        self.emit = None

    def on_receive(self, data: bytes, now: float):
        pkt = Packet.decode(data)
        if pkt is None or not (pkt.flags & FLAG_DATA):
            return
        fin = bool(pkt.flags & FLAG_FIN)
        if pkt.seq < self.expected or pkt.seq in self.buffered:
            self.duplicates_dropped += 1  # 重复包：丢弃但仍需重发 ACK
        else:
            self.buffered[pkt.seq] = (pkt.payload, fin)
        # 只交付从 expected 开始的连续片段
        while self.expected in self.buffered:
            payload, fin = self.buffered.pop(self.expected)
            self.delivered += payload
            if fin:
                self.fin_received = True
            self.expected += 1
        self.emit(Packet(ack=self.expected, flags=FLAG_ACK).encode())


def wire(net: SimulatedNetwork, sender: Sender, receiver: Receiver,
         ch_data: str = "data", ch_ack: str = "ack"):
    """把发送端/接收端接到模拟网络的两个方向上。"""
    sender.emit = lambda b: net.send(ch_data, b, receiver.on_receive)
    receiver.emit = lambda b: net.send(ch_ack, b, sender.on_receive)


def run(net: SimulatedNetwork, sender: Sender, receiver: Receiver,
        max_steps: int = 1_000_000) -> bool:
    """驱动虚拟时钟直到传输完成或出错。返回 True 表示成功。"""
    for _ in range(max_steps):
        sender.tick(net.now)
        if sender.error:
            raise ConnectionError_(sender.error)
        if sender.finished:
            return True
        candidates = [t for t in (net.next_event_time(), sender.next_timeout())
                      if t is not None]
        if not candidates:
            return False  # 无事件可推进（理论上不会发生）
        net.advance_to(min(candidates))
    raise RuntimeError("max_steps exceeded")
