"""Packet 编解码：固定 15 字节头部 + 负载，CRC32 完整性校验。

头部布局（网络字节序）：
    seq    : uint32  数据包序号（按包计数）/ ACK 包中保留
    ack    : uint32  累积确认号（下一个期望收到的序号）
    flags  : uint8   DATA=0x01, ACK=0x02, FIN=0x04
    length : uint16  负载字节数
    crc32  : uint32  对 "crc 字段置 0 的头部 + 负载" 计算
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

FLAG_DATA = 0x01
FLAG_ACK = 0x02
FLAG_FIN = 0x04

MAX_PAYLOAD = 4096

_HEADER = struct.Struct("!IIBHI")  # seq, ack, flags, length, crc32
HEADER_SIZE = _HEADER.size


@dataclass
class Packet:
    seq: int = 0
    ack: int = 0
    flags: int = 0
    payload: bytes = b""

    def encode(self) -> bytes:
        if len(self.payload) > MAX_PAYLOAD:
            raise ValueError("payload too large")
        header = _HEADER.pack(self.seq, self.ack, self.flags, len(self.payload), 0)
        crc = zlib.crc32(header + self.payload) & 0xFFFFFFFF
        return _HEADER.pack(self.seq, self.ack, self.flags, len(self.payload), crc) + self.payload

    @staticmethod
    def decode(data: bytes) -> "Packet | None":
        """解码并校验 CRC；损坏或格式非法时返回 None（由上层丢弃）。"""
        if len(data) < HEADER_SIZE:
            return None
        seq, ack, flags, length, crc = _HEADER.unpack(data[:HEADER_SIZE])
        payload = data[HEADER_SIZE:]
        if len(payload) != length:
            return None
        zeroed = _HEADER.pack(seq, ack, flags, length, 0)
        if zlib.crc32(zeroed + payload) & 0xFFFFFFFF != crc:
            return None
        return Packet(seq=seq, ack=ack, flags=flags, payload=payload)
