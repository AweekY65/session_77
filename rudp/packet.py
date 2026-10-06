"""Packet encoding/decoding with sequence number and CRC-32 integrity check.

Wire format (big-endian):

    +----------------+----------------+---------------+------------------+
    | seq  (4 bytes) | flags (1 byte) | len (2 bytes) | crc32 (4 bytes)  |
    +----------------+----------------+---------------+------------------+
    | payload (len bytes) ...                                         |
    +------------------------------------------------------------------+

The CRC-32 covers the header fields (seq, flags, len) plus the payload.
Packets whose checksum does not match are treated as corrupted and dropped.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

FLAG_DATA = 0x01
FLAG_ACK = 0x02

_HEADER_FMT = "!IBH"  # seq, flags, payload length (crc appended separately)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_CRC_FMT = "!I"
_CRC_SIZE = struct.calcsize(_CRC_FMT)
PACKET_HEADER_SIZE = _HEADER_SIZE + _CRC_SIZE

MAX_PAYLOAD = 0xFFFF


class CorruptPacketError(ValueError):
    """Raised when a packet fails its integrity check or is malformed."""


@dataclass(frozen=True)
class Packet:
    seq: int
    flags: int
    payload: bytes = b""

    @property
    def is_data(self) -> bool:
        return bool(self.flags & FLAG_DATA)

    @property
    def is_ack(self) -> bool:
        return bool(self.flags & FLAG_ACK)


def encode(packet: Packet) -> bytes:
    if len(packet.payload) > MAX_PAYLOAD:
        raise ValueError("payload too large")
    header = struct.pack(_HEADER_FMT, packet.seq, packet.flags, len(packet.payload))
    crc = zlib.crc32(header + packet.payload) & 0xFFFFFFFF
    return header + struct.pack(_CRC_FMT, crc) + packet.payload


def decode(raw: bytes) -> Packet:
    if len(raw) < PACKET_HEADER_SIZE:
        raise CorruptPacketError("packet shorter than header")
    seq, flags, length = struct.unpack(_HEADER_FMT, raw[:_HEADER_SIZE])
    (crc,) = struct.unpack(_CRC_FMT, raw[_HEADER_SIZE:PACKET_HEADER_SIZE])
    payload = raw[PACKET_HEADER_SIZE:]
    if len(payload) != length:
        raise CorruptPacketError("payload length mismatch")
    expected = zlib.crc32(raw[:_HEADER_SIZE] + payload) & 0xFFFFFFFF
    if crc != expected:
        raise CorruptPacketError("crc32 mismatch")
    return Packet(seq=seq, flags=flags, payload=payload)
