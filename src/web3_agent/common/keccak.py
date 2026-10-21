"""keccak-256（以太坊用的那版，不是 NIST SHA3-256）。

EIP-55 地址校验和必须用 keccak-256，而 hashlib 只有 sha3_256——
两者填充字节不同（0x01 vs 0x06），结果不通用，不能拿来顶替。
本文件是为校验和单独补的纯 Python 实现，无第三方依赖；
等 ABI 编解码引入 eth-utils 之后，这里的实现可直接删掉换成库函数。
"""

from __future__ import annotations

_MASK = (1 << 64) - 1
_RATE = 136  # keccak-256 的 rate：1088 bit

_ROUND_CONSTANTS = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)

# 旋转偏移量，按 [x][y] 索引
_ROTATIONS = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)


def _rotl(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & _MASK if shift else value


def _keccak_f(lanes: list[list[int]]) -> None:
    for round_constant in _ROUND_CONSTANTS:
        c = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                lanes[x][y] ^= d[x]

        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl(lanes[x][y], _ROTATIONS[x][y])

        for x in range(5):
            for y in range(5):
                lanes[x][y] = b[x][y] ^ (~b[(x + 1) % 5][y] & b[(x + 2) % 5][y])

        lanes[0][0] ^= round_constant


def keccak_256(data: bytes) -> bytes:
    pad_len = _RATE - (len(data) % _RATE)
    if pad_len == 1:
        padded = data + b"\x81"
    else:
        padded = data + b"\x01" + b"\x00" * (pad_len - 2) + b"\x80"

    lanes = [[0] * 5 for _ in range(5)]
    for offset in range(0, len(padded), _RATE):
        block = padded[offset : offset + _RATE]
        for lane in range(_RATE // 8):
            lanes[lane % 5][lane // 5] ^= int.from_bytes(block[lane * 8 : lane * 8 + 8], "little")
        _keccak_f(lanes)

    return b"".join(lanes[lane % 5][lane // 5].to_bytes(8, "little") for lane in range(4))
