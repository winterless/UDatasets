from __future__ import annotations

import hashlib
import math
import mmap
import os
import struct
from pathlib import Path
from typing import Iterable, Optional


_HDR_MAGIC = b"UDBL"
_HDR_VERSION = 1
# 64-byte header
_HDR_STRUCT = struct.Struct("<4sI Q I Q f 32s")
_HDR_SIZE = _HDR_STRUCT.size


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _bloom_params(expected_items: int, fp_rate: float) -> tuple[int, int]:
    """
    Return (m_bits, k) for a Bloom filter.

    We round m_bits up to a power of two for fast indexing (mask instead of modulo).
    """
    n = max(1, int(expected_items))
    p = float(fp_rate)
    if not (0.0 < p < 1.0):
        p = 1e-4
    # m = -n ln(p) / (ln 2)^2
    m_ideal = int(math.ceil(-n * math.log(p) / (math.log(2.0) ** 2)))
    m_bits = _next_pow2(max(8, m_ideal))
    # k = (m/n) ln 2
    k = int(max(1, round((m_bits / n) * math.log(2.0))))
    return m_bits, k


class ExcludeBloom:
    """
    Mmap-backed Bloom filter for fast exclude-id membership tests.

    - Read-only and process-safe after build.
    - Designed to be shared across many worker processes without duplicating Python objects.
    """

    def __init__(self, *, mm: mmap.mmap, m_bits: int, k: int, count: int, fp_rate: float, path: Path):
        self._mm = mm
        self._mv = memoryview(mm)
        self.m_bits = int(m_bits)
        self.k = int(k)
        self.count = int(count)
        self.fp_rate = float(fp_rate)
        self.path = Path(path)
        self._mask = self.m_bits - 1
        self._h1h2 = struct.Struct("<QQ")

    def __len__(self) -> int:  # for logging only
        return self.count

    def close(self) -> None:
        try:
            self._mv.release()
        except Exception:
            pass
        try:
            self._mm.close()
        except Exception:
            pass

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ExcludeBloom":
        p = Path(path)
        f = p.open("rb")
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        finally:
            # keep mapping, close fd
            f.close()

        try:
            magic, ver, m_bits, k, count, fp_rate, _pad = _HDR_STRUCT.unpack_from(mm, 0)
            if magic != _HDR_MAGIC or ver != _HDR_VERSION:
                raise ValueError(f"Bad exclude index header: magic={magic!r} ver={ver!r}")
            if m_bits <= 0 or (m_bits & (m_bits - 1)) != 0:
                raise ValueError(f"Bad exclude index m_bits={m_bits} (must be power of two)")
            if k <= 0:
                raise ValueError(f"Bad exclude index k={k}")
            return cls(mm=mm, m_bits=int(m_bits), k=int(k), count=int(count), fp_rate=float(fp_rate), path=p)
        except Exception:
            try:
                mm.close()
            except Exception:
                pass
            raise

    @classmethod
    def build(
        cls,
        ids: Iterable[str],
        *,
        out_path: str | os.PathLike[str],
        expected_items: int,
        fp_rate: float = 1e-4,
        overwrite: bool = False,
    ) -> Path:
        """
        Build an exclude index from ids and write it to out_path.

        fp_rate is false-positive probability: smaller => bigger filter => fewer false drops.
        """
        outp = Path(out_path)
        if outp.exists() and not overwrite:
            return outp
        outp.parent.mkdir(parents=True, exist_ok=True)

        m_bits, k = _bloom_params(int(expected_items), float(fp_rate))
        nbytes = (m_bits + 7) // 8
        bits = bytearray(nbytes)

        h1h2 = struct.Struct("<QQ")
        count = 0

        for s in ids:
            if s is None:
                continue
            if not isinstance(s, str):
                s = str(s)
            s = s.strip()
            if not s:
                continue
            d = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=16).digest()
            h1, h2 = h1h2.unpack(d)
            if h2 == 0:
                h2 = 0x9E3779B97F4A7C15  # odd constant
            # power-of-two indexing
            x = h1
            for _i in range(k):
                idx = x & (m_bits - 1)
                bits[idx >> 3] |= 1 << (idx & 7)
                x = (x + h2) & 0xFFFFFFFFFFFFFFFF
            count += 1

        hdr = _HDR_STRUCT.pack(_HDR_MAGIC, _HDR_VERSION, int(m_bits), int(k), int(count), float(fp_rate), b"\x00" * 32)
        tmp = outp.with_name(outp.name + f".tmp.{os.getpid()}")
        with tmp.open("wb") as w:
            w.write(hdr)
            w.write(bits)
        tmp.replace(outp)
        return outp

    def __contains__(self, item: object) -> bool:
        if item is None:
            return False
        if not isinstance(item, str):
            item = str(item)
        s = item.strip()
        if not s:
            return False

        d = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=16).digest()
        h1, h2 = self._h1h2.unpack(d)
        if h2 == 0:
            h2 = 0x9E3779B97F4A7C15

        base = _HDR_SIZE
        x = h1
        for _i in range(self.k):
            idx = x & self._mask
            b = self._mv[base + (idx >> 3)]
            if (b & (1 << (idx & 7))) == 0:
                return False
            x = (x + h2) & 0xFFFFFFFFFFFFFFFF
        return True


def try_load_exclude_bloom(path: str | os.PathLike[str]) -> Optional[ExcludeBloom]:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        return ExcludeBloom.load(p)
    except Exception:
        return None

