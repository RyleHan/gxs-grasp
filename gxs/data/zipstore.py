"""Random access to members of (possibly split) ZIP archives, local or over HTTP.

Grasp-Anything ships its 65 GB image archive as `image_part_aa` + `image_part_ab`
(a byte-level split of one zip64 file). We only need a few tens of thousands of
its ~1M images, so we read the central directory once, cache it, and then fetch
individual members by byte offset (HTTP range requests for remote files).
"""
import io
import os
import pickle
import struct
import threading
import zipfile
import zlib

import requests

CHUNK = 4 << 20


class SplitFile(io.RawIOBase):
    """Seekable read-only view over several parts concatenated in order."""

    def __init__(self, parts):
        self.parts = list(parts)
        self.sizes = [self._part_size(i) for i in range(len(self.parts))]
        self.starts, off = [], 0
        for s in self.sizes:
            self.starts.append(off)
            off += s
        self.total = off
        self.pos = 0
        self._cache = {}

    # subclasses provide these two
    def _part_size(self, i):
        raise NotImplementedError

    def _read_part(self, i, offset, n):
        raise NotImplementedError

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.pos, os.SEEK_END: self.total}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        data = self.pread(self.pos, n)
        self.pos += len(data)
        return data

    def raw(self, start, end):
        """Bytes [start, end), crossing part boundaries if needed."""
        out = bytearray()
        while start < end:
            i = max(j for j in range(len(self.parts)) if self.starts[j] <= start)
            local = start - self.starts[i]
            take = min(end - start, self.sizes[i] - local)
            out += self._read_part(i, local, take)
            start += take
        return bytes(out)

    def pread(self, offset, length):
        """Cached read, used while zipfile scans the central directory."""
        length = max(0, min(length, self.total - offset))
        if length == 0:
            return b""
        if length > CHUNK:
            return self.raw(offset, offset + length)
        first, last = offset // CHUNK, (offset + length - 1) // CHUNK
        missing = [c for c in range(first, last + 1) if c not in self._cache]
        if missing:
            lo, hi = min(missing) * CHUNK, min((max(missing) + 1) * CHUNK, self.total)
            blob = self.raw(lo, hi)
            for c in range(min(missing), max(missing) + 1):
                self._cache[c] = blob[c * CHUNK - lo:(c + 1) * CHUNK - lo]
            if len(self._cache) > 64:
                for c in list(self._cache)[:-32]:
                    del self._cache[c]
        buf = b"".join(self._cache[c] for c in range(first, last + 1))
        s = offset - first * CHUNK
        return buf[s:s + length]


class LocalSplitFile(SplitFile):
    def __init__(self, paths):
        self._fds = [os.open(p, os.O_RDONLY) for p in paths]
        super().__init__(paths)

    def _part_size(self, i):
        return os.fstat(self._fds[i]).st_size

    def _read_part(self, i, offset, n):
        return os.pread(self._fds[i], n, offset)


class RemoteSplitFile(SplitFile):
    def __init__(self, urls, session=None):
        self.session = session or requests.Session()
        self.origins = list(urls)
        self.resolved = [None] * len(urls)
        super().__init__(urls)

    def _resolve(self, i):
        r = self.session.head(self.origins[i], allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.resolved[i] = r.url                      # signed CDN link: one hop per request
        return int(r.headers["content-length"])

    def _part_size(self, i):
        return self._resolve(i)

    def _read_part(self, i, offset, n):
        hdr = {"Range": f"bytes={offset}-{offset + n - 1}"}
        for attempt in range(4):
            try:
                r = self.session.get(self.resolved[i], headers=hdr, timeout=300)
                if r.status_code in (400, 401, 403, 404):   # expired signature
                    self._resolve(i)
                    continue
                r.raise_for_status()
                if len(r.content) == n:
                    return r.content
            except requests.RequestException:
                if attempt == 3:
                    raise
        raise IOError(f"range read failed: part {i} @ {offset}+{n}")


def open_split(parts):
    remote = str(parts[0]).startswith("http")
    return RemoteSplitFile(parts) if remote else LocalSplitFile(parts)


def build_index(parts, cache_path):
    """name -> (header_offset, compress_size, compress_type); cached as a pickle."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    zf = zipfile.ZipFile(open_split(parts))
    index = {i.filename: (i.header_offset, i.compress_size, i.compress_type)
             for i in zf.infolist() if not i.is_dir()}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f, protocol=4)
    return index


def extract_member(fp, entry):
    """Decompressed bytes of one member: header + payload in a single read."""
    header_offset, compress_size, compress_type = entry
    buf = fp.raw(header_offset, min(header_offset + 30 + 1024 + compress_size, fp.total))
    if buf[:4] != b"PK\x03\x04":
        raise ValueError(f"no local header at {header_offset}")
    namelen, extralen = struct.unpack_from("<HH", buf, 26)
    start = 30 + namelen + extralen
    if start + compress_size > len(buf):
        raw = fp.raw(header_offset + start, header_offset + start + compress_size)
    else:
        raw = buf[start:start + compress_size]
    return raw if compress_type == 0 else zlib.decompressobj(-15).decompress(raw)


class Archive:
    """A zip (local or remote) plus its cached index; safe to share across threads."""

    def __init__(self, parts, cache_path):
        self.parts = list(parts)
        self.index = build_index(self.parts, cache_path)
        self._local = threading.local()

    @property
    def fp(self):
        if not hasattr(self._local, "fp"):
            self._local.fp = open_split(self.parts)
        return self._local.fp

    def get(self, name):
        return extract_member(self.fp, self.index[name])
