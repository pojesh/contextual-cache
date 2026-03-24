"""
Write-Ahead Log (WAL) for crash recovery.

Every cache mutation (store, remove, evict, threshold update) is logged
to an append-only file BEFORE being applied. On startup, uncommitted
operations are replayed to restore consistent state.

Record format: [4B length][8B LSN][1B op_type][payload][4B CRC32]
"""

from __future__ import annotations

import logging
import os
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import AsyncIterator, Optional

from .config import settings

logger = logging.getLogger(__name__)


class WALOperation(IntEnum):
    """Types of operations logged to the WAL."""
    STORE = 1
    REMOVE = 2
    EVICT = 3
    UPDATE_THRESHOLD = 4


@dataclass(slots=True)
class WALEntry:
    """A single WAL record."""
    lsn: int                   # Log Sequence Number
    operation: WALOperation
    payload: bytes
    crc: int


# Header format: 4B length (of payload + op_type) | 8B LSN | 1B op_type
_HEADER_FMT = "!IQB"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_CRC_FMT = "!I"
_CRC_SIZE = struct.calcsize(_CRC_FMT)


class WriteAheadLog:
    """
    Append-only WAL with CRC32 integrity checks.

    Supports:
      append()     — write a new operation
      replay()     — iterate over all records (for crash recovery)
      checkpoint() — truncate committed records up to a given LSN
      close()      — flush and close the file
    """

    def __init__(self, path: str = settings.wal_path
                 if hasattr(settings, "wal_path")
                 else "contextual_cache.wal") -> None:
        self._path = path
        self._file: Optional[object] = None
        self._lsn: int = 0
        self._checkpoint_lsn: int = 0

    async def initialize(self) -> None:
        """Open or create the WAL file and recover the current LSN."""
        self._file = open(self._path, "ab+")  # noqa: SIM115
        # Scan to find the highest LSN
        self._lsn = 0
        for entry in self._replay_sync():
            self._lsn = max(self._lsn, entry.lsn)
        logger.info("WAL initialized at %s (current LSN=%d).", self._path,
                     self._lsn)

    async def append(self, operation: WALOperation,
                     payload: bytes) -> int:
        """
        Append a new operation to the WAL.

        Returns the assigned LSN.
        """
        if self._file is None:
            await self.initialize()

        self._lsn += 1
        lsn = self._lsn

        # Build record
        body = struct.pack("B", operation) + payload
        header = struct.pack(_HEADER_FMT, len(body), lsn, operation)
        crc = zlib.crc32(header + payload)
        record = header + payload + struct.pack(_CRC_FMT, crc)

        self._file.write(record)  # type: ignore
        self._file.flush()  # type: ignore
        os.fsync(self._file.fileno())  # type: ignore

        return lsn

    def _replay_sync(self):
        """Synchronous replay for internal use during init."""
        path = self._path
        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            while True:
                header_bytes = f.read(_HEADER_SIZE)
                if len(header_bytes) < _HEADER_SIZE:
                    break

                body_len, lsn, op_type = struct.unpack(_HEADER_FMT, header_bytes)
                # payload_len = body_len - 1 (op_type already in header)
                payload = f.read(body_len - 1)
                if len(payload) < body_len - 1:
                    logger.warning("Truncated WAL record at LSN=%d", lsn)
                    break

                crc_bytes = f.read(_CRC_SIZE)
                if len(crc_bytes) < _CRC_SIZE:
                    logger.warning("Missing CRC at LSN=%d", lsn)
                    break

                (stored_crc,) = struct.unpack(_CRC_FMT, crc_bytes)
                computed_crc = zlib.crc32(header_bytes + payload)

                if stored_crc != computed_crc:
                    logger.warning("CRC mismatch at LSN=%d — skipping.", lsn)
                    continue

                yield WALEntry(
                    lsn=lsn,
                    operation=WALOperation(op_type),
                    payload=payload,
                    crc=stored_crc,
                )

    async def replay(self):
        """Async iterator over all valid WAL records."""
        for entry in self._replay_sync():
            if entry.lsn > self._checkpoint_lsn:
                yield entry

    async def checkpoint(self, lsn: int) -> None:
        """
        Mark all records up to `lsn` as committed.
        On next replay, these will be skipped. The file is truncated
        to reclaim space.
        """
        self._checkpoint_lsn = lsn
        # Rewrite only uncommitted records
        remaining = []
        for entry in self._replay_sync():
            if entry.lsn > lsn:
                remaining.append(entry)

        if self._file is not None:
            self._file.close()  # type: ignore

        # Rewrite file with only remaining records
        with open(self._path, "wb") as f:
            for entry in remaining:
                body = struct.pack("B", entry.operation) + entry.payload
                header = struct.pack(_HEADER_FMT, len(body), entry.lsn,
                                     entry.operation)
                crc = zlib.crc32(header + entry.payload)
                f.write(header + entry.payload + struct.pack(_CRC_FMT, crc))

        self._file = open(self._path, "ab+")  # noqa: SIM115
        logger.info("WAL checkpointed up to LSN=%d (%d records remaining).",
                     lsn, len(remaining))

    async def close(self) -> None:
        """Flush and close the WAL file."""
        if self._file is not None:
            self._file.close()  # type: ignore
            self._file = None

    @property
    def current_lsn(self) -> int:
        return self._lsn
