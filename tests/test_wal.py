"""Tests for the Write-Ahead Log."""

import os
import tempfile

import pytest

from contextual_cache.wal import WALOperation, WriteAheadLog


class TestWriteAheadLog:

    @pytest.fixture
    def wal_path(self, tmp_path):
        return str(tmp_path / "test.wal")

    async def test_append_and_replay(self, wal_path):
        wal = WriteAheadLog(path=wal_path)
        await wal.initialize()

        lsn1 = await wal.append(WALOperation.STORE, b"entry-1-data")
        lsn2 = await wal.append(WALOperation.REMOVE, b"entry-2-id")

        assert lsn1 == 1
        assert lsn2 == 2
        assert wal.current_lsn == 2

        await wal.close()

        # Replay
        wal2 = WriteAheadLog(path=wal_path)
        await wal2.initialize()

        entries = []
        async for entry in wal2.replay():
            entries.append(entry)

        assert len(entries) == 2
        assert entries[0].lsn == 1
        assert entries[0].operation == WALOperation.STORE
        assert entries[1].lsn == 2
        assert entries[1].operation == WALOperation.REMOVE

        await wal2.close()

    async def test_checkpoint_truncates(self, wal_path):
        wal = WriteAheadLog(path=wal_path)
        await wal.initialize()

        for i in range(5):
            await wal.append(WALOperation.STORE, f"data-{i}".encode())

        # Checkpoint up to LSN 3
        await wal.checkpoint(3)

        # Replay should only return LSNs 4, 5
        entries = []
        async for entry in wal.replay():
            entries.append(entry)

        assert len(entries) == 2
        assert entries[0].lsn == 4
        assert entries[1].lsn == 5

        await wal.close()

    async def test_empty_wal_replay(self, wal_path):
        wal = WriteAheadLog(path=wal_path)
        await wal.initialize()

        entries = []
        async for entry in wal.replay():
            entries.append(entry)

        assert len(entries) == 0
        await wal.close()

    async def test_lsn_recovers_after_restart(self, wal_path):
        wal = WriteAheadLog(path=wal_path)
        await wal.initialize()

        await wal.append(WALOperation.STORE, b"data1")
        await wal.append(WALOperation.STORE, b"data2")
        await wal.close()

        # Reopen
        wal2 = WriteAheadLog(path=wal_path)
        await wal2.initialize()
        assert wal2.current_lsn == 2

        lsn3 = await wal2.append(WALOperation.REMOVE, b"data3")
        assert lsn3 == 3

        await wal2.close()
