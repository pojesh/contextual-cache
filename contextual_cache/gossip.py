"""
Push-pull gossip protocol for bandit posterior synchronization.

Each node periodically selects a random peer and exchanges Thompson
Sampling parameters. Merge is via FedAvg (arithmetic mean of α, β).
This provides eventual consistency of bandit priors across the cluster
without a central coordinator.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class GossipProtocol:
    """
    Push-pull gossip for distributed state synchronization.

    Protocol:
      1. Select random peer
      2. Push local state to peer
      3. Pull peer's state
      4. Merge via registered merge function

    Convergence: O(log N) rounds for N nodes (epidemic-style).
    """

    def __init__(
        self,
        node_id: str,
        peers: List[str],
        interval_s: float = 60.0,
        base_url_template: str = "http://{peer}",
    ) -> None:
        self.node_id = node_id
        self._peers = [p for p in peers if p != node_id]
        self._interval = interval_s
        self._url_template = base_url_template
        self._client = httpx.AsyncClient(timeout=10.0)
        self._task: Optional[asyncio.Task] = None

        # Registered state providers
        self._state_providers: Dict[str, Callable[[], dict]] = {}
        self._state_mergers: Dict[str, Callable[[dict], None]] = {}

        # Stats
        self.total_exchanges = 0
        self.failed_exchanges = 0
        self.last_exchange_time: Optional[float] = None

    def register_state(
        self,
        key: str,
        provider: Callable[[], dict],
        merger: Callable[[dict], None],
    ) -> None:
        """
        Register a state to be gossiped.

        provider: callable returning current state as dict
        merger: callable that merges received state into local state
        """
        self._state_providers[key] = provider
        self._state_mergers[key] = merger

    async def start(self) -> None:
        """Start the background gossip loop."""
        if not self._peers:
            logger.info("Gossip: no peers configured, running in standalone mode.")
            return
        self._task = asyncio.create_task(self._gossip_loop())
        logger.info("Gossip protocol started (interval=%.0fs, peers=%d).",
                     self._interval, len(self._peers))

    async def stop(self) -> None:
        """Stop the gossip loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()

    async def _gossip_loop(self) -> None:
        """Periodic gossip exchange."""
        while True:
            await asyncio.sleep(self._interval)
            peer = random.choice(self._peers)
            try:
                await self._exchange(peer)
            except Exception:
                self.failed_exchanges += 1
                logger.debug("Gossip exchange with %s failed.", peer,
                             exc_info=True)

    async def _exchange(self, peer: str) -> None:
        """Push-pull exchange with a single peer."""
        # Collect local state
        local_state = {
            key: provider()
            for key, provider in self._state_providers.items()
        }

        # Push to peer and pull their state
        url = f"{self._url_template.format(peer=peer)}/internal/gossip/exchange"
        resp = await self._client.post(url, json={
            "node_id": self.node_id,
            "state": local_state,
        })
        resp.raise_for_status()
        peer_data = resp.json()

        # Merge peer state
        peer_state = peer_data.get("state", {})
        for key, merger in self._state_mergers.items():
            if key in peer_state:
                merger(peer_state[key])

        self.total_exchanges += 1
        self.last_exchange_time = time.time()

    def handle_incoming_exchange(self, incoming: dict) -> dict:
        """
        Handle an incoming gossip exchange request.
        Returns local state for the peer to merge.
        """
        # Merge incoming state
        incoming_state = incoming.get("state", {})
        for key, merger in self._state_mergers.items():
            if key in incoming_state:
                merger(incoming_state[key])

        # Return local state
        return {
            "node_id": self.node_id,
            "state": {
                key: provider()
                for key, provider in self._state_providers.items()
            },
        }

    def get_stats(self) -> dict:
        return {
            "node_id": self.node_id,
            "peers": self._peers,
            "total_exchanges": self.total_exchanges,
            "failed_exchanges": self.failed_exchanges,
            "last_exchange_time": self.last_exchange_time,
            "registered_states": list(self._state_providers.keys()),
        }
