"""
MarketDataClient — fetches market data from Yahoo Finance (primary)
with Alpha Vantage as fallback, and caches results in-memory with a
configurable TTL (default 15 minutes).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from app.config import settings
from app.models.internal import MarketDataSnapshot

logger = logging.getLogger(__name__)


class MarketDataClient:
    """Fetches and caches market data snapshots.

    Primary provider : Yahoo Finance (yfinance)
    Fallback provider: Alpha Vantage (simple quote endpoint)

    Cache behaviour:
    - Fresh cache (within TTL) → return cached snapshot as-is.
    - Provider failure + valid cache → return cached snapshot with is_stale=True.
    - Provider failure + no cache → return null-signal snapshot with is_stale=True.
    """

    def __init__(self) -> None:
        self._cache: Optional[MarketDataSnapshot] = None
        self._cache_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self) -> MarketDataSnapshot:
        """Return a MarketDataSnapshot, using cache when fresh."""
        if self._cache is not None and self._is_cache_fresh():
            return self._cache

        # Try primary provider
        try:
            snapshot = self._fetch_from_yfinance()
            self._update_cache(snapshot)
            return snapshot
        except Exception as primary_exc:
            logger.warning("yfinance fetch failed: %s — trying Alpha Vantage fallback", primary_exc)

        # Try fallback provider
        try:
            snapshot = self._fetch_from_alpha_vantage()
            self._update_cache(snapshot)
            return snapshot
        except Exception as fallback_exc:
            logger.warning("Alpha Vantage fallback failed: %s", fallback_exc)

        # Both providers failed — return stale cache or null snapshot
        if self._cache is not None:
            logger.warning("All providers failed; returning stale cached snapshot")
            return MarketDataSnapshot(
                fetched_at=self._cache.fetched_at,
                equity_index=self._cache.equity_index,
                interest_rate=self._cache.interest_rate,
                inflation_indicator=self._cache.inflation_indicator,
                is_stale=True,
            )

        logger.warning("All providers failed and no cache exists; returning null snapshot")
        return MarketDataSnapshot(
            fetched_at=datetime.utcnow(),
            equity_index=None,
            interest_rate=None,
            inflation_indicator=None,
            is_stale=True,
        )

    def ping(self) -> bool:
        """Return True if a market data snapshot can be obtained (even stale)."""
        try:
            self.fetch()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_cache_fresh(self) -> bool:
        if self._cache_time is None:
            return False
        ttl = timedelta(seconds=settings.market_data_cache_ttl_seconds)
        return (datetime.utcnow() - self._cache_time) < ttl

    def _update_cache(self, snapshot: MarketDataSnapshot) -> None:
        self._cache = snapshot
        self._cache_time = datetime.utcnow()

    def _fetch_from_yfinance(self) -> MarketDataSnapshot:
        """Fetch equity index from Yahoo Finance using yfinance."""
        import yfinance as yf  # lazy import — optional dependency

        ticker = yf.Ticker("^NSEI")
        info = ticker.fast_info

        equity_index: Optional[float] = None
        if hasattr(info, "last_price") and info.last_price is not None:
            equity_index = float(info.last_price)

        return MarketDataSnapshot(
            fetched_at=datetime.utcnow(),
            equity_index=equity_index,
            interest_rate=None,       # not available from this endpoint
            inflation_indicator=None, # not available from this endpoint
            is_stale=False,
        )

    def _fetch_from_alpha_vantage(self) -> MarketDataSnapshot:
        """Attempt a simple Alpha Vantage quote as fallback."""
        import urllib.request
        import json

        api_key = settings.alpha_vantage_api_key
        if not api_key:
            raise ValueError("Alpha Vantage API key not configured")

        # Use NIFTY 50 equivalent on Alpha Vantage (BSE:SENSEX or similar)
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol=NSEI.BSE&apikey={api_key}"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())

        quote = data.get("Global Quote", {})
        price_str = quote.get("05. price")
        equity_index = float(price_str) if price_str else None

        return MarketDataSnapshot(
            fetched_at=datetime.utcnow(),
            equity_index=equity_index,
            interest_rate=None,
            inflation_indicator=None,
            is_stale=False,
        )
