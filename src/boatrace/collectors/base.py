"""コレクター基底."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import httpx

from boatrace.config import get_settings
from boatrace.db.models import FetchLog
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)


class CollectorError(Exception):
    pass


class BaseCollector(ABC):
    source_name: str = "base"

    def __init__(self) -> None:
        settings = get_settings()
        self.timeout = settings.collect.timeout_sec
        self.max_retries = settings.collect.max_retries
        self.backoff = settings.collect.retry_backoff_sec
        self.interval = settings.collect.request_interval_sec
        self.headers = {"User-Agent": settings.collect.user_agent}
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request_at = time.monotonic()

    def get_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True)

    def fetch_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        attempts = 0
        err: Exception | None = None
        while attempts < self.max_retries:
            attempts += 1
            try:
                self._throttle()
                with self.get_client() as client:
                    resp = client.get(url, params=params)
                    # 4xx は再試行しても無意味
                    if 400 <= resp.status_code < 500:
                        self._log_fetch(
                            url, False, attempts, f"HTTP {resp.status_code}", None
                        )
                        raise CollectorError(f"HTTP {resp.status_code} for {url}")
                    if resp.status_code >= 500:
                        raise CollectorError(f"HTTP {resp.status_code} for {url}")
                    resp.raise_for_status()
                    content = resp.text
                self._log_fetch(url, True, attempts, None, {"bytes": len(content)})
                return content
            except CollectorError as e:
                if "HTTP 4" in str(e):
                    raise
                err = e
                logger.warning("fetch_failed", url=url, attempt=attempts, error=str(e))
                if attempts < self.max_retries:
                    time.sleep(self.backoff * attempts)
            except Exception as e:  # noqa: BLE001
                err = e
                logger.warning("fetch_failed", url=url, attempt=attempts, error=str(e))
                if attempts < self.max_retries:
                    time.sleep(self.backoff * attempts)
        self._log_fetch(url, False, attempts, str(err) if err else "unknown", None)
        raise CollectorError(f"Failed to fetch {url}: {err}")

    def _log_fetch(
        self,
        target_key: str,
        success: bool,
        attempts: int,
        error_message: str | None,
        summary: dict[str, Any] | None,
    ) -> None:
        try:
            with session_scope() as session:
                session.add(
                    FetchLog(
                        source=self.source_name,
                        target_key=target_key[:128],
                        success=success,
                        attempts=attempts,
                        error_message=error_message,
                        payload_summary=summary,
                        fetched_at=datetime.utcnow(),
                    )
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_log_write_failed", error=str(e))

    @abstractmethod
    def collect(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
