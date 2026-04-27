"""Helpers for exposing Mammotion streams through a shared go2rtc instance."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from http import HTTPStatus
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from aiohttp import ClientError, ClientSession, ClientTimeout

from homeassistant.components.go2rtc.const import HA_MANAGED_URL
from homeassistant.components.http.auth import async_sign_path
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.network import NoURLAvailableError, get_url

from .const import DOMAIN, LOGGER

_GO2RTC_DOMAIN = "go2rtc"
_HA_MANAGED_URL_ALIASES = {
    HA_MANAGED_URL,
    "http://127.0.0.1:11984/",
}
_SIGN_EXPIRATION = timedelta(days=365)
_GO2RTC_API_PATH = "api/streams"
_REQUEST_TIMEOUT = ClientTimeout(total=10)
_INFO_TIMEOUT = ClientTimeout(total=5)


class MammotionGo2RTCStreamManager:
    """Manage fixed go2rtc streams per Mammotion device."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize shared stream bookkeeping for this Home Assistant instance."""
        self.hass = hass
        self._locks: dict[str, asyncio.Lock] = {}
        self._server_info: dict[str, dict] = {}

    @property
    def _session(self) -> ClientSession:
        return async_get_clientsession(self.hass)

    @property
    def _go2rtc_data(self):
        return self.hass.data.get(_GO2RTC_DOMAIN)

    @property
    def _configured_url(self) -> str | None:
        go2rtc_data = self.hass.data.get(_GO2RTC_DOMAIN)
        return getattr(go2rtc_data, "url", go2rtc_data)

    def stream_name(self, device_name: str) -> str:
        """Return the deterministic go2rtc stream name for one device."""
        # Sanitize device_name for use as stream name
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in device_name)
        return f"mammotion_{safe}"

    def configured_url(self) -> str | None:
        """Return the preferred go2rtc API base URL."""
        url = self._configured_url
        return self._normalize_url(url) if url is not None else None

    def is_available(self) -> bool:
        """Return whether a shared go2rtc instance is configured."""
        return self.configured_url() is not None

    def api_session(self) -> ClientSession:
        """Return the aiohttp session that can reach the configured go2rtc API."""
        base_url = self.configured_url()
        if base_url is None:
            return self._session
        return self._session_for_base_url(base_url)

    async def rtsp_url(self, camera) -> str | None:
        """Return the RTSP URL exposed by the shared go2rtc stream."""
        device_name = self._device_name(camera)
        if device_name is None:
            return None

        base_url = self.configured_url()
        if base_url is None:
            return None

        rtsp_base = await self._rtsp_base_url(base_url)
        if rtsp_base is None:
            return None
        return f"{rtsp_base}/{self.stream_name(device_name)}"

    def internal_webrtc_source(self, camera) -> str | None:
        """Return the signed HA WHEP source URL consumed by the shared go2rtc."""
        device_name = self._device_name(camera)
        if device_name is None:
            return None

        source_base_url = self._ha_source_base_url()
        if source_base_url is None:
            return None

        signed_path = async_sign_path(
            self.hass,
            f"/api/mammotion/whep_upstream/{device_name}",
            _SIGN_EXPIRATION,
        )
        return f"webrtc:{source_base_url}{signed_path}"

    async def async_ensure_stream(
        self,
        camera,
        *,
        raise_on_failure: bool = False,
    ) -> str | None:
        """Ensure the shared go2rtc stream exists and return its RTSP URL."""
        device_name = self._device_name(camera)
        if device_name is None:
            return None

        base_url = self.configured_url()
        if base_url is None:
            return None

        source = self.internal_webrtc_source(camera)
        if source is None:
            message = (
                f"Mammotion go2rtc stream {device_name} unavailable:"
                " HA HTTP server not reachable from go2rtc"
            )
            LOGGER.debug(message)
            if raise_on_failure:
                raise RuntimeError(message)
            return None

        stream_name = self.stream_name(device_name)
        lock = self._locks.setdefault(stream_name, asyncio.Lock())
        async with lock:
            if await self._async_stream_matches(base_url, stream_name, source):
                return await self.rtsp_url(camera)

            methods: tuple[tuple[str, dict[str, str]], ...] = (
                ("post", {"dst": stream_name, "src": source}),
                ("put", {"name": stream_name, "src": source}),
                ("patch", {"name": stream_name, "src": source}),
                ("patch", {"dst": stream_name, "src": source}),
            )

            statuses: list[str] = []
            for method, params in methods:
                status, detail = await self._async_call_api(base_url, method, params)
                detail_suffix = f" ({detail})" if detail else ""
                statuses.append(f"{method.upper()}={status}{detail_suffix}")
                if status in (HTTPStatus.OK, HTTPStatus.CREATED, HTTPStatus.NO_CONTENT):
                    return await self.rtsp_url(camera)
                if await self._async_stream_matches(base_url, stream_name, source):
                    return await self.rtsp_url(camera)

            LOGGER.warning(
                "Failed to register Mammotion go2rtc stream %s (%s)",
                device_name,
                ", ".join(statuses),
            )
            if raise_on_failure:
                raise RuntimeError(
                    f"Failed to register Mammotion go2rtc stream {device_name}"
                    f" ({', '.join(statuses)})"
                )
            return None

    async def async_remove_stream(self, camera) -> bool:
        """Remove the shared go2rtc stream if it exists."""
        device_name = self._device_name(camera)
        if device_name is None:
            return False

        base_url = self.configured_url()
        if base_url is None:
            return False

        stream_name = self.stream_name(device_name)
        lock = self._locks.setdefault(stream_name, asyncio.Lock())
        async with lock:
            for params in ({"dst": stream_name}, {"name": stream_name}):
                status, _ = await self._async_call_api(base_url, "delete", params)
                if status in (HTTPStatus.OK, HTTPStatus.NO_CONTENT):
                    return True
                if status == HTTPStatus.NOT_FOUND:
                    continue
        return False

    def api_base_url(self) -> str | None:
        """Return the normalized go2rtc API base URL."""
        return self.configured_url()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _async_stream_matches(
        self,
        base_url: str,
        stream_name: str,
        source: str,
    ) -> bool:
        streams = await self._async_get_streams(base_url)
        if streams is None:
            return False
        stream = streams.get(stream_name)
        if not isinstance(stream, dict):
            return False
        producers = stream.get("producers") or []
        normalized_source = self._normalize_source_url(source)
        return any(
            isinstance(producer, dict)
            and self._normalize_source_url(str(producer.get("url", "")))
            == normalized_source
            for producer in producers
        )

    async def _async_get_streams(self, base_url: str) -> dict[str, dict] | None:
        session = self._session_for_base_url(base_url)
        try:
            async with session.get(
                urljoin(base_url, _GO2RTC_API_PATH),
                timeout=_REQUEST_TIMEOUT,
            ) as response:
                if response.status != HTTPStatus.OK:
                    return None
                payload = await response.json()
        except (ClientError, TimeoutError, ValueError) as err:
            LOGGER.debug("Failed to query go2rtc streams: %s", err)
            return None

        if not isinstance(payload, dict):
            return None
        return payload

    async def _async_call_api(
        self, base_url: str, method: str, params: dict[str, str]
    ) -> tuple[int, str | None]:
        request: Callable[..., object] = getattr(
            self._session_for_base_url(base_url), method
        )
        try:
            async with request(
                urljoin(base_url, _GO2RTC_API_PATH),
                params=params,
                timeout=_REQUEST_TIMEOUT,
            ) as response:
                await response.read()
                return response.status, None
        except (ClientError, TimeoutError) as err:
            LOGGER.debug("go2rtc %s failed for %s: %s", method.upper(), params, err)
            return 0, str(err)

    async def _rtsp_base_url(self, base_url: str) -> str | None:
        server_info = await self._async_get_server_info(base_url)
        if server_info is None:
            return None

        api_parts = urlsplit(base_url)
        host = api_parts.hostname
        if not host:
            return None

        rtsp_config = server_info.get("rtsp") or {}
        listen_value = str(rtsp_config.get("listen", ":8554"))
        if listen_value.startswith(":"):
            return f"rtsp://{host}{listen_value}"

        listen_parts = urlsplit(f"rtsp://{listen_value.lstrip('/')}")
        rtsp_host = listen_parts.hostname or host
        rtsp_port = listen_parts.port or 8554
        return f"rtsp://{rtsp_host}:{rtsp_port}"

    async def _async_get_server_info(self, base_url: str) -> dict | None:
        if base_url in self._server_info:
            return self._server_info[base_url]

        session = self._session_for_base_url(base_url)
        try:
            async with session.get(
                urljoin(base_url, "api"),
                timeout=_INFO_TIMEOUT,
            ) as response:
                if response.status != HTTPStatus.OK:
                    return None
                payload = await response.json()
        except (ClientError, TimeoutError, ValueError) as err:
            LOGGER.debug("Failed to query go2rtc server info from %s: %s", base_url, err)
            return None

        if not isinstance(payload, dict):
            return None
        self._server_info[base_url] = payload
        return payload

    def _ha_source_base_url(self) -> str | None:
        for prefer_external in (False, True):
            try:
                url = get_url(self.hass, prefer_external=prefer_external)
            except NoURLAvailableError:
                continue
            if url:
                return url.rstrip("/")
        return None

    def _device_name(self, target) -> str | None:
        """Resolve the device_name from a camera entity or a raw string."""
        if isinstance(target, str):
            return target
        if hasattr(target, "coordinator") and hasattr(target.coordinator, "device_name"):
            return target.coordinator.device_name
        return None

    def _session_for_base_url(self, base_url: str) -> ClientSession:
        go2rtc_data = self._go2rtc_data
        configured_url = getattr(go2rtc_data, "url", go2rtc_data)
        configured_session = getattr(go2rtc_data, "session", None)
        if (
            configured_session is not None
            and isinstance(configured_url, str)
            and self._normalize_url(configured_url) == self._normalize_url(base_url)
        ):
            return configured_session
        return self._session

    @staticmethod
    def _normalize_url(url: str) -> str:
        return url.rstrip("/") + "/"

    @staticmethod
    def _normalize_source_url(source: str) -> str:
        if not source:
            return source

        raw_url = source
        if ":" in source:
            prefix, remainder = source.split(":", 1)
            if urlsplit(remainder).scheme in {"http", "https"}:
                raw_url = remainder
                if prefix != "webrtc":
                    raw_url = f"{prefix}:{remainder}"

        parts = urlsplit(raw_url)
        if not parts.scheme:
            return source.rstrip("/")

        filtered_query = urlencode(
            [
                (key, value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if key != "authSig"
            ],
            doseq=True,
        )
        normalized = urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path.rstrip("/"),
                filtered_query,
                "",
            )
        )
        return normalized.rstrip("/")


def get_go2rtc_stream_manager(hass: HomeAssistant) -> MammotionGo2RTCStreamManager:
    """Return the shared HA-managed go2rtc helper."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("go2rtc_stream_manager")
    if manager is None:
        manager = MammotionGo2RTCStreamManager(hass)
        domain_data["go2rtc_stream_manager"] = manager
    return manager
