"""WHEP endpoints for Mammotion upstream ingest and shared go2rtc output.

Architecture
------------
go2rtc (pull side)
    POST /api/mammotion/whep_upstream/{device_name}
        → creates an AgoraWebSocketHandler session to the robot camera
        → returns SDP answer to go2rtc
        → go2rtc holds the WebRTC track and exposes it as RTSP + WHEP

Browser / external app (viewer side)
    POST /api/mammotion/whep_direct/{device_name}
        → ensures the upstream session exists in go2rtc
        → proxies the offer through go2rtc's internal WHEP endpoint
        → returns SDP answer to the viewer

Both endpoints accept signed HA auth tokens (authSig) and standard HA
long-lived access tokens so they can be reached from go2rtc without a
browser session.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from http import HTTPStatus
import secrets
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from sdp_transform import parse as sdp_parse
from webrtc_models import RTCIceCandidateInit

from homeassistant.auth import jwt_wrapper
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.http.auth import DATA_SIGN_SECRET, SIGN_QUERY_PARAM
from homeassistant.components.http.const import KEY_HASS_REFRESH_TOKEN_ID, KEY_HASS_USER
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .agora_websocket import AgoraWebSocketHandler
from .const import DOMAIN, LOGGER
from .go2rtc_stream import get_go2rtc_stream_manager

TOKEN_REFRESH_INTERVAL_SECONDS = 20 * 60
_GO2RTC_WHEP_PATH = "api/webrtc"
_REQUEST_TIMEOUT = ClientTimeout(total=15)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _check_external_auth(request: web.Request) -> web.Response | None:
    """Allow authenticated HA users or requests signed with authSig / access token."""
    hass = request.app["hass"]
    if request.get("hass_user"):
        return None

    if _validate_signed_request(hass, request):
        return None

    token = request.query.get("token")
    if token:
        if hass.auth.async_validate_access_token(token) is None:
            return web.Response(status=401, text="Invalid token")
        return None

    return web.Response(status=401, text="Authentication required")


def _validate_signed_request(hass, request: web.Request) -> bool:
    """Validate an authSig-signed request."""
    if (secret := hass.data.get(DATA_SIGN_SECRET)) is None:
        return False

    if (signature := request.query.get(SIGN_QUERY_PARAM)) is None:
        return False

    try:
        claims = jwt_wrapper.verify_and_decode(
            signature,
            secret,
            algorithms=["HS256"],
            options={"verify_iss": False},
        )
    except Exception:  # noqa: BLE001
        return False

    if claims.get("path") != request.path:
        return False

    params = [
        list(item) for item in request.query.items() if item[0] != SIGN_QUERY_PARAM
    ]
    if claims.get("params") != params:
        return False

    refresh_token = hass.auth.async_get_refresh_token(claims.get("iss"))
    if refresh_token is None:
        return False

    request[KEY_HASS_USER] = refresh_token.user
    request[KEY_HASS_REFRESH_TOKEN_ID] = refresh_token.id
    return True


# ---------------------------------------------------------------------------
# Upstream session management  (go2rtc → Agora)
# ---------------------------------------------------------------------------


@dataclass
class AgoraUpstreamSession:
    """One direct Agora session feeding the shared go2rtc stream."""

    session_id: str
    device_name: str
    agora_handler: AgoraWebSocketHandler
    refresh_task: asyncio.Task | None = None
    location_path: str = ""


class MammotionAgoraUpstreamManager:
    """Manage one direct Agora session per device for go2rtc ingest."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._sessions: dict[str, AgoraUpstreamSession] = {}

    async def create_session(
        self,
        camera,
        offer_sdp: str,
    ) -> tuple[str, str]:
        """Create or replace one direct Agora session for the device.

        Returns (session_id, answer_sdp).
        """
        device_name = camera.coordinator.device_name
        await self.close_session(device_name)

        # Wake the camera (tells the robot to open the video channel via IoT)
        await camera.coordinator.join_webrtc_channel()
        # Give the robot a moment to open the channel before connecting
        await asyncio.sleep(2)

        # Refresh stream token + Agora edge servers
        stream_data, agora_response = await camera.coordinator.async_check_stream_expiry()
        if stream_data is None or stream_data.data is None:
            raise RuntimeError(f"No stream data available for {device_name}")
        if agora_response is None:
            raise RuntimeError(f"Failed to retrieve Agora edge servers for {device_name}")

        agora_data = stream_data.data  # StreamSubscriptionResponse

        # Collect inline ICE candidates from the go2rtc offer SDP
        agora_handler = AgoraWebSocketHandler(self.hass)
        for line in offer_sdp.splitlines():
            stripped = line.strip()
            if stripped.startswith("a=candidate:"):
                agora_handler.add_ice_candidate(
                    RTCIceCandidateInit(candidate=stripped.removeprefix("a="))
                )

        session_id = secrets.token_hex(16)
        try:
            answer_sdp = await agora_handler.connect_and_join(
                agora_data=agora_data,
                offer_sdp=offer_sdp,
                session_id=session_id,
                agora_response=agora_response,
            )
        except Exception:
            await agora_handler.disconnect()
            raise

        if not answer_sdp:
            await agora_handler.disconnect()
            raise RuntimeError("Agora upstream negotiation did not return an SDP answer")

        def _on_gone() -> None:
            """Called if the WebSocket drops unexpectedly."""
            self.hass.async_create_task(
                self.close_session(device_name),
                f"mammotion upstream cleanup {device_name}",
            )

        session = AgoraUpstreamSession(
            session_id=session_id,
            device_name=device_name,
            agora_handler=agora_handler,
            location_path=f"/api/mammotion/whep_upstream/{device_name}/{session_id}",
        )
        session.refresh_task = self.hass.async_create_background_task(
            self._refresh_tokens(session, camera),
            f"mammotion go2rtc upstream refresh {device_name}",
        )

        async with self._lock:
            self._sessions[device_name] = session

        return session_id, answer_sdp

    async def close_session(self, device_name: str) -> bool:
        """Close one direct Agora session and tell the robot to stop streaming."""
        async with self._lock:
            session = self._sessions.pop(device_name, None)

        if session is None:
            return False

        if session.refresh_task is not None:
            session.refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.refresh_task

        await session.agora_handler.disconnect()
        return True

    async def has_session(self, device_name: str) -> bool:
        async with self._lock:
            return device_name in self._sessions

    async def add_session_candidates(
        self,
        device_name: str,
        session_id: str,
        sdp_fragment: str,
    ) -> bool:
        """Forward trickled ICE candidates for one active upstream session."""
        async with self._lock:
            session = self._sessions.get(device_name)
        if session is None or session.session_id != session_id:
            return False

        added = 0
        for candidate in _parse_trickle_candidates(sdp_fragment):
            session.agora_handler.add_ice_candidate(candidate)
            added += 1

        if added:
            LOGGER.debug(
                "Collected %d upstream PATCH candidates for %s", added, device_name
            )
        return True

    async def close_all(self) -> None:
        """Close all active upstream sessions."""
        async with self._lock:
            device_names = list(self._sessions)
        for device_name in device_names:
            await self.close_session(device_name)

    async def _refresh_tokens(self, session: AgoraUpstreamSession, camera) -> None:
        """Periodically refresh stream token while the upstream session is alive."""
        while True:
            await asyncio.sleep(TOKEN_REFRESH_INTERVAL_SECONDS)
            try:
                stream_data, _ = await camera.coordinator.async_check_stream_expiry()
                if stream_data is not None and stream_data.data is not None:
                    LOGGER.debug(
                        "Refreshed stream token for upstream session %s",
                        session.device_name,
                    )
            except Exception as err:  # noqa: BLE001
                LOGGER.debug(
                    "Token refresh error for %s: %s", session.device_name, err
                )


# ---------------------------------------------------------------------------
# Proxy session management  (browser → go2rtc)
# ---------------------------------------------------------------------------


@dataclass
class Go2RTCProxySession:
    """One public WHEP session proxied to the internal shared go2rtc stream."""

    session_id: str
    device_name: str
    upstream_location: str | None


class MammotionGo2RTCProxyManager:
    """Proxy public WHEP sessions to the internal shared go2rtc stream."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._sessions: dict[tuple[str, str], Go2RTCProxySession] = {}

    @property
    def _session(self) -> ClientSession:
        return async_get_clientsession(self.hass)

    async def create_session(
        self,
        device_name: str,
        offer_sdp: str,
        headers,
    ) -> tuple[str, str]:
        """Create one public WHEP session against the shared go2rtc stream."""
        stream_manager = get_go2rtc_stream_manager(self.hass)
        base_url = stream_manager.api_base_url()
        if base_url is None:
            raise RuntimeError("No shared go2rtc instance configured")

        await stream_manager.async_ensure_stream(device_name, raise_on_failure=True)

        response = await self._request(
            "POST",
            self._stream_url(device_name, base_url),
            body=offer_sdp.encode(),
            headers=headers,
        )
        if response.status not in (HTTPStatus.OK, HTTPStatus.CREATED):
            raise RuntimeError(
                f"go2rtc WHEP setup failed with status {response.status}: {response.body_text}"
            )

        upstream_location = response.headers.get("Location")
        proxy_session_id = secrets.token_hex(16)
        async with self._lock:
            self._sessions[(device_name, proxy_session_id)] = Go2RTCProxySession(
                session_id=proxy_session_id,
                device_name=device_name,
                upstream_location=(
                    self._normalize_location(base_url, upstream_location)
                    if upstream_location
                    else None
                ),
            )

        return proxy_session_id, response.body_text

    async def proxy_session_request(
        self,
        device_name: str,
        session_id: str,
        method: str,
        *,
        body: bytes = b"",
        headers,
        forget: bool = False,
    ) -> _ProxyResponse | None:
        async with self._lock:
            session = self._sessions.get((device_name, session_id))
        if session is None:
            return None

        if session.upstream_location is None:
            if forget or method == "DELETE":
                async with self._lock:
                    self._sessions.pop((device_name, session_id), None)
            status = HTTPStatus.OK if method == "DELETE" else HTTPStatus.NO_CONTENT
            return _ProxyResponse(status=status, body=b"", body_text="", headers={}, content_type=None)

        response = await self._request(method, session.upstream_location, body=body, headers=headers)

        if forget or method == "DELETE":
            async with self._lock:
                self._sessions.pop((device_name, session_id), None)

        return response

    async def close_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            with contextlib.suppress(Exception):
                if session.upstream_location:
                    await self._request("DELETE", session.upstream_location, headers={})

    def _stream_url(self, device_name: str, base_url: str) -> str:
        stream_name = get_go2rtc_stream_manager(self.hass).stream_name(device_name)
        return f"{base_url}{_GO2RTC_WHEP_PATH}?src={stream_name}"

    def _normalize_location(self, base_url: str, location: str) -> str:
        if urlsplit(location).scheme in {"http", "https"}:
            return location
        if location.startswith("/"):
            return f"{base_url.rstrip('/')}{location}"
        return f"{base_url}{location}"

    async def _request(self, method: str, url: str, *, body: bytes = b"", headers) -> "_ProxyResponse":
        forward_headers = _filter_proxy_headers(headers)
        stream_manager = get_go2rtc_stream_manager(self.hass)
        session = stream_manager.api_session()
        try:
            async with session.request(
                method,
                url,
                data=body or None,
                headers=forward_headers,
                timeout=_REQUEST_TIMEOUT,
            ) as response:
                raw_body = await response.read()
                response_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() in {"content-type", "etag", "location"}
                }
                return _ProxyResponse(
                    status=response.status,
                    body=raw_body,
                    body_text=raw_body.decode(errors="ignore"),
                    headers=response_headers,
                    content_type=response.content_type,
                )
        except (ClientError, TimeoutError) as err:
            raise RuntimeError(f"go2rtc proxy request failed: {err}") from err


@dataclass(frozen=True)
class _ProxyResponse:
    status: int
    body: bytes
    body_text: str
    headers: dict[str, str]
    content_type: str | None


def _filter_proxy_headers(headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() in {"content-type", "accept", "if-match"}
    }


# ---------------------------------------------------------------------------
# Helper: parse trickle ICE SDP fragment
# ---------------------------------------------------------------------------


def _parse_trickle_candidates(sdp_fragment: str) -> list[RTCIceCandidateInit]:
    try:
        parsed = sdp_parse(sdp_fragment)
    except Exception:  # noqa: BLE001
        return []

    candidates: list[RTCIceCandidateInit] = []
    for media in parsed.get("media", []) or []:
        mid = media.get("mid")
        mline_index = media.get("mLineIndex")
        for candidate in media.get("candidates", []) or []:
            foundation = candidate.get("foundation", "0")
            component = candidate.get("component", 1)
            transport = candidate.get("transport", "udp")
            priority = candidate.get("priority", 0)
            ip = candidate.get("ip", "")
            port = candidate.get("port", 0)
            candidate_type = candidate.get("type", "host")
            candidate_line = (
                f"candidate:{foundation} {component} {transport} "
                f"{priority} {ip} {port} typ {candidate_type}"
            )
            candidates.append(
                RTCIceCandidateInit(
                    candidate=candidate_line,
                    sdp_mid=str(mid) if mid is not None else None,
                    sdp_m_line_index=(int(mline_index) if isinstance(mline_index, int) else None),
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# Module-level manager accessors
# ---------------------------------------------------------------------------


def get_whep_upstream_manager(hass) -> MammotionAgoraUpstreamManager:
    """Return the shared upstream manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("whep_upstream_manager")
    if manager is None:
        manager = MammotionAgoraUpstreamManager(hass)
        domain_data["whep_upstream_manager"] = manager
    return manager


def get_whep_proxy_manager(hass) -> MammotionGo2RTCProxyManager:
    """Return the shared public WHEP proxy manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("whep_proxy_manager")
    if manager is None:
        manager = MammotionGo2RTCProxyManager(hass)
        domain_data["whep_proxy_manager"] = manager
    return manager


async def async_cleanup_whep_sessions(hass) -> None:
    """Close all active WHEP state (called on integration unload)."""
    proxy_manager = hass.data.get(DOMAIN, {}).pop("whep_proxy_manager", None)
    if proxy_manager is not None:
        await proxy_manager.close_all()

    upstream_manager = hass.data.get(DOMAIN, {}).pop("whep_upstream_manager", None)
    if upstream_manager is not None:
        await upstream_manager.close_all()


# ---------------------------------------------------------------------------
# WHEP HTTP views
# ---------------------------------------------------------------------------


class MammotionUpstreamWhepView(HomeAssistantView):
    """Internal WHEP endpoint used by the shared go2rtc stream (go2rtc → HA → Agora)."""

    url = "/api/mammotion/whep_upstream/{device_name}"
    name = "api:mammotion:whep_upstream"
    requires_auth = False

    async def post(self, request: web.Request, device_name: str) -> web.Response:
        """Receive the internal go2rtc SDP offer, return the Agora SDP answer."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        hass = request.app["hass"]
        cameras = hass.data.get(DOMAIN, {}).get("cameras", {})
        camera = cameras.get(device_name)
        if camera is None:
            return web.Response(status=404, text="Camera not found")

        offer_sdp = await request.text()
        if not offer_sdp or not offer_sdp.strip():
            return web.Response(status=400, text="Empty SDP offer")

        try:
            session_id, answer_sdp = await get_whep_upstream_manager(hass).create_session(
                camera, offer_sdp
            )
        except (OSError, RuntimeError, ValueError) as err:
            LOGGER.error("go2rtc upstream WHEP failed for %s: %s", device_name, err)
            return web.Response(status=502, text=str(err))

        return web.Response(
            status=201,
            text=answer_sdp,
            content_type="application/sdp",
            headers={"Location": f"{request.path}/{session_id}"},
        )


class MammotionUpstreamWhepSessionView(HomeAssistantView):
    """Session-scoped internal upstream WHEP resource (PATCH = trickle ICE, DELETE = teardown)."""

    url = "/api/mammotion/whep_upstream/{device_name}/{session_id}"
    name = "api:mammotion:whep_upstream:session"
    requires_auth = False

    async def patch(
        self, request: web.Request, device_name: str, session_id: str
    ) -> web.Response:
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        body = await request.text()
        if not await get_whep_upstream_manager(request.app["hass"]).add_session_candidates(
            device_name, session_id, body
        ):
            return web.Response(status=404, text="No active upstream WHEP session")
        return web.Response(status=204)

    async def delete(
        self, request: web.Request, device_name: str, session_id: str
    ) -> web.Response:
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        if not await get_whep_upstream_manager(request.app["hass"]).close_session(device_name):
            return web.Response(status=404, text="No active upstream WHEP session")
        return web.Response(status=200, text="Session closed")


class MammotionDirectWhepProxyView(HomeAssistantView):
    """Public WHEP endpoint backed by the shared internal go2rtc stream."""

    url = "/api/mammotion/whep_direct/{device_name}"
    name = "api:mammotion:whep_direct"
    requires_auth = False

    async def post(self, request: web.Request, device_name: str) -> web.Response:
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        hass = request.app["hass"]
        cameras = hass.data.get(DOMAIN, {}).get("cameras", {})
        if device_name not in cameras:
            return web.Response(status=404, text="Camera not found")

        offer_sdp = await request.text()
        if not offer_sdp or not offer_sdp.strip():
            return web.Response(status=400, text="Empty SDP offer")

        stream_manager = get_go2rtc_stream_manager(hass)
        if not stream_manager.is_available():
            return web.Response(
                status=503,
                text="HA-managed go2rtc is required for the shared Mammotion stream",
            )

        try:
            session_id, answer_sdp = await get_whep_proxy_manager(hass).create_session(
                device_name, offer_sdp, request.headers
            )
        except RuntimeError as err:
            LOGGER.error("Direct WHEP proxy failed for %s: %s", device_name, err)
            return web.Response(status=502, text=str(err))

        return web.Response(
            status=201,
            text=answer_sdp,
            content_type="application/sdp",
            headers={"Location": f"{request.path}/{session_id}"},
        )


class MammotionDirectWhepProxySessionView(HomeAssistantView):
    """Session-scoped public WHEP resource for PATCH / DELETE."""

    url = "/api/mammotion/whep_direct/{device_name}/{session_id}"
    name = "api:mammotion:whep_direct:session"
    requires_auth = False

    async def patch(
        self, request: web.Request, device_name: str, session_id: str
    ) -> web.Response:
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        body = await request.read()
        proxied = await get_whep_proxy_manager(request.app["hass"]).proxy_session_request(
            device_name, session_id, "PATCH", body=body, headers=request.headers
        )
        if proxied is not None:
            return web.Response(status=proxied.status, headers=proxied.headers)
        return web.Response(status=404, text="No active direct WHEP session")

    async def delete(
        self, request: web.Request, device_name: str, session_id: str
    ) -> web.Response:
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        proxied = await get_whep_proxy_manager(request.app["hass"]).proxy_session_request(
            device_name, session_id, "DELETE", headers=request.headers, forget=True
        )
        if proxied is not None:
            return web.Response(status=proxied.status, headers=proxied.headers)
        return web.Response(status=404, text="No active direct WHEP session")
