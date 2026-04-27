"""Mammotion camera entities."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import functools
import json
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import websockets
from go2rtc_client.ws import (
    Go2RtcWsClient,
    WebRTCAnswer as Go2RTCAnswer,
    WebRTCCandidate as Go2RTCCandidate,
    WebRTCOffer as Go2RTCOffer,
    WsError as Go2RTCWsError,
)
from homeassistant.components.camera import (
    CameraCapabilities,
    CameraEntityDescription,
    WebRTCAnswer,
    WebRTCCandidate,
    WebRTCError,
    WebRTCSendMessage,
)
from homeassistant.components.camera.const import StreamType
from homeassistant.components.web_rtc import (
    async_register_ice_servers,
)
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.network import NoURLAvailableError, get_url
from pymammotion.http.model.camera_stream import (
    StreamSubscriptionResponse,
)
from pymammotion.utility.device_type import DeviceType
from webrtc_models import RTCIceCandidateInit, RTCIceServer

from . import MammotionConfigEntry
from .agora_api import AgoraResponse
from .agora_websocket import AgoraWebSocketHandler
from .coordinator import MammotionBaseUpdateCoordinator
from .entity import MammotionCameraBaseEntity
from .go2rtc_stream import get_go2rtc_stream_manager
from .models import MammotionMowerData
from .whep_proxy import (
    MammotionDirectWhepProxySessionView,
    MammotionDirectWhepProxyView,
    MammotionUpstreamWhepSessionView,
    MammotionUpstreamWhepView,
    async_cleanup_whep_sessions,
    get_whep_upstream_manager,
)

_LOGGER = logging.getLogger(__name__)

PLACEHOLDER = Path(__file__).parent / "placeholder.png"


# ---------------------------------------------------------------------------
# Browser session state (for go2rtc path)
# ---------------------------------------------------------------------------


class _BrowserSessionState(Enum):
    PENDING = auto()
    ACTIVE = auto()
    CLOSED = auto()
    FAILED = auto()


@dataclass
class _BrowserSession:
    state: _BrowserSessionState
    ws_client: Go2RtcWsClient | None = None
    queued_candidates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity description
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MammotionCameraEntityDescription(CameraEntityDescription):
    """Describes Mammotion camera entity."""

    key: str
    stream_fn: Callable[[MammotionBaseUpdateCoordinator], StreamSubscriptionResponse]


CAMERAS: tuple[MammotionCameraEntityDescription, ...] = (
    MammotionCameraEntityDescription(
        key="webrtc_camera",
        stream_fn=lambda coordinator: coordinator.get_stream_data(),
    ),
)


# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mammotion camera entities."""
    mowers = entry.runtime_data.mowers
    entities = []
    ice_servers = []

    non_luba1_mower = next(
        (
            mower
            for mower in mowers
            if not DeviceType.is_luba1(mower.device.device_name)
        ),
        None,
    )

    if non_luba1_mower is None:
        return

    (
        stream_data,
        agora_response,
    ) = await non_luba1_mower.reporting_coordinator.async_check_stream_expiry()

    if agora_response is not None:
        ice_servers = [
            RTCIceServer(
                urls=ice_server.urls,
                username=ice_server.username,
                credential=ice_server.credential,
            )
            for ice_server in agora_response.get_ice_servers(use_all_turn_servers=False)
        ]

    for mower in mowers:
        if not DeviceType.is_luba1(mower.device.device_name):
            _LOGGER.debug("Config camera for %s", mower.device.device_name)
            mower.reporting_coordinator._ice_servers = ice_servers

            for entity_description in CAMERAS:
                entities.append(
                    MammotionWebRTCCamera(
                        mower.reporting_coordinator, entity_description, hass
                    )
                )

    # Register WHEP HTTP views (idempotent)
    hass.http.register_view(MammotionUpstreamWhepView())
    hass.http.register_view(MammotionUpstreamWhepSessionView())
    hass.http.register_view(MammotionDirectWhepProxyView())
    hass.http.register_view(MammotionDirectWhepProxySessionView())

    async_add_entities(entities)
    await async_setup_platform_services(hass, entry)


# ---------------------------------------------------------------------------
# Camera entity
# ---------------------------------------------------------------------------


class MammotionWebRTCCamera(MammotionCameraBaseEntity):
    """Mammotion WebRTC camera entity.

    When a shared go2rtc instance is configured:
      - go2rtc pulls from HA via the upstream WHEP endpoint
        (/api/mammotion/whep_upstream/{device_name})
      - Browser offers are relayed through the go2rtc WS client
      - stream_source() returns the go2rtc RTSP URL (usable by VLC, NVR, etc.)

    When go2rtc is NOT configured the entity falls back to direct Agora
    WebSocket negotiation (previous behaviour).
    """

    entity_description: MammotionCameraEntityDescription
    _attr_capability_attributes = None

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator,
        entity_description: MammotionCameraEntityDescription,
        hass: HomeAssistant,
    ) -> None:
        super().__init__(coordinator, entity_description.key)
        self._cache: dict[str, Any] = {}
        self.access_tokens: collections.deque = collections.deque([], 2)
        self.async_update_token()
        self._create_stream_lock: asyncio.Lock | None = None
        # Direct Agora handler (fallback path)
        self._agora_handler = AgoraWebSocketHandler(hass)
        self.coordinator = coordinator
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key
        self._stream_data: StreamSubscriptionResponse | None = None
        self._attr_model = coordinator.device.device_name
        self.access_tokens = [secrets.token_hex(16)]
        # ICE servers (populated in setup)
        self.ice_servers = getattr(coordinator, "_ice_servers", [])
        self._remove_ice_servers: Callable[[], None] | None = None
        # go2rtc browser session state
        self._go2rtc_browser_sessions: dict[str, _BrowserSession] = {}

    # ------------------------------------------------------------------ #
    # HA lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        # Register this camera so WHEP views can find it by device_name
        self.hass.data.setdefault(DOMAIN, {}).setdefault("cameras", {})
        device_name = self.coordinator.device_name
        self.hass.data[DOMAIN]["cameras"][device_name] = self

        # Register ICE servers
        self._remove_ice_servers = async_register_ice_servers(
            self.hass, self.get_ice_servers
        )

        # Ensure go2rtc stream is registered (no-op if go2rtc not configured)
        go2rtc_manager = get_go2rtc_stream_manager(self.hass)
        if go2rtc_manager.is_available():
            try:
                await go2rtc_manager.async_ensure_stream(self)
            except RuntimeError as err:
                _LOGGER.debug(
                    "Failed to register shared go2rtc stream for %s: %s",
                    device_name,
                    err,
                )

    async def async_will_remove_from_hass(self) -> None:
        if self._remove_ice_servers:
            self._remove_ice_servers()
            self._remove_ice_servers = None

        device_name = self.coordinator.device_name
        if DOMAIN in self.hass.data and "cameras" in self.hass.data[DOMAIN]:
            self.hass.data[DOMAIN]["cameras"].pop(device_name, None)

        go2rtc_manager = get_go2rtc_stream_manager(self.hass)
        if go2rtc_manager.is_available():
            await go2rtc_manager.async_remove_stream(self)

        # Close all browser sessions
        await self._async_close_browser_sessions()
        await self._agora_handler.disconnect()

    # ------------------------------------------------------------------ #
    # Camera interface                                                      #
    # ------------------------------------------------------------------ #

    @property
    def camera_capabilities(self) -> CameraCapabilities:
        """Advertise WebRTC stream type."""
        return CameraCapabilities(frontend_stream_types={StreamType.WEB_RTC})

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        """Expose shared stream metadata."""
        go2rtc_manager = get_go2rtc_stream_manager(self.hass)
        device_name = self.coordinator.device_name
        attributes: dict[str, str] = {
            "whep_direct_url": self._whep_direct_url(),
        }
        if go2rtc_manager.is_available():
            attributes["go2rtc_stream_name"] = go2rtc_manager.stream_name(device_name)
        return attributes

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return a placeholder image for WebRTC cameras that don't support snapshots."""
        return await self.hass.async_add_executor_job(self.placeholder_image)

    @classmethod
    @functools.cache
    def placeholder_image(cls) -> bytes:
        return PLACEHOLDER.read_bytes()

    async def stream_source(self) -> str | None:
        """Return RTSP URL from go2rtc if available (used by stream consumers)."""
        go2rtc_manager = get_go2rtc_stream_manager(self.hass)
        if go2rtc_manager.is_available():
            rtsp_url = await go2rtc_manager.async_ensure_stream(self)
            if rtsp_url:
                return rtsp_url
        return None

    # ------------------------------------------------------------------ #
    # WebRTC signaling                                                      #
    # ------------------------------------------------------------------ #

    async def async_handle_async_webrtc_offer(
        self, offer_sdp: str, session_id: str, send_message: WebRTCSendMessage
    ) -> None:
        """Handle WebRTC offer.

        Uses go2rtc relay path when a go2rtc instance is configured,
        falls back to direct Agora WebSocket otherwise.
        """
        go2rtc_manager = get_go2rtc_stream_manager(self.hass)

        if go2rtc_manager.is_available():
            await self._handle_go2rtc_browser_offer(offer_sdp, session_id, send_message)
        else:
            await self._handle_direct_agora_offer(offer_sdp, session_id, send_message)

    async def async_on_webrtc_candidate(
        self, session_id: str, candidate: RTCIceCandidateInit
    ) -> None:
        """Handle incoming ICE candidate from browser."""
        # go2rtc path
        session = self._go2rtc_browser_sessions.get(session_id)
        if session is not None:
            if session.state in (_BrowserSessionState.CLOSED, _BrowserSessionState.FAILED):
                return
            if session.state == _BrowserSessionState.PENDING:
                session.queued_candidates.append(candidate.candidate)
                return
            if session.ws_client is not None:
                await session.ws_client.send(Go2RTCCandidate(candidate.candidate))
            return

        # Direct Agora path
        _LOGGER.info(
            "Received WebRTC candidate for direct session %s: %s", session_id, candidate
        )
        self._agora_handler.candidates.append(candidate)

    @callback
    def close_webrtc_session(self, session_id: str) -> None:
        """Close a go2rtc browser WebRTC session."""
        session = self._go2rtc_browser_sessions.pop(session_id, None)
        if session is None:
            return
        session.state = _BrowserSessionState.CLOSED
        session.queued_candidates.clear()
        if session.ws_client is not None:
            self.hass.async_create_task(session.ws_client.close())

    @callback
    async def async_close_webrtc_session(self, session_id: str) -> None:
        """Close WebRTC session (called by HA core)."""
        # go2rtc path
        self.close_webrtc_session(session_id)
        # Direct Agora path
        await self._agora_handler.disconnect()

    def get_ice_servers(self) -> list[RTCIceServer]:
        """Return the ICE servers from Agora API."""
        return self.ice_servers

    # ------------------------------------------------------------------ #
    # go2rtc browser offer relay                                           #
    # ------------------------------------------------------------------ #

    async def _handle_go2rtc_browser_offer(
        self,
        offer_sdp: str,
        session_id: str,
        send_message: WebRTCSendMessage,
    ) -> None:
        """Proxy one browser WebRTC session to the canonical internal go2rtc stream."""
        go2rtc_manager = get_go2rtc_stream_manager(self.hass)
        base_url = go2rtc_manager.api_base_url()
        if base_url is None:
            send_message(
                WebRTCError(
                    code="go2rtc_provider_missing",
                    message="No shared go2rtc instance is configured for this camera",
                )
            )
            return

        # Register PENDING session *before* any await so early ICE candidates can buffer
        existing = self._go2rtc_browser_sessions.pop(session_id, None)
        session = _BrowserSession(state=_BrowserSessionState.PENDING)
        self._go2rtc_browser_sessions[session_id] = session

        if existing is not None:
            existing.state = _BrowserSessionState.CLOSED
            existing.queued_candidates.clear()
            if existing.ws_client is not None:
                await existing.ws_client.close()

        try:
            await go2rtc_manager.async_ensure_stream(self, raise_on_failure=True)
        except RuntimeError as err:
            session.state = _BrowserSessionState.FAILED
            self._go2rtc_browser_sessions.pop(session_id, None)
            send_message(WebRTCError(code="go2rtc_stream_unavailable", message=str(err)))
            return

        if session.state in (_BrowserSessionState.CLOSED, _BrowserSessionState.FAILED):
            self._go2rtc_browser_sessions.pop(session_id, None)
            return

        ws_client = Go2RtcWsClient(
            go2rtc_manager.api_session(),
            base_url,
            source=go2rtc_manager.stream_name(self.coordinator.device_name),
        )

        @callback
        def on_messages(message) -> None:
            match message:
                case Go2RTCCandidate():
                    send_message(WebRTCCandidate(RTCIceCandidateInit(message.candidate)))
                case Go2RTCAnswer():
                    send_message(WebRTCAnswer(message.sdp))
                case Go2RTCWsError():
                    send_message(WebRTCError("go2rtc_webrtc_offer_failed", message.error))

        ws_client.subscribe(on_messages)
        session.ws_client = ws_client

        try:
            config = self.async_get_webrtc_client_configuration()
            await ws_client.send(Go2RTCOffer(offer_sdp, config.configuration.ice_servers))
        except Exception as err:  # noqa: BLE001
            session.state = _BrowserSessionState.FAILED
            self._go2rtc_browser_sessions.pop(session_id, None)
            await ws_client.close()
            send_message(WebRTCError(code="go2rtc_webrtc_offer_failed", message=str(err)))
            return

        if session.state in (_BrowserSessionState.CLOSED, _BrowserSessionState.FAILED):
            self._go2rtc_browser_sessions.pop(session_id, None)
            await ws_client.close()
            return

        # Flush buffered candidates and transition to ACTIVE
        buffered = list(session.queued_candidates)
        session.queued_candidates.clear()
        session.state = _BrowserSessionState.ACTIVE

        try:
            for candidate_str in buffered:
                await ws_client.send(Go2RTCCandidate(candidate_str))
        except Exception as err:  # noqa: BLE001
            session.state = _BrowserSessionState.FAILED
            self._go2rtc_browser_sessions.pop(session_id, None)
            await ws_client.close()
            send_message(WebRTCError(code="go2rtc_webrtc_offer_failed", message=str(err)))

    # ------------------------------------------------------------------ #
    # Direct Agora fallback (no go2rtc)                                    #
    # ------------------------------------------------------------------ #

    async def _handle_direct_agora_offer(
        self, offer_sdp: str, session_id: str, send_message: WebRTCSendMessage
    ) -> None:
        """Handle WebRTC offer by initiating a direct WebSocket connection to Agora."""
        stream_data, agora_response = await self.coordinator.async_check_stream_expiry()
        self._agora_handler.candidates = []
        _LOGGER.info("Handling direct WebRTC offer for session %s", session_id)

        try:
            await self.coordinator.join_webrtc_channel()
            if not stream_data or stream_data.data is None:
                _LOGGER.error("No stream data available for WebRTC offer")
                send_message(
                    WebRTCError("500", "No stream data available for WebRTC offer")
                )
                return

            agora_data = stream_data.data

            answer_sdp = await self._agora_handler.connect_and_join(
                agora_data=agora_data,
                offer_sdp=offer_sdp,
                session_id=session_id,
                agora_response=agora_response,
            )

            if answer_sdp:
                send_message(WebRTCAnswer(answer_sdp))
                _LOGGER.info("Direct WebRTC negotiation completed successfully")
            else:
                send_message(WebRTCError("500", "WebRTC negotiation failed"))

        except (websockets.exceptions.WebSocketException, json.JSONDecodeError) as ex:
            _LOGGER.error("Error handling WebRTC offer: %s", ex)
            send_message(WebRTCError("500", f"Error handling WebRTC offer: {ex}"))

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    async def _async_close_browser_sessions(self) -> None:
        sessions = list(self._go2rtc_browser_sessions.values())
        self._go2rtc_browser_sessions.clear()
        for s in sessions:
            s.state = _BrowserSessionState.CLOSED
            s.queued_candidates.clear()
        ws_clients = [s.ws_client for s in sessions if s.ws_client is not None]
        if ws_clients:
            await asyncio.gather(*(c.close() for c in ws_clients), return_exceptions=True)

    def _whep_direct_url(self) -> str:
        """Return the stable direct WHEP URL exposed by Home Assistant."""
        device_name = self.coordinator.device_name
        for prefer_external in (True, False):
            try:
                base_url = get_url(self.hass, prefer_external=prefer_external)
            except NoURLAvailableError:
                continue
            return f"{base_url.rstrip('/')}/api/mammotion/whep_direct/{device_name}"
        return f"/api/mammotion/whep_direct/{device_name}"


# ---------------------------------------------------------------------------
# Platform services (unchanged from original, kept for compatibility)
# ---------------------------------------------------------------------------

# Import DOMAIN here to avoid circular import at module level
from .const import DOMAIN  # noqa: E402


async def async_setup_platform_services(
    hass: HomeAssistant, entry: MammotionConfigEntry
) -> None:
    """Register custom services for streaming."""

    def _get_mower_by_entity_id(entity_id: str):
        state = hass.states.get(entity_id)
        name = state.attributes.get("model_name")
        return next(
            (
                mower
                for mower in entry.runtime_data.mowers
                if mower.device.device_name == name
            ),
            None,
        )

    async def handle_refresh_stream(call) -> None:
        entity_id = call.data["entity_id"]
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower:
            stream_data = await mower.api.get_stream_subscription(
                mower.device.device_name, mower.device.iot_id
            )
            _LOGGER.debug("Refresh stream data : %s", stream_data)
            mower.reporting_coordinator.set_stream_data(stream_data)
            mower.reporting_coordinator.async_update_listeners()

    async def handle_start_video(call) -> None:
        entity_id = call.data["entity_id"]
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.join_webrtc_channel()

    async def handle_stop_video(call) -> None:
        entity_id = call.data["entity_id"]
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.leave_webrtc_channel()

    async def handle_get_tokens(call: ServiceCall) -> ServiceResponse:
        entity_id = call.data["entity_id"]
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower is not None:
            stream_data = mower.reporting_coordinator.get_stream_data()
            if not stream_data or stream_data.data is None:
                return {}
            return stream_data.data.to_dict()
        return {}

    async def handle_move_forward(call) -> None:
        entity_id = call.data["entity_id"]
        speed = 0.4
        raw_speed = call.data["speed"]
        use_wifi = call.data["use_wifi"]
        if raw_speed is not None:
            try:
                speed_value = float(raw_speed)
                if 0.1 <= speed_value <= 1:
                    speed = speed_value
            except (ValueError, TypeError):
                pass
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_forward(speed=speed, use_wifi=use_wifi)

    async def handle_move_left(call) -> None:
        entity_id = call.data["entity_id"]
        speed = 0.4
        raw_speed = call.data["speed"]
        use_wifi = call.data["use_wifi"]
        if raw_speed is not None:
            try:
                speed_value = float(raw_speed)
                if 0.1 <= speed_value <= 1:
                    speed = speed_value
            except (ValueError, TypeError):
                pass
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_left(speed=speed, use_wifi=use_wifi)

    async def handle_move_right(call) -> None:
        entity_id = call.data["entity_id"]
        speed = 0.4
        raw_speed = call.data["speed"]
        use_wifi = call.data["use_wifi"]
        if raw_speed is not None:
            try:
                speed_value = float(raw_speed)
                if 0.1 <= speed_value <= 1:
                    speed = speed_value
            except (ValueError, TypeError):
                pass
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_right(speed=speed, use_wifi=use_wifi)

    async def handle_move_backward(call) -> None:
        entity_id = call.data["entity_id"]
        speed = 0.4
        raw_speed = call.data["speed"]
        use_wifi = call.data["use_wifi"]
        if raw_speed is not None:
            try:
                speed_value = float(raw_speed)
                if 0.1 <= speed_value <= 1:
                    speed = speed_value
            except (ValueError, TypeError):
                pass
        mower: MammotionMowerData = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_back(speed=speed, use_wifi=use_wifi)

    hass.services.async_register("mammotion", "refresh_stream", handle_refresh_stream)
    hass.services.async_register("mammotion", "start_video", handle_start_video)
    hass.services.async_register("mammotion", "stop_video", handle_stop_video)
    hass.services.async_register(
        "mammotion",
        "get_tokens",
        handle_get_tokens,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register("mammotion", "move_forward", handle_move_forward)
    hass.services.async_register("mammotion", "move_left", handle_move_left)
    hass.services.async_register("mammotion", "move_right", handle_move_right)
    hass.services.async_register("mammotion", "move_backward", handle_move_backward)
