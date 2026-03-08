from __future__ import annotations

from collections import deque
import json
import socket
import threading
import time
from typing import Any, Callable


_PROTOCOL_NAME = "EVE_SIM_LAN"
_PROTOCOL_VERSION = 1
_MAX_FRAME_BYTES = 4 * 1024 * 1024
_MAX_OUTGOING_QUEUE = 512
_MAX_COMMAND_QUEUE = 4096
_SOCKET_TIMEOUT_SEC = 0.5
_RECV_CHUNK_SIZE = 8192


def _close_socket_quietly(sock: socket.socket | None) -> None:
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


def _encode_packet(packet_type: str, payload: dict[str, Any], packet_id: int) -> bytes:
    envelope = {
        "protocol": _PROTOCOL_NAME,
        "version": _PROTOCOL_VERSION,
        "id": int(packet_id),
        "ts": float(time.time()),
        # Keep legacy field name for backward compatibility.
        "type": str(packet_type),
        "payload": payload,
    }
    return (json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _decode_packet(line: bytes) -> tuple[str, dict[str, Any]] | None:
    try:
        raw = json.loads(line.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    protocol = raw.get("protocol")
    if protocol is not None and str(protocol) != _PROTOCOL_NAME:
        return None

    version = raw.get("version")
    if version is not None:
        try:
            if int(version) > _PROTOCOL_VERSION:
                return None
        except Exception:
            return None

    packet_type = raw.get("type", raw.get("kind"))
    payload = raw.get("payload")
    if not isinstance(packet_type, str) or not packet_type:
        return None
    if not isinstance(payload, dict):
        return None
    return packet_type, payload


PacketHandler = Callable[["_SocketChannel", str, dict[str, Any]], None]
ClosedHandler = Callable[["_SocketChannel"], None]


class _SocketChannel:
    def __init__(
        self,
        sock: socket.socket,
        *,
        packet_handler: PacketHandler,
        closed_handler: ClosedHandler,
        name: str,
    ) -> None:
        self._sock = sock
        self._packet_handler = packet_handler
        self._closed_handler = closed_handler
        self._name = name

        self._lock = threading.Lock()
        self._closed = False
        self._stop = threading.Event()
        self._send_event = threading.Event()
        self._outgoing: deque[tuple[str, bytes]] = deque()
        self._next_packet_id = 0

        self._rx_thread: threading.Thread | None = None
        self._tx_thread: threading.Thread | None = None

        try:
            self._sock.settimeout(_SOCKET_TIMEOUT_SEC)
        except Exception:
            pass

    def start(self) -> None:
        self._rx_thread = threading.Thread(target=self._recv_loop, name=f"{self._name}-rx", daemon=True)
        self._tx_thread = threading.Thread(target=self._send_loop, name=f"{self._name}-tx", daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()

    def send_packet(self, packet_type: str, payload: dict[str, Any], *, prefer_latest: bool) -> bool:
        if not isinstance(payload, dict):
            return False

        with self._lock:
            if self._closed:
                return False
            self._next_packet_id += 1
            frame = _encode_packet(packet_type, payload, self._next_packet_id)

            if prefer_latest:
                self._outgoing = deque((kind, data) for kind, data in self._outgoing if kind != packet_type)

            while len(self._outgoing) >= _MAX_OUTGOING_QUEUE:
                self._outgoing.popleft()

            self._outgoing.append((packet_type, frame))

        self._send_event.set()
        return True

    def close(self) -> None:
        should_notify = self._request_close()
        if should_notify:
            try:
                self._closed_handler(self)
            except Exception:
                pass
        self._join_threads()

    def _request_close(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._closed = True
        self._stop.set()
        self._send_event.set()
        _close_socket_quietly(self._sock)
        return True

    def _join_threads(self) -> None:
        current = threading.current_thread()
        if self._rx_thread is not None and self._rx_thread.is_alive() and self._rx_thread is not current:
            self._rx_thread.join(timeout=0.8)
        if self._tx_thread is not None and self._tx_thread.is_alive() and self._tx_thread is not current:
            self._tx_thread.join(timeout=0.8)

    def _finalize_from_worker(self) -> None:
        if self._request_close():
            try:
                self._closed_handler(self)
            except Exception:
                pass

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            self._send_event.wait(timeout=_SOCKET_TIMEOUT_SEC)
            self._send_event.clear()

            while True:
                with self._lock:
                    if self._closed or not self._outgoing:
                        break
                    _kind, frame = self._outgoing.popleft()
                try:
                    self._sock.sendall(frame)
                except Exception:
                    self._finalize_from_worker()
                    return

    def _recv_loop(self) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(_RECV_CHUNK_SIZE)
            except socket.timeout:
                continue
            except Exception:
                break

            if not chunk:
                break

            buf += chunk
            if len(buf) > (_MAX_FRAME_BYTES * 2):
                # Drop connection if peer keeps sending unframed/unbounded payload.
                break

            while True:
                idx = buf.find(b"\n")
                if idx < 0:
                    break
                line = buf[:idx]
                buf = buf[idx + 1 :]
                if not line:
                    continue
                if len(line) > _MAX_FRAME_BYTES:
                    continue
                packet = _decode_packet(line)
                if packet is None:
                    continue
                packet_type, payload = packet
                try:
                    self._packet_handler(self, packet_type, payload)
                except Exception:
                    continue

        self._finalize_from_worker()


class HostLanSession:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._client_channel: _SocketChannel | None = None
        self._client_connected = threading.Event()
        self._commands: deque[dict[str, Any]] = deque()

    @property
    def client_connected(self) -> bool:
        return self._client_connected.is_set()

    def start(self) -> None:
        if self._accept_thread is not None and self._accept_thread.is_alive():
            return

        self._stop.clear()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(_SOCKET_TIMEOUT_SEC)
        self._server_sock = server

        self._accept_thread = threading.Thread(target=self._accept_loop, name="lan-host-accept", daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            server = self._server_sock
            if server is None:
                break
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

            channel = _SocketChannel(
                conn,
                packet_handler=self._handle_client_packet,
                closed_handler=self._handle_client_closed,
                name="lan-host-client",
            )

            with self._lock:
                old_channel = self._client_channel
                self._client_channel = channel
                self._client_connected.set()

            if old_channel is not None:
                old_channel.close()

            channel.start()
            channel.send_packet(
                "hello",
                {"role": "host", "protocol_version": _PROTOCOL_VERSION},
                prefer_latest=True,
            )

    def _handle_client_packet(self, channel: _SocketChannel, packet_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if channel is not self._client_channel:
                return

            if packet_type == "command":
                while len(self._commands) >= _MAX_COMMAND_QUEUE:
                    self._commands.popleft()
                self._commands.append(dict(payload))
                return

            if packet_type == "ping":
                channel.send_packet("pong", {"at": float(time.time())}, prefer_latest=True)

    def _handle_client_closed(self, channel: _SocketChannel) -> None:
        with self._lock:
            if channel is not self._client_channel:
                return
            self._client_channel = None
            self._client_connected.clear()

    def poll_commands(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._commands)
            self._commands.clear()
        return out

    def send_state(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            channel = self._client_channel
        if channel is None:
            self._client_connected.clear()
            return
        ok = channel.send_packet("state", dict(snapshot), prefer_latest=True)
        if not ok:
            self._client_connected.clear()

    def stop(self) -> None:
        self._stop.set()

        with self._lock:
            channel = self._client_channel
            self._client_channel = None
            server = self._server_sock
            self._server_sock = None

        if channel is not None:
            channel.close()

        _close_socket_quietly(server)
        self._client_connected.clear()

        if self._accept_thread is not None and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=1.0)


class ClientLanSession:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)

        self._lock = threading.Lock()
        self._channel: _SocketChannel | None = None
        self._connected = threading.Event()
        self._latest_state: dict[str, Any] | None = None

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def connect(self, timeout_sec: float = 6.0, timeout: float | None = None) -> bool:
        effective_timeout = float(timeout) if timeout is not None else float(timeout_sec)

        self.close()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            sock.settimeout(effective_timeout)
            sock.connect((self.host, self.port))
            sock.settimeout(_SOCKET_TIMEOUT_SEC)
        except Exception:
            _close_socket_quietly(sock)
            self._connected.clear()
            return False

        channel = _SocketChannel(
            sock,
            packet_handler=self._handle_server_packet,
            closed_handler=self._handle_server_closed,
            name="lan-client",
        )
        with self._lock:
            self._channel = channel
            self._latest_state = None
            self._connected.set()

        channel.start()
        channel.send_packet(
            "hello",
            {"role": "client", "protocol_version": _PROTOCOL_VERSION},
            prefer_latest=True,
        )
        return True

    def _handle_server_packet(self, channel: _SocketChannel, packet_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if channel is not self._channel:
                return

            if packet_type == "state":
                self._latest_state = dict(payload)
                return

        if packet_type == "ping":
            channel.send_packet("pong", {"at": float(time.time())}, prefer_latest=True)

    def _handle_server_closed(self, channel: _SocketChannel) -> None:
        with self._lock:
            if channel is not self._channel:
                return
            self._channel = None
            self._connected.clear()

    def consume_latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            state = self._latest_state
            self._latest_state = None
        return dict(state) if isinstance(state, dict) else None

    def send_command(self, command: dict[str, Any]) -> None:
        if not isinstance(command, dict):
            return
        with self._lock:
            channel = self._channel
        if channel is None:
            self._connected.clear()
            return
        ok = channel.send_packet("command", dict(command), prefer_latest=False)
        if not ok:
            self._connected.clear()

    def close(self) -> None:
        with self._lock:
            channel = self._channel
            self._channel = None
            self._latest_state = None
            self._connected.clear()
        if channel is not None:
            channel.close()
