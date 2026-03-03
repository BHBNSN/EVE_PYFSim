from __future__ import annotations

from collections import deque
import json
import socket
import threading
from typing import Any


class HostLanSession:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._client_connected = threading.Event()
        self._stop = threading.Event()
        self._rx_thread: threading.Thread | None = None
        self._accept_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._commands: deque[dict[str, Any]] = deque()

    @property
    def client_connected(self) -> bool:
        return self._client_connected.is_set()

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        self._server_sock = server
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while not self._stop.is_set():
            try:
                self._server_sock.settimeout(0.5)
                conn, _addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with self._lock:
                old = self._client_sock
                try:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                self._client_sock = conn
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
            self._client_connected.set()
            self._rx_thread = threading.Thread(target=self._recv_loop, args=(conn,), daemon=True)
            self._rx_thread.start()

    def _recv_loop(self, sock: socket.socket) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(4096)
            except Exception:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if msg.get("type") == "command" and isinstance(msg.get("payload"), dict):
                    with self._lock:
                        self._commands.append(dict(msg["payload"]))
        with self._lock:
            if self._client_sock is sock:
                self._client_sock = None
                self._client_connected.clear()
        try:
            sock.close()
        except Exception:
            pass

    def poll_commands(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._commands)
            self._commands.clear()
        return out

    def send_state(self, snapshot: dict[str, Any]) -> None:
        if not self.client_connected:
            return
        payload = {"type": "state", "payload": snapshot}
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            sock = self._client_sock
        if sock is None:
            return
        try:
            sock.sendall(data)
        except Exception:
            self._client_connected.clear()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            sock = self._client_sock
            self._client_sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass


class ClientLanSession:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._rx_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_state: dict[str, Any] | None = None

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def connect(self, timeout_sec: float = 6.0, timeout: float | None = None) -> bool:
        effective_timeout = float(timeout) if timeout is not None else float(timeout_sec)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            sock.settimeout(effective_timeout)
            sock.connect((self.host, self.port))
            sock.settimeout(None)
            self._sock = sock
            self._connected.set()
            self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._rx_thread.start()
            return True
        except Exception:
            try:
                sock.close()
            except Exception:
                pass
            self._connected.clear()
            return False

    def _recv_loop(self) -> None:
        sock = self._sock
        if sock is None:
            return
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(4096)
            except Exception:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if msg.get("type") == "state" and isinstance(msg.get("payload"), dict):
                    with self._lock:
                        self._latest_state = dict(msg["payload"])
        self._connected.clear()

    def consume_latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            state = self._latest_state
            self._latest_state = None
        return state

    def send_command(self, command: dict[str, Any]) -> None:
        if not self.connected:
            return
        payload = {"type": "command", "payload": command}
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        sock = self._sock
        if sock is None:
            return
        try:
            sock.sendall(data)
        except Exception:
            self._connected.clear()

    def close(self) -> None:
        self._stop.set()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        self._connected.clear()
