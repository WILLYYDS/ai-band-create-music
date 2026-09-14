from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from app.main import create_app
from tests.helpers import make_orchestrator, make_settings


def test_complete_generation_over_real_http(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, enable_audio_splitting=True)
    app = create_app(settings, make_orchestrator(settings))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert server.started
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False
        ) as client:
            health = client.get("/api/health")
            generated = client.post(
                "/api/generate",
                json={"prompt": "cinematic rock with Mandarin vocal", "durationMinutes": 2},
            )
            body = generated.json()
            downloads = {name: client.get(url) for name, url in body["stems"].items()}
        assert health.status_code == 200
        assert generated.status_code == 200
        assert body["durationMinutes"] == 2
        assert body["structuredPrompt"].startswith("[Genre: Test]")
        assert all(response.status_code == 200 for response in downloads.values())
        assert all(response.content == b"ID3-stem-audio" for response in downloads.values())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
