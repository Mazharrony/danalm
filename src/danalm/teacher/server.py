"""Start, reuse and stop the local llama.cpp server that hosts the teacher model.

Brief rule: teacher and student never share the GPU. Generation jobs wrap their work in
`TeacherServer`, and training jobs call `assert_teacher_stopped` before touching the GPU.
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def teacher_running(host: str, port: int) -> bool:
    """True if a ready llama.cpp server answers /health with {"status": "ok"} on host:port.
    (Any other service on the port, e.g. Docker Desktop on 8080, does not count.)"""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            return json.load(resp).get("status") == "ok"
    except (OSError, ValueError, AttributeError):
        return False


def assert_teacher_stopped(host: str, port: int) -> None:
    """Raise if the teacher server is still up (call before any student GPU job)."""
    if teacher_running(host, port):
        raise RuntimeError(
            f"The teacher server is running at {host}:{port}. Stop it first: teacher and "
            "student must never share the GPU."
        )


class TeacherServer:
    """Context manager: reuse a server already running on host:port, else start one and stop
    it on exit. `cfg` is the `teacher:` config section."""

    def __init__(self, cfg: dict[str, Any], log_path: Path) -> None:
        self.cfg = cfg
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://{cfg['host']}:{cfg['port']}"

    def __enter__(self) -> "TeacherServer":
        c = self.cfg
        if teacher_running(c["host"], c["port"]):
            print(f"[teacher] reusing the server already running at {self.base_url}")
            return self
        cmd = [
            c["server_exe"],
            "--model",
            c["model_path"],
            "--host",
            c["host"],
            "--port",
            str(c["port"]),
            "--ctx-size",
            str(c["ctx_size"]),
            "--parallel",
            str(c["parallel"]),
            *c["extra_args"],
        ]
        self._log = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed in __exit__
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + c["startup_timeout_s"]
        while not teacher_running(c["host"], c["port"]):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log_path}")
            if time.time() > deadline:
                self.__exit__()
                raise TimeoutError(f"llama-server not ready after {c['startup_timeout_s']}s")
            time.sleep(2)
        print(f"[teacher] started llama-server at {self.base_url} (log: {self.log_path})")
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._log.close()
        self.proc = None
        print("[teacher] stopped llama-server")
