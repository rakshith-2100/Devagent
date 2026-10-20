"""Phase 2 sandbox: every command runs in a throwaway Docker container.

What this buys over ``LocalSandbox`` (the file tools are unchanged; only
command execution moves into the container):

- the worktree is bind-mounted at ``/work`` and is the only writable path;
- ``--network none`` by default; a command the user explicitly approved
  (``npm install``, ``pip install``) gets network for that one call;
- memory, CPU and PID limits;
- a read-only root filesystem with a tmpfs for ``/tmp`` (``HOME`` points there);
- all capabilities dropped, and the container is removed when the command ends.

The network default is doing security work, not resource work: a coding agent
reads repository content that someone else may have written.
"""
from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from ..policy.rules import classify_command, parse_command
from .local import CommandNotAllowedError, CommandResult, LocalSandbox, SandboxError, _as_text

DEFAULT_IMAGE = "devagent-sandbox:latest"

# Built from stdin by ``DockerSandbox.build_image`` so the package needs no extra files.
SANDBOX_DOCKERFILE = r"""FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pytest \
 && git config --system --add safe.directory '*'
WORKDIR /work
"""


class DockerSandbox(LocalSandbox):
    """Same contract as LocalSandbox; ``run`` executes inside a fresh container."""

    def __init__(
        self,
        root: Path | str,
        *,
        image: str = DEFAULT_IMAGE,
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 256,
        run_id: str = "",
        docker_bin: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(root, **kwargs)  # type: ignore[arg-type]
        self.image = image
        self.memory = memory
        self.cpus = str(cpus)
        self.pids_limit = pids_limit
        self.run_id = run_id or uuid.uuid4().hex[:6]
        self.docker = docker_bin or shutil.which("docker") or "docker"
        self._seq = 0

    # ------------------------------------------------------------ engine
    @staticmethod
    def daemon_available(docker_bin: Optional[str] = None) -> tuple[bool, str]:
        exe = docker_bin or shutil.which("docker")
        if not exe:
            return False, "docker executable not found on PATH"
        try:
            r = subprocess.run([exe, "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        if r.returncode != 0:
            lines = (r.stderr or r.stdout).strip().splitlines()
            return False, (lines[-1] if lines else "docker engine not reachable")
        return True, r.stdout.strip()

    def image_present(self) -> bool:
        try:
            r = subprocess.run([self.docker, "image", "inspect", self.image], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0

    @classmethod
    def build_image(cls, image: str = DEFAULT_IMAGE, *, docker_bin: Optional[str] = None, log: Callable[[str], None] = print) -> None:
        exe = docker_bin or shutil.which("docker")
        if not exe:
            raise SandboxError("docker executable not found on PATH")
        log(f"building {image} (python 3.11, git, node 22, pytest) ...")
        proc = subprocess.Popen(
            [exe, "build", "-t", image, "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(SANDBOX_DOCKERFILE)
        proc.stdin.close()
        for line in proc.stdout:
            log(line.rstrip())
        if proc.wait() != 0:
            raise SandboxError(f"docker build failed with exit code {proc.returncode}")
        log(f"built {image}")

    # ---------------------------------------------------------- commands
    def container_env(self) -> dict[str, str]:
        """Nothing from the host environment crosses into the container."""
        return {
            "HOME": "/tmp",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_CACHE_DIR": "/tmp/pip",
            "npm_config_cache": "/tmp/npm",
            "npm_config_update_notifier": "false",
            "CI": "1",
            "NO_COLOR": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "DEVAGENT_SANDBOX": "docker",
        }

    def container_argv(
        self,
        argv: list[str],
        *,
        name: str,
        workdir: str = "/work",
        network: bool = False,
        env: Optional[dict[str, str]] = None,
    ) -> list[str]:
        cmd = [
            self.docker,
            "run",
            "--rm",
            "--name",
            name,
            "-v",
            f"{self.root}:/work",
            "-w",
            workdir,
            "--network",
            "bridge" if network else "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,size=1g",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        for key, value in (env or {}).items():
            cmd += ["-e", f"{key}={value}"]
        cmd.append(self.image)
        cmd.extend(argv)
        return cmd

    def run(  # type: ignore[override]
        self,
        command: str | list[str],
        *,
        timeout: float = 60,
        check_allowlist: bool = True,
        cwd: Optional[Path | str] = None,
        stdin_text: Optional[str] = None,
        approved: bool = False,
    ) -> CommandResult:
        argv = parse_command(command) if isinstance(command, str) else list(command)
        if not argv:
            raise CommandNotAllowedError("Empty command.")
        if check_allowlist:
            tier, _rule, reason = classify_command(argv)
            if tier < 0 or (tier >= 3 and not approved):
                raise CommandNotAllowedError(reason)
        workdir = "/work"
        if cwd is not None:
            rel = self.relpath(self.resolve(cwd))
            workdir = "/work" if rel in ("", ".") else f"/work/{rel}"
        self._seq += 1
        name = f"devagent-{self.run_id}-{self._seq}"
        docker_cmd = self.container_argv(argv, name=name, workdir=workdir, network=approved, env=self.container_env())
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                docker_cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout, stderr, rc = _as_text(e.stdout), _as_text(e.stderr), -1
            subprocess.run([self.docker, "kill", name], capture_output=True, timeout=30)
        except FileNotFoundError:
            return CommandResult(argv=argv, returncode=127, stdout="", stderr="docker executable not found on PATH", duration_ms=0)
        duration_ms = int((time.monotonic() - started) * 1000)
        if rc == 125 and ("Unable to find image" in stderr or "No such image" in stderr):
            stderr += f"\n(devagent: sandbox image {self.image} is missing; build it with `devagent build-sandbox`)"
        stdout, stderr, truncated = self._cap_output(stdout, stderr)
        return CommandResult(
            argv=argv,
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
            truncated=truncated,
            meta={"container": name, "network": approved, "image": self.image},
        )

    def run_git(self, *args: str, timeout: float = 60, cwd: Optional[Path] = None) -> CommandResult:
        """git stays on the host: it manages the worktree, which is host state."""
        return LocalSandbox.run(self, ["git", *args], timeout=timeout, check_allowlist=False, cwd=cwd)
