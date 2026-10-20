from __future__ import annotations

import os
from pathlib import Path

import pytest

from devagent.core.config import Config, global_env_path, load_settings
from devagent.sandbox.docker import SANDBOX_DOCKERFILE, DockerSandbox
from devagent.sandbox.local import CommandNotAllowedError


def test_container_argv_is_locked_down(tmp_path: Path) -> None:
    sb = DockerSandbox(tmp_path, image="img:1", memory="1g", cpus="1.5", pids_limit=64, run_id="r1", docker_bin="docker")
    argv = sb.container_argv(["python", "-m", "pytest", "-q"], name="devagent-r1-1", env={"HOME": "/tmp"})
    joined = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert f"{tmp_path.resolve()}:/work" in argv and "-w" in argv and argv[argv.index("-w") + 1] == "/work"
    assert "--network none" in joined
    assert "--memory 1g" in joined and "--cpus 1.5" in joined and "--pids-limit 64" in joined
    assert "--read-only" in argv and "--tmpfs" in argv and "--cap-drop ALL" in joined and "no-new-privileges" in joined
    assert "-e HOME=/tmp" in joined
    assert argv[-5:] == ["img:1", "python", "-m", "pytest", "-q"]
    # an approved install gets network for that one call
    assert "--network bridge" in " ".join(sb.container_argv(["npm", "install"], name="n", network=True))


def test_container_env_has_no_host_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_API_KEY", "secret")
    env = DockerSandbox(tmp_path, docker_bin="docker").container_env()
    assert "BEDROCK_API_KEY" not in env and "PATH" not in env
    assert env["HOME"] == "/tmp" and env["DEVAGENT_SANDBOX"] == "docker"


def test_allowlist_still_applies_before_any_container_starts(tmp_path: Path) -> None:
    sb = DockerSandbox(tmp_path, docker_bin="definitely-not-docker")
    with pytest.raises(CommandNotAllowedError):
        sb.run("curl http://x")
    with pytest.raises(CommandNotAllowedError):
        sb.run("npm install")  # tier 3 without approval
    with pytest.raises(ValueError):
        sb.run("ls; rm -rf /")


def test_git_runs_on_the_host(temp_repo: Path) -> None:
    sb = DockerSandbox(temp_repo, docker_bin="definitely-not-docker")
    res = sb.run_git("rev-parse", "--is-inside-work-tree")
    assert res.returncode == 0 and res.stdout.strip() == "true"
    assert sb.tracked_and_untracked_files()


def test_dockerfile_builds_the_toolchain() -> None:
    assert "python:3.11-slim" in SANDBOX_DOCKERFILE
    for needle in ("nodejs", "pytest", "safe.directory", "WORKDIR /work"):
        assert needle in SANDBOX_DOCKERFILE


def test_global_settings_fill_gaps_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    (home / ".devagent").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))  # type: ignore[method-assign]
    assert global_env_path() == home / ".devagent" / ".env"
    (home / ".devagent" / ".env").write_text("OPENROUTER_MODEL=global/model\nDEVAGENT_SANDBOX=docker\n", encoding="utf-8")
    project = tmp_path / ".env"
    project.write_text("OPENROUTER_MODEL=project/model\n", encoding="utf-8")
    for key in ("OPENROUTER_MODEL", "DEVAGENT_SANDBOX"):
        monkeypatch.delenv(key, raising=False)
    found = load_settings(project)
    assert found == [project, home / ".devagent" / ".env"]
    cfg = Config.from_env()
    assert cfg.openrouter_model == "project/model"  # project wins
    assert cfg.sandbox == "docker"  # global fills the gap


@pytest.mark.skipif(not DockerSandbox.daemon_available()[0], reason="docker engine not running")
def test_real_container_run(temp_repo: Path) -> None:
    sb = DockerSandbox(temp_repo, run_id="t")
    if not sb.image_present():
        pytest.skip(f"sandbox image {sb.image} not built")
    res = sb.run(["python", "-c", "import os; print(os.getcwd(), os.environ.get('HOME'))"], check_allowlist=False)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "/work /tmp"
    # the worktree is mounted read-write, the rest of the filesystem is not
    res = sb.run(["python", "-c", "open('/work/.probe','w').write('x'); print('ok')"], check_allowlist=False)
    assert res.stdout.strip() == "ok" and (temp_repo / ".probe").is_file()
    res = sb.run(["python", "-c", "open('/etc/probe','w')"], check_allowlist=False)
    assert res.returncode != 0 and "Read-only" in res.stderr
    # no network by default
    res = sb.run(["python", "-c", "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)"], check_allowlist=False, timeout=30)
    assert res.returncode != 0
    os.remove(temp_repo / ".probe")
