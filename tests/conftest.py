import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["SOTTO_PLATFORM"] = "null"  # ни звука, ни окон, ни хоткеев


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Каждый тест — со своими каталогами, чтобы не трогать настоящие данные."""
    from sotto import paths
    for name, sub in [("CONFIG_DIR", "config"), ("DATA_DIR", "data"), ("STATE_DIR", "state")]:
        monkeypatch.setattr(paths, name, tmp_path / sub)
    monkeypatch.setattr(paths, "CONFIG_FILE", tmp_path / "config" / "config.toml")
    monkeypatch.setattr(paths, "PROFILE_FILE", tmp_path / "config" / "profile.md")
    monkeypatch.setattr(paths, "PROMPTS_DIR", tmp_path / "config" / "prompts")
    monkeypatch.setattr(paths, "DB_FILE", tmp_path / "data" / "sotto.db")
    monkeypatch.setattr(paths, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(paths, "RECORDINGS_DIR", tmp_path / "data" / "recordings")
    monkeypatch.setattr(paths, "SCREENSHOTS_DIR", tmp_path / "data" / "screenshots")
    paths.ensure_dirs()
    # Настоящие CLI-агенты (claude, codex…) в тестах не запускаем: фоновые задачи вроде обновления тем
    # получат «нет агентов». Тесты, которым агент нужен, подменяют его сами.
    from sotto.infrastructure import platform
    from sotto.infrastructure.llm.agents import Agents
    from sotto.infrastructure.platform.null import NullPlatform
    monkeypatch.setattr(Agents, "installed", lambda self: [])
    platform.use(NullPlatform())
    yield tmp_path
