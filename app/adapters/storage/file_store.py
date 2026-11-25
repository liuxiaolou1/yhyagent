"""本地 JSON 文件会话存储适配器（按用户 + 日期分层落盘，仿 Codex 目录风格）。

默认目录结构：
    .data/<user_id>/sessions/YYYY/MM/DD/<session_id>.json

同时兼容读取旧版平铺目录（.data/sessions/<session_id>.json）。
"""

from __future__ import annotations

from pathlib import Path

from app.domain import Session


def _sanitize_segment(raw: str, *, fallback: str = "") -> str:
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_"))
    return safe or fallback


class JsonFileSessionStore:
    """实现 application.ports.SessionStore 端口，按用户和日期持久化会话 JSON。"""

    def __init__(self, base_dir: Path | str = Path(".data"), *, user_id: str = "default") -> None:
        self.base_dir = Path(base_dir)
        self.user_id = _sanitize_segment(user_id, fallback="default")
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    @property
    def sessions_root(self) -> Path:
        return self.base_dir / self.user_id / "sessions"

    def _safe_session_id(self, session_id: str) -> str:
        safe_id = _sanitize_segment(session_id)
        if not safe_id:
            raise ValueError(f"非法 session_id: {session_id}")
        return safe_id

    def _path_for_session(self, session: Session) -> Path:
        safe_id = self._safe_session_id(session.id)
        owner = _sanitize_segment(session.user_id or self.user_id, fallback=self.user_id)
        local_dt = session.created_at.astimezone()
        dated_dir = (
            self.base_dir
            / owner
            / "sessions"
            / f"{local_dt.year:04d}"
            / f"{local_dt.month:02d}"
            / f"{local_dt.day:02d}"
        )
        dated_dir.mkdir(parents=True, exist_ok=True)
        return dated_dir / f"{safe_id}.json"

    def _find_existing_path(self, session_id: str) -> Path | None:
        safe_id = self._safe_session_id(session_id)
        filename = f"{safe_id}.json"

        # 1. 优先在当前用户的 YYYY/MM/DD 分层目录查找
        matches = list(self.sessions_root.rglob(filename)) if self.sessions_root.exists() else []
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

        # 2. 兼容旧版平铺路径（.data/sessions/<id>.json 或直接传入目录的 <id>.json）
        if self.user_id == "default":
            for legacy in (self.base_dir / "sessions" / filename, self.base_dir / filename):
                if legacy.exists():
                    return legacy
        return None

    async def get(self, session_id: str) -> Session | None:
        path = self._find_existing_path(session_id)
        if path is None or not path.exists():
            return None
        return Session.model_validate_json(path.read_text(encoding="utf-8"))

    async def save(self, session: Session) -> None:
        if not session.user_id:
            session.user_id = self.user_id
        path = self._path_for_session(session)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)

    async def list_sessions(self) -> list[Session]:
        candidates: list[Path] = []
        if self.sessions_root.exists():
            candidates.extend(self.sessions_root.rglob("*.json"))

        if self.user_id == "default":
            legacy_dir = self.base_dir / "sessions"
            if legacy_dir.exists():
                candidates.extend(legacy_dir.glob("*.json"))
            if self.base_dir.exists():
                candidates.extend(self.base_dir.glob("*.json"))

        by_id: dict[str, Session] = {}
        for file in candidates:
            try:
                session = Session.model_validate_json(file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            existing = by_id.get(session.id)
            if existing is None or session.updated_at > existing.updated_at:
                by_id[session.id] = session

        return sorted(by_id.values(), key=lambda s: s.updated_at, reverse=True)
