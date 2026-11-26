"""侧边问答分支存储适配器（按用户 + 日期单独落盘，与主会话 sessions/ 完全隔离）。

默认目录结构：
    .data/<user_id>/side_chats/YYYY/MM/DD/<side_id>.json
"""

from __future__ import annotations

from pathlib import Path

from app.domain import SideChat


def _sanitize_segment(raw: str, *, fallback: str = "") -> str:
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_"))
    return safe or fallback


class InMemorySideChatStore:
    """内存侧边问答分支存储（适合单元测试与 --storage memory）。"""

    def __init__(self, *, user_id: str = "default") -> None:
        self.user_id = _sanitize_segment(user_id, fallback="default")
        self._items: dict[str, SideChat] = {}

    async def get(self, side_id: str) -> SideChat | None:
        item = self._items.get(side_id)
        return item.model_copy(deep=True) if item else None

    async def save(self, side_chat: SideChat) -> None:
        self._items[side_chat.id] = side_chat.model_copy(deep=True)

    async def list_by_parent(self, parent_session_id: str) -> list[SideChat]:
        matched = [
            item.model_copy(deep=True)
            for item in self._items.values()
            if item.parent_session_id == parent_session_id
        ]
        matched.sort(key=lambda s: s.updated_at, reverse=True)
        return matched

    async def delete(self, side_id: str) -> bool:
        return self._items.pop(side_id, None) is not None


class JsonFileSideChatStore:
    """本地 JSON 文件侧边问答分支存储：.data/<user_id>/side_chats/YYYY/MM/DD/<side_id>.json。"""

    def __init__(self, base_dir: Path | str = Path(".data"), *, user_id: str = "default") -> None:
        self.base_dir = Path(base_dir)
        self.user_id = _sanitize_segment(user_id, fallback="default")
        self.side_chats_root.mkdir(parents=True, exist_ok=True)

    @property
    def side_chats_root(self) -> Path:
        return self.base_dir / self.user_id / "side_chats"

    def _safe_side_id(self, side_id: str) -> str:
        safe = _sanitize_segment(side_id)
        if not safe:
            raise ValueError(f"非法 side_id: {side_id}")
        return safe

    def _path_for(self, side_chat: SideChat) -> Path:
        safe_id = self._safe_side_id(side_chat.id)
        owner = _sanitize_segment(side_chat.user_id or self.user_id, fallback=self.user_id)
        local_dt = side_chat.created_at.astimezone()
        dated_dir = (
            self.base_dir
            / owner
            / "side_chats"
            / f"{local_dt.year:04d}"
            / f"{local_dt.month:02d}"
            / f"{local_dt.day:02d}"
        )
        dated_dir.mkdir(parents=True, exist_ok=True)
        return dated_dir / f"{safe_id}.json"

    def _find_path(self, side_id: str) -> Path | None:
        safe_id = self._safe_side_id(side_id)
        if not self.side_chats_root.exists():
            return None
        matches = list(self.side_chats_root.rglob(f"{safe_id}.json"))
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    async def get(self, side_id: str) -> SideChat | None:
        path = self._find_path(side_id)
        if path is None or not path.exists():
            return None
        return SideChat.model_validate_json(path.read_text(encoding="utf-8"))

    async def save(self, side_chat: SideChat) -> None:
        if not side_chat.user_id:
            side_chat.user_id = self.user_id
        path = self._path_for(side_chat)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(side_chat.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    async def list_by_parent(self, parent_session_id: str) -> list[SideChat]:
        if not self.side_chats_root.exists():
            return []
        by_id: dict[str, SideChat] = {}
        for file in self.side_chats_root.rglob("*.json"):
            try:
                side = SideChat.model_validate_json(file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if side.parent_session_id != parent_session_id:
                continue
            existing = by_id.get(side.id)
            if existing is None or side.updated_at > existing.updated_at:
                by_id[side.id] = side
        return sorted(by_id.values(), key=lambda s: s.updated_at, reverse=True)

    async def delete(self, side_id: str) -> bool:
        path = self._find_path(side_id)
        if path is None or not path.exists():
            return False
        path.unlink()
        return True
