"""本地按用户 + 日期存储的长期记忆适配器（Markdown 文件 + 内存实现）。

默认目录结构：
    .data/<user_id>/memory/MEMORY.md         # 长期偏好、规则与固定事实
    .data/<user_id>/memory/YYYY/MM/DD.md     # 按日期追加的每日笔记与结论
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from app.domain import MemoryEntry, MemoryScope

_DATE_FILE_RE = re.compile(r"^\d{2}\.md$")
_TIMESTAMP_PREFIX_RE = re.compile(r"^\[(\d{2}:\d{2})\]\s*")


def _sanitize_user(user_id: str) -> str:
    safe = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_"))
    return safe or "default"


def _normalize_content(content: str) -> str:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    return " / ".join(lines)


def _extract_bullets(markdown_text: str) -> list[str]:
    items: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            items.append(line[2:].strip())
        else:
            items.append(line)
    return items


class InMemoryMemoryStore:
    """内存长期记忆适配器（适合单元测试或无状态临时运行）。"""

    def __init__(self, *, user_id: str = "default") -> None:
        self.user_id = _sanitize_user(user_id)
        self._entries: dict[str, list[MemoryEntry]] = {}

    async def remember(
        self,
        content: str,
        *,
        user_id: str = "default",
        scope: MemoryScope = MemoryScope.DAILY,
        recorded_at: datetime | None = None,
    ) -> MemoryEntry:
        owner = _sanitize_user(user_id or self.user_id)
        clean = _normalize_content(content)
        ts = (recorded_at or datetime.now(timezone.utc)).astimezone()
        source = "MEMORY.md" if scope == MemoryScope.PROFILE else f"{ts.year:04d}/{ts.month:02d}/{ts.day:02d}.md"

        bucket = self._entries.setdefault(owner, [])
        if scope == MemoryScope.PROFILE:
            for existing in bucket:
                if existing.scope == MemoryScope.PROFILE and existing.content == clean:
                    return existing.model_copy(deep=True)

        entry = MemoryEntry(
            user_id=owner,
            scope=scope,
            content=clean,
            source=source,
            created_at=ts.astimezone(timezone.utc),
        )
        bucket.append(entry)
        return entry.model_copy(deep=True)

    async def load_context(
        self,
        *,
        user_id: str = "default",
        recent_days: int = 2,
        max_chars: int = 3000,
    ) -> str:
        owner = _sanitize_user(user_id or self.user_id)
        entries = self._entries.get(owner, [])
        if not entries:
            return ""

        sections: list[str] = []
        profile_items = [e.content for e in entries if e.scope == MemoryScope.PROFILE]
        if profile_items:
            body = "\n".join(f"- {item}" for item in profile_items)
            sections.append(f"### 长期偏好与事实 (MEMORY.md)\n{body}")

        daily_by_day: dict[str, list[str]] = {}
        for entry in entries:
            if entry.scope == MemoryScope.DAILY:
                day_label = entry.source.removesuffix(".md").replace("/", "-")
                daily_by_day.setdefault(day_label, []).append(entry.content)

        for day_label in sorted(daily_by_day.keys(), reverse=True)[: max(0, recent_days)]:
            body = "\n".join(f"- {item}" for item in daily_by_day[day_label])
            sections.append(f"### 近期记录 ({day_label})\n{body}")

        combined = "\n\n".join(sections)
        return combined[:max_chars] if len(combined) > max_chars else combined

    async def search(
        self,
        query: str,
        *,
        user_id: str = "default",
        limit: int = 10,
    ) -> list[MemoryEntry]:
        owner = _sanitize_user(user_id or self.user_id)
        entries = list(reversed(self._entries.get(owner, [])))
        tokens = [tok.lower() for tok in query.strip().split() if tok.strip()]
        if not tokens:
            return [e.model_copy(deep=True) for e in entries[:limit]]

        scored: list[tuple[int, MemoryEntry]] = []
        for entry in entries:
            hay = f"{entry.content} {entry.source}".lower()
            hits = sum(1 for tok in tokens if tok in hay)
            if hits > 0:
                scored.append((hits, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [e.model_copy(deep=True) for _, e in scored[:limit]]


class LocalFileMemoryStore:
    """基于本地 Markdown 文件的长期记忆实现（按用户 + 日期分层落盘）。"""

    def __init__(self, base_dir: Path | str = Path(".data"), *, user_id: str = "default") -> None:
        self.base_dir = Path(base_dir)
        self.user_id = _sanitize_user(user_id)
        self._memory_root(self.user_id).mkdir(parents=True, exist_ok=True)

    def _memory_root(self, user_id: str) -> Path:
        return self.base_dir / _sanitize_user(user_id) / "memory"

    def _profile_path(self, user_id: str) -> Path:
        root = self._memory_root(user_id)
        root.mkdir(parents=True, exist_ok=True)
        return root / "MEMORY.md"

    def _daily_path(self, user_id: str, local_dt: datetime) -> Path:
        dated_dir = (
            self._memory_root(user_id)
            / f"{local_dt.year:04d}"
            / f"{local_dt.month:02d}"
        )
        dated_dir.mkdir(parents=True, exist_ok=True)
        return dated_dir / f"{local_dt.day:02d}.md"

    def _list_daily_files(self, user_id: str) -> list[tuple[str, Path]]:
        root = self._memory_root(user_id)
        if not root.exists():
            return []
        found: list[tuple[str, Path]] = []
        for path in root.rglob("*.md"):
            if path.name == "MEMORY.md" or not _DATE_FILE_RE.match(path.name):
                continue
            rel = path.relative_to(root).as_posix()
            found.append((rel, path))
        found.sort(key=lambda item: item[0], reverse=True)
        return found

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    async def remember(
        self,
        content: str,
        *,
        user_id: str = "default",
        scope: MemoryScope = MemoryScope.DAILY,
        recorded_at: datetime | None = None,
    ) -> MemoryEntry:
        owner = _sanitize_user(user_id or self.user_id)
        clean = _normalize_content(content)
        if not clean:
            raise ValueError("记忆内容不能为空")

        local_dt = (recorded_at or datetime.now(timezone.utc)).astimezone()
        if scope == MemoryScope.PROFILE:
            path = self._profile_path(owner)
            existing_text = path.read_text(encoding="utf-8") if path.exists() else "# 长期记忆 (MEMORY)\n\n"
            existing_bullets = _extract_bullets(existing_text)
            if clean not in existing_bullets:
                if existing_text and not existing_text.endswith("\n"):
                    existing_text += "\n"
                existing_text += f"- {clean}\n"
                self._atomic_write(path, existing_text)
            source = "MEMORY.md"
        else:
            path = self._daily_path(owner, local_dt)
            day_str = f"{local_dt.year:04d}-{local_dt.month:02d}-{local_dt.day:02d}"
            existing_text = path.read_text(encoding="utf-8") if path.exists() else f"# {day_str} 记忆日志\n\n"
            if existing_text and not existing_text.endswith("\n"):
                existing_text += "\n"
            existing_text += f"- [{local_dt.strftime('%H:%M')}] {clean}\n"
            self._atomic_write(path, existing_text)
            source = f"{local_dt.year:04d}/{local_dt.month:02d}/{local_dt.day:02d}.md"

        return MemoryEntry(
            user_id=owner,
            scope=scope,
            content=clean,
            source=source,
            created_at=local_dt.astimezone(timezone.utc),
        )

    async def load_context(
        self,
        *,
        user_id: str = "default",
        recent_days: int = 2,
        max_chars: int = 3000,
    ) -> str:
        owner = _sanitize_user(user_id or self.user_id)
        sections: list[str] = []

        profile_path = self._memory_root(owner) / "MEMORY.md"
        if profile_path.exists():
            bullets = _extract_bullets(profile_path.read_text(encoding="utf-8"))
            if bullets:
                body = "\n".join(f"- {b}" for b in bullets)
                sections.append(f"### 长期偏好与事实 (MEMORY.md)\n{body}")

        for rel, path in self._list_daily_files(owner)[: max(0, recent_days)]:
            bullets = _extract_bullets(path.read_text(encoding="utf-8"))
            if bullets:
                day_label = rel.removesuffix(".md").replace("/", "-")
                body = "\n".join(f"- {b}" for b in bullets[-15:])
                sections.append(f"### 近期记录 ({day_label})\n{body}")

        combined = "\n\n".join(sections)
        if len(combined) > max_chars:
            return combined[: max_chars - 12].rstrip() + "\n...(已截断)"
        return combined

    async def search(
        self,
        query: str,
        *,
        user_id: str = "default",
        limit: int = 10,
    ) -> list[MemoryEntry]:
        owner = _sanitize_user(user_id or self.user_id)
        tokens = [tok.lower() for tok in query.strip().split() if tok.strip()]

        candidates: list[tuple[MemoryScope, str, str]] = []
        profile_path = self._memory_root(owner) / "MEMORY.md"
        if profile_path.exists():
            for bullet in reversed(_extract_bullets(profile_path.read_text(encoding="utf-8"))):
                candidates.append((MemoryScope.PROFILE, "MEMORY.md", bullet))

        for rel, path in self._list_daily_files(owner):
            for bullet in reversed(_extract_bullets(path.read_text(encoding="utf-8"))):
                cleaned = _TIMESTAMP_PREFIX_RE.sub("", bullet)
                candidates.append((MemoryScope.DAILY, rel, cleaned))

        if not tokens:
            return [
                MemoryEntry(user_id=owner, scope=scope, source=source, content=content)
                for scope, source, content in candidates[:limit]
            ]

        scored: list[tuple[int, int, MemoryEntry]] = []
        for idx, (scope, source, content) in enumerate(candidates):
            hay = f"{content} {source}".lower()
            hits = sum(1 for tok in tokens if tok in hay)
            if hits > 0:
                scored.append(
                    (
                        hits,
                        -idx,
                        MemoryEntry(user_id=owner, scope=scope, source=source, content=content),
                    )
                )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]
