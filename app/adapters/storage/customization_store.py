"""本地自定义扩展存储适配器：扫描用户目录下的 rules/*.md 与 skills/*/SKILL.md。

目录约定：
    .data/<user_id>/rules/*.md             # 本地规则文件
    .data/<user_id>/skills/<dir>/SKILL.md  # 本地技能（列表仅提取 frontmatter name + description）

若目录尚未创建，零报错返回空列表，不影响系统正常运行。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.domain import RuleEntry, SkillSummary

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)


def _sanitize_user(user_id: str) -> str:
    safe = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_"))
    return safe or "default"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """轻量零依赖解析 Markdown YAML frontmatter 中的简单键值对。"""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    body = text[match.end() :]
    return meta, body


def _first_summary_line(body: str, *, max_chars: int = 120) -> str:
    for raw_line in body.splitlines():
        line = raw_line.strip().lstrip("#").strip()
        if line:
            return line[:max_chars]
    return ""


class InMemoryCustomizationStore:
    """内存规则与技能存储（适合单元测试与 --storage memory）。"""

    def __init__(self, *, user_id: str = "default") -> None:
        self.user_id = _sanitize_user(user_id)
        self._rules: dict[str, list[RuleEntry]] = {}
        self._skills: dict[str, dict[str, tuple[SkillSummary, str]]] = {}

    def add_rule(self, rule: RuleEntry, *, user_id: str | None = None) -> None:
        owner = _sanitize_user(user_id or self.user_id)
        self._rules.setdefault(owner, []).append(rule.model_copy(deep=True))

    def add_skill(self, summary: SkillSummary, full_content: str, *, user_id: str | None = None) -> None:
        owner = _sanitize_user(user_id or self.user_id)
        self._skills.setdefault(owner, {})[summary.name] = (summary.model_copy(deep=True), full_content)

    async def load_rules(self, *, user_id: str = "default") -> list[RuleEntry]:
        owner = _sanitize_user(user_id or self.user_id)
        return [r.model_copy(deep=True) for r in self._rules.get(owner, [])]

    async def load_skills(self, *, user_id: str = "default") -> list[SkillSummary]:
        owner = _sanitize_user(user_id or self.user_id)
        return [pair[0].model_copy(deep=True) for pair in self._skills.get(owner, {}).values()]

    async def read_skill_content(
        self,
        name: str,
        *,
        user_id: str = "default",
    ) -> tuple[str, str] | None:
        owner = _sanitize_user(user_id or self.user_id)
        target = name.strip().lower()
        for skill_name, (summary, content) in self._skills.get(owner, {}).items():
            if skill_name.lower() == target or summary.id.lower() == target:
                return summary.source_path, content
        return None


class LocalFileCustomizationStore:
    """扫描 .data/<user_id>/rules/*.md 与 .data/<user_id>/skills/*/SKILL.md 的只读适配器。"""

    def __init__(self, base_dir: Path | str = Path(".data"), *, user_id: str = "default") -> None:
        self.base_dir = Path(base_dir)
        self.user_id = _sanitize_user(user_id)

    def _user_root(self, user_id: str) -> Path:
        return self.base_dir / _sanitize_user(user_id or self.user_id)

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.base_dir).as_posix()
        except ValueError:
            return path.as_posix()

    async def load_rules(self, *, user_id: str = "default") -> list[RuleEntry]:
        rules_dir = self._user_root(user_id) / "rules"
        if not rules_dir.exists() or not rules_dir.is_dir():
            return []

        results: list[RuleEntry] = []
        for file in sorted(rules_dir.glob("*.md")):
            if not file.is_file():
                continue
            try:
                raw_text = file.read_text(encoding="utf-8")
            except OSError:
                continue
            if not raw_text.strip():
                continue
            meta, body = _parse_frontmatter(raw_text)
            name = meta.get("name") or file.stem
            description = meta.get("description") or _first_summary_line(body)
            results.append(
                RuleEntry(
                    id=f"rule:{file.stem}",
                    name=name,
                    source_path=self._display_path(file),
                    description=description,
                    content=raw_text.strip(),
                )
            )
        return results

    async def load_skills(self, *, user_id: str = "default") -> list[SkillSummary]:
        skills_dir = self._user_root(user_id) / "skills"
        if not skills_dir.exists() or not skills_dir.is_dir():
            return []

        results: list[SkillSummary] = []
        for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
            if not skill_file.is_file():
                continue
            try:
                raw_text = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if not raw_text.strip():
                continue
            meta, body = _parse_frontmatter(raw_text)
            folder_name = skill_file.parent.name
            name = meta.get("name") or folder_name
            description = meta.get("description") or _first_summary_line(body) or f"本地技能 {name}"
            results.append(
                SkillSummary(
                    id=f"skill:{folder_name}",
                    name=name,
                    description=description,
                    source_path=self._display_path(skill_file),
                )
            )
        return results

    async def read_skill_content(
        self,
        name: str,
        *,
        user_id: str = "default",
    ) -> tuple[str, str] | None:
        target = name.strip().lower()
        if not target or "/" in target or "\\" in target or ".." in target:
            return None

        skills_dir = self._user_root(user_id) / "skills"
        if not skills_dir.exists() or not skills_dir.is_dir():
            return None

        for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
            if not skill_file.is_file():
                continue
            try:
                raw_text = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, _ = _parse_frontmatter(raw_text)
            folder_name = skill_file.parent.name
            declared_name = (meta.get("name") or folder_name).strip()
            if declared_name.lower() == target or folder_name.lower() == target:
                return self._display_path(skill_file), raw_text
        return None
