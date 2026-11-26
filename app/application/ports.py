"""应用层端口接口（Ports）：通过依赖反转隔离模型 API、存储介质、长期记忆、上下文压缩与工具审核策略。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.domain import (
    AgentContext,
    MemoryEntry,
    MemoryScope,
    Message,
    ModelResponse,
    ReviewDecision,
    RuleEntry,
    Session,
    SideChat,
    SkillSummary,
    ToolCall,
    ToolSpec,
)


class ModelClient(Protocol):
    """大模型访问端口：屏蔽各家厂商协议差异。"""

    async def generate(self, context: AgentContext) -> ModelResponse: ...


class SessionStore(Protocol):
    """会话持久化端口：支持在运行过程中持续保存消息、滚动摘要与回滚快照。"""

    async def get(self, session_id: str) -> Session | None: ...

    async def save(self, session: Session) -> None: ...

    async def list_sessions(self) -> list[Session]: ...


class SideChatStore(Protocol):
    """侧边问答分支持久化端口：与主会话隔离存储，绝不污染主会话。"""

    async def get(self, side_id: str) -> SideChat | None: ...

    async def save(self, side_chat: SideChat) -> None: ...

    async def list_by_parent(self, parent_session_id: str) -> list[SideChat]: ...

    async def delete(self, side_id: str) -> bool: ...


class MemoryStore(Protocol):
    """长期记忆端口：按用户与日期记录长期偏好和每日笔记，支持上下文加载与检索。"""

    async def remember(
        self,
        content: str,
        *,
        user_id: str = "default",
        scope: MemoryScope = MemoryScope.DAILY,
        recorded_at: datetime | None = None,
    ) -> MemoryEntry: ...

    async def load_context(
        self,
        *,
        user_id: str = "default",
        recent_days: int = 2,
        max_chars: int = 3000,
    ) -> str: ...

    async def search(
        self,
        query: str,
        *,
        user_id: str = "default",
        limit: int = 10,
    ) -> list[MemoryEntry]: ...


class ContextCompressor(Protocol):
    """上下文压缩端口：将早期对话与已有摘要合并为增量滚动摘要。"""

    async def summarize(
        self,
        *,
        previous_summary: str,
        messages: list[Message],
    ) -> str: ...


class CustomizationStore(Protocol):
    """本地规则与技能扩展读取端口（支持渐进式披露与按需加载完整 SKILL.md）。"""

    async def load_rules(self, *, user_id: str = "default") -> list[RuleEntry]: ...

    async def load_skills(self, *, user_id: str = "default") -> list[SkillSummary]: ...

    async def read_skill_content(
        self,
        name: str,
        *,
        user_id: str = "default",
    ) -> tuple[str, str] | None: ...


class ToolReviewer(Protocol):
    """工具审核端口：在执行任何工具前做规则校验、风控模型审核或人工确认。"""

    async def review(
        self,
        session: Session,
        tool_spec: ToolSpec,
        tool_call: ToolCall,
    ) -> ReviewDecision: ...
