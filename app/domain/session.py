"""领域会话模型：管理持续累积的消息历史与快照回滚能力。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .message import Message


class SessionSnapshot(BaseModel):
    """会话某一时刻的轻量快照（记录消息截断点与备注，用于失败回滚）。"""

    id: str
    label: str
    message_count: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContextSummary(BaseModel):
    """上下文压缩滚动摘要检查点（不删除原始消息，支持与快照回滚联动）。"""

    covered_until: int
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Session(BaseModel):
    """会话聚合根：承载对话历史、快照点、滚动摘要与跨轮次元数据。"""

    id: str = Field(default_factory=lambda: f"sess_{uuid4().hex[:10]}")
    user_id: str = "default"
    title: str = ""
    messages: list[Message] = Field(default_factory=list)
    snapshots: list[SessionSnapshot] = Field(default_factory=list)
    summaries: list[ContextSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def latest_summary(self) -> ContextSummary | None:
        return self.summaries[-1] if self.summaries else None

    @property
    def summarized_until(self) -> int:
        return self.summaries[-1].covered_until if self.summaries else 0

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def add_summary(self, covered_until: int, content: str) -> ContextSummary:
        summary = ContextSummary(covered_until=covered_until, content=content)
        self.summaries.append(summary)
        self.updated_at = datetime.now(timezone.utc)
        return summary

    def create_snapshot(self, label: str = "checkpoint") -> SessionSnapshot:
        snap = SessionSnapshot(
            id=f"cp_{len(self.snapshots) + 1:03d}",
            label=label,
            message_count=len(self.messages),
        )
        self.snapshots.append(snap)
        self.updated_at = datetime.now(timezone.utc)
        return snap

    def rollback_to(self, snapshot_id: str) -> SessionSnapshot:
        """将消息历史回滚至指定快照位置，并自动作废超出截断点的滚动摘要。"""
        target = next((s for s in self.snapshots if s.id == snapshot_id), None)
        if target is None:
            raise KeyError(f"未找到快照: {snapshot_id}")
        self.messages = self.messages[: target.message_count]
        self.summaries = [s for s in self.summaries if s.covered_until <= target.message_count]
        self.updated_at = datetime.now(timezone.utc)
        return target


class SideChat(BaseModel):
    """从主会话分叉出的独立侧边问答分支（只读可见主会话历史，单独落盘，绝不写回主会话）。"""

    id: str = Field(default_factory=lambda: f"side_{uuid4().hex[:10]}")
    parent_session_id: str
    user_id: str = "default"
    title: str = ""
    quote: str = ""
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
