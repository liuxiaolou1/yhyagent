"""领域长期记忆模型：定义跨会话的用户记忆范围与条目结构。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryScope(StrEnum):
    """记忆落盘范围：profile 存长期画像 (MEMORY.md)，daily 存按日期记录 (YYYY/MM/DD.md)。"""

    PROFILE = "profile"
    DAILY = "daily"


class MemoryEntry(BaseModel):
    """单条长期记忆记录。"""

    user_id: str = "default"
    scope: MemoryScope = MemoryScope.DAILY
    content: str
    source: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
