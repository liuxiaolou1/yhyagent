"""领域消息类型：统一模型输入输出与工具调用协议，屏蔽各家厂商字段差异。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """模型发起的一次工具调用请求。"""

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"name": self.name, "arguments": self.arguments},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ToolResult(BaseModel):
    """工具执行结果或审核拒绝原因。"""

    tool_call_id: str
    tool_name: str
    success: bool
    approved: bool = True
    output: str = ""
    error: str | None = None

    def to_model_content(self) -> str:
        """序列化为回传给模型的工具消息正文。"""
        if not self.approved:
            return f"[REJECTED] 工具 `{self.tool_name}` 调用被审核拒绝：{self.error or '未获批准'}。请根据拒绝原因调整方案或向用户说明。"
        if not self.success:
            return f"[ERROR] 工具 `{self.tool_name}` 执行失败：{self.error or '未知错误'}"
        return self.output


class Message(BaseModel):
    """会话中的标准消息实体。"""

    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:12]}")
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    model_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        *,
        model_name: str | None = None,
    ) -> Message:
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
            model_name=model_name,
        )

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        return cls(
            role=Role.TOOL,
            content=result.to_model_content(),
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
        )


class ModelResponse(BaseModel):
    """ModelClient 返回给应用层的标准响应。"""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model_name: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
