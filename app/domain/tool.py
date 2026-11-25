"""领域工具定义与审核结果模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolRiskLevel(StrEnum):
    """工具风险等级，供工具审核器决定自动放行、规则拦截还是人工审批。"""

    SAFE = "safe"              # 只读、无副作用
    MODERATE = "moderate"      # 外部请求或可逆写入
    DANGEROUS = "dangerous"    # 不可逆写操作 / 删除 / 资金敏感


class ToolSpec(BaseModel):
    """暴露给模型和审核器的工具元数据（JSON Schema 格式）。"""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    risk_level: ToolRiskLevel = ToolRiskLevel.SAFE


class ReviewDecision(BaseModel):
    """系统对一次 ToolCall 的审核判决。"""

    approved: bool
    reason: str = ""
    reviewer: str = "policy"


class Tool(ABC):
    """领域层工具抽象，具体实现放在 adapters/tools/。"""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """返回工具规格说明。"""

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> str:
        """执行工具并返回字符串形式的结果。"""
