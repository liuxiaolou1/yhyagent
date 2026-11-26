"""领域层（Domain Layer）：纯业务模型与规则，零外部框架/IO 依赖。"""

from .context import AgentContext, ContextUsageReport, estimate_tokens
from .customization import (
    CustomizationBudgetReport,
    CustomizationCategory,
    CustomizationCategorySummary,
    CustomizationGovernor,
    CustomizationItem,
    CustomizationItemReport,
    CustomizationStatus,
    GovernedCustomizations,
    RuleEntry,
    SkillSummary,
)
from .guard import GuardCheckResult, LoopGuard
from .memory import MemoryEntry, MemoryScope
from .message import Message, ModelResponse, Role, ToolCall, ToolResult
from .session import ContextSummary, Session, SessionSnapshot, SideChat
from .tool import ReviewDecision, Tool, ToolRiskLevel, ToolSpec

__all__ = [
    "Role",
    "ToolCall",
    "ToolResult",
    "Message",
    "ModelResponse",
    "ToolRiskLevel",
    "ToolSpec",
    "Tool",
    "ReviewDecision",
    "MemoryScope",
    "MemoryEntry",
    "ContextSummary",
    "Session",
    "SessionSnapshot",
    "SideChat",
    "AgentContext",
    "ContextUsageReport",
    "estimate_tokens",
    "CustomizationCategory",
    "CustomizationStatus",
    "RuleEntry",
    "SkillSummary",
    "CustomizationItem",
    "CustomizationItemReport",
    "CustomizationCategorySummary",
    "CustomizationBudgetReport",
    "GovernedCustomizations",
    "CustomizationGovernor",
    "GuardCheckResult",
    "LoopGuard",
]
