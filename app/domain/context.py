"""领域上下文模型：封装单次模型调用所需的指令、长期记忆、滚动摘要、历史消息与可用工具。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .message import Message, Role
from .tool import ToolSpec


def _fold_text(text: str, max_chars: int) -> str:
    """对过长的历史工具输出保留头尾并折叠中间内容，降低 token 消耗。"""
    if len(text) <= max_chars or max_chars < 40:
        return text
    head_len = int(max_chars * 0.7)
    tail_len = max(20, max_chars - head_len - 32)
    omitted = len(text) - head_len - tail_len
    if omitted <= 0:
        return text
    return f"{text[:head_len]}\n...[已折叠 {omitted} 字符]...\n{text[-tail_len:]}"


def estimate_tokens(text: str) -> int:
    """轻量启发式 token 估算（无外部 tokenizer 依赖；非 ASCII 字符按 ~1 token，ASCII 按 ~4 字符/token）。"""
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = len(text) - non_ascii
    return non_ascii + (ascii_chars + 3) // 4


class ContextUsageReport(BaseModel):
    """单个会话的上下文使用量、压缩进度与真实 token 统计快照。"""

    session_id: str
    user_id: str = "default"
    model_name: str = ""

    # 消息与摘要计数
    total_messages: int = 0
    summarized_until: int = 0
    active_messages: int = 0
    windowed_messages: int = 0
    summary_count: int = 0

    # 下次发送给模型的上下文字符分布
    instruction_chars: int = 0
    memory_chars: int = 0
    summary_chars: int = 0
    active_messages_chars: int = 0
    folded_messages_chars: int = 0
    total_context_chars: int = 0

    # Token 估算 + 真实上报用量
    estimated_context_tokens: int = 0
    context_window_tokens: int | None = None
    context_window_ratio: float | None = None
    last_prompt_tokens: int | None = None
    last_completion_tokens: int | None = None
    last_total_tokens: int | None = None
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    cumulative_total_tokens: int = 0
    model_calls: int = 0
    usage_source: str = "estimated"  # "reported" | "estimated"

    # 滚动摘要压缩阈值与进度
    compress_trigger_messages: int = 24
    compress_trigger_chars: int = 12000
    compress_keep_recent: int = 10
    compress_message_ratio: float = 0.0
    compress_char_ratio: float = 0.0
    will_compress_next_turn: bool = False


class AgentContext(BaseModel):
    """后端在每一轮循环中交给 ModelClient 的完整上下文。"""

    session_id: str
    user_id: str = "default"
    provider: str | None = None
    model: str | None = None
    instruction: str
    summary: str = ""
    memory_context: str = ""
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    cycle: int = 1
    max_cycles: int = 12
    max_history_messages: int = 40
    max_old_tool_chars: int = 600
    keep_recent_tool_messages: int = 2

    def composed_instruction(self) -> str:
        """拼装系统指令、跨会话长期记忆以及本会话早期滚动摘要。"""
        sections: list[str] = []
        if self.instruction.strip():
            sections.append(self.instruction.strip())
        if self.memory_context.strip():
            sections.append(
                "## 用户长期记忆（仅供参考，若与用户当前明确要求冲突，以当前要求为准）\n"
                f"{self.memory_context.strip()}"
            )
        if self.summary.strip():
            sections.append(f"## 本会话早期对话摘要\n{self.summary.strip()}")
        return "\n\n".join(sections)

    def windowed_messages(self) -> list[Message]:
        """按窗口裁剪历史消息、对齐 tool 边界，并折叠早于最近 N 条的超长工具输出。"""
        if len(self.messages) <= self.max_history_messages:
            sliced = list(self.messages)
        else:
            sliced = list(self.messages[-self.max_history_messages :])
            while sliced and sliced[0].role == Role.TOOL:
                sliced.pop(0)

        tool_indices = [idx for idx, msg in enumerate(sliced) if msg.role == Role.TOOL]
        if len(tool_indices) <= self.keep_recent_tool_messages:
            return sliced

        foldable = set(tool_indices[: -self.keep_recent_tool_messages])
        result: list[Message] = []
        for idx, msg in enumerate(sliced):
            if idx in foldable and len(msg.content) > self.max_old_tool_chars:
                result.append(
                    msg.model_copy(update={"content": _fold_text(msg.content, self.max_old_tool_chars)})
                )
            else:
                result.append(msg)
        return result
