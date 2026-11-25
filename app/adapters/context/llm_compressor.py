"""上下文滚动摘要压缩器：支持真实模型增量摘要与零成本离线规则降级摘要。"""

from __future__ import annotations

import json

from app.application.ports import ModelClient
from app.domain import AgentContext, Message, Role

_SUMMARY_INSTRUCTION = (
    "你是对话上下文压缩助手。请将「已有摘要」和「新增早期对话」合并成一份简洁、高密度的中文滚动摘要，"
    "以便主助手在后续轮次无缝衔接。\n"
    "要求：\n"
    "1. 只保留客观事实，不编造信息；\n"
    "2. 必须保留关键标识（如订单号、文件路径、配置项、用户名、数字结果与明确约束）；\n"
    "3. 按以下 3 个小节输出，总长度尽量控制在 400 字以内：\n"
    "   - 目标与约束\n"
    "   - 已完成操作与关键结果\n"
    "   - 待办或未解决事项"
)


def _format_transcript(messages: list[Message], *, max_item_chars: int = 500) -> str:
    lines: list[str] = []
    for msg in messages:
        if msg.role == Role.USER:
            lines.append(f"[用户] {msg.content[:max_item_chars]}")
        elif msg.role == Role.ASSISTANT:
            if msg.tool_calls:
                calls = ", ".join(
                    f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:120]})"
                    for tc in msg.tool_calls
                )
                text = f"{msg.content[:200]} " if msg.content else ""
                lines.append(f"[助手] {text}调用工具: {calls}")
            elif msg.content:
                lines.append(f"[助手] {msg.content[:max_item_chars]}")
        elif msg.role == Role.TOOL:
            tool_name = msg.tool_name or "tool"
            lines.append(f"[工具:{tool_name}] {msg.content[:max_item_chars]}")
    return "\n".join(lines)


class LLMContextCompressor:
    """调用 ModelClient 生成增量滚动摘要。"""

    def __init__(self, model_client: ModelClient) -> None:
        self.model_client = model_client

    async def summarize(
        self,
        *,
        previous_summary: str,
        messages: list[Message],
    ) -> str:
        transcript = _format_transcript(messages)
        if not transcript.strip():
            return previous_summary

        prompt_parts: list[str] = []
        if previous_summary.strip():
            prompt_parts.append(f"【已有摘要】\n{previous_summary.strip()}")
        prompt_parts.append(f"【新增早期对话】\n{transcript}")
        prompt_parts.append("请输出合并后的最新滚动摘要：")

        ctx = AgentContext(
            session_id="context_compressor",
            instruction=_SUMMARY_INSTRUCTION,
            messages=[Message.user("\n\n".join(prompt_parts))],
            tools=[],
            cycle=1,
            max_cycles=1,
        )
        response = await self.model_client.generate(ctx)
        return response.content.strip() or previous_summary


class HeuristicContextCompressor:
    """无 LLM 依赖的本地规则滚动摘要器（用于 --offline 模式或单元测试）。"""

    def __init__(self, *, max_summary_chars: int = 1200) -> None:
        self.max_summary_chars = max_summary_chars

    async def summarize(
        self,
        *,
        previous_summary: str,
        messages: list[Message],
    ) -> str:
        lines: list[str] = []
        if previous_summary.strip():
            lines.append(previous_summary.strip())

        for msg in messages:
            if msg.role == Role.USER and msg.content.strip():
                lines.append(f"- 用户需求：{msg.content.strip()[:120]}")
            elif msg.role == Role.TOOL and msg.content.strip():
                tool_name = msg.tool_name or "tool"
                lines.append(f"- 工具 `{tool_name}` 结果：{msg.content.strip()[:120]}")
            elif msg.role == Role.ASSISTANT and msg.content.strip() and not msg.tool_calls:
                lines.append(f"- 助手结论：{msg.content.strip()[:120]}")

        combined = "\n".join(lines)
        if len(combined) > self.max_summary_chars:
            return combined[-self.max_summary_chars :]
        return combined
