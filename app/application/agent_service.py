"""应用层核心编排服务：AgentService。

核心运行流程：
1. 加载或创建会话 -> 记录本轮快照 -> 追加用户消息并立即落盘。
2. 进入决策循环（受 max_cycles 上限保护）：
   - 按需加载该用户的长期记忆，并对过长历史触发滚动摘要压缩并落盘。
   - 组装上下文（指令 + 长期记忆 + 滚动摘要 + 近期消息 + 可用工具）交给 ModelClient。
   - 追加助手回复（含可能的 tool_calls）并立即落盘。
   - 若模型无工具调用 -> 返回最终回复，本轮结束。
   - 若模型提出工具调用 -> 依次经过 LoopGuard 防死循环检查 + ToolReviewer 审核：
     * 审核通过 -> 执行工具，将结果写回消息并落盘。
     * 审核拒绝 -> 不执行工具，将拒绝原因写回消息并落盘，交回模型重新决策。
3. 达到循环上限时安全收尾并持久化，确保会话随时可续聊或回滚。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Iterable

from pydantic import BaseModel, Field

from app.domain import (
    AgentContext,
    ContextUsageReport,
    CustomizationBudgetReport,
    CustomizationCategory,
    CustomizationGovernor,
    CustomizationItem,
    GovernedCustomizations,
    LoopGuard,
    Message,
    Role,
    Session,
    SessionSnapshot,
    SideChat,
    Tool,
    ToolCall,
    ToolResult,
    ToolRiskLevel,
    estimate_tokens,
)

from .ports import (
    ContextCompressor,
    CustomizationStore,
    MemoryStore,
    ModelClient,
    SessionStore,
    SideChatStore,
    ToolReviewer,
)

logger = logging.getLogger(__name__)


def _infer_context_window(model_name: str) -> int | None:
    """按常见模型家族给出参考上下文窗口大小（仅作占比展示参考）。"""
    lower = model_name.strip().lower()
    if not lower or "offline" in lower:
        return None
    if "gemini" in lower:
        return 1_000_000
    if any(key in lower for key in ("qwen", "deepseek", "glm", "gpt-4o", "gpt-5", "minimax", "kimi")):
        return 128_000
    return 128_000


def _record_session_usage(session: Session, usage: dict[str, int]) -> None:
    """将上游真实返回的 usage 记录到 session.metadata 中，便于跨重启查询累计消耗。"""
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    cum = session.metadata.get("cumulative_usage")
    if not isinstance(cum, dict):
        cum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    cum = {
        "prompt_tokens": int(cum.get("prompt_tokens", 0)) + int(prompt_tokens or 0),
        "completion_tokens": int(cum.get("completion_tokens", 0)) + int(completion_tokens or 0),
        "total_tokens": int(cum.get("total_tokens", 0)) + int(total_tokens or 0),
        "calls": int(cum.get("calls", 0)) + 1,
    }
    session.metadata["cumulative_usage"] = cum

    if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
        session.metadata["last_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

DEFAULT_INSTRUCTION = (
    "你是一个严谨、可靠的任务执行助手。\n"
    "1. 工具使用原则：当需要外部信息或执行操作时，必须通过真实工具调用完成，严禁口头编造执行结果。\n"
    "2. 长期记忆规则（严格遵守）：\n"
    "   - 只要用户告知自己的姓名、称呼、身份、长期偏好、常用配置，或明确要求“记住某件事”，"
    "必须立即实际发起 `remember` 工具调用（姓名/偏好/固定信息用 `scope=\"profile\"`，当天事件或阶段性结论用 `scope=\"daily\"`）；\n"
    "   - 严禁在未真正调用 `remember` 工具并收到成功返回前，口头声称「已记录」「已保存到长期记忆 / MEMORY.md」；\n"
    "   - 当用户询问过往记录或个人信息且当前上下文未体现时，应先调用 `search_memory` 工具检索。\n"
    "3. 审核拒绝处理：如果某次工具调用被系统审核拒绝，请认真阅读拒绝原因，调整方案或向用户说明情况，不要盲目重复同样的调用。"
)


DEFAULT_SIDE_CHAT_BLOCKED_TOOLS: frozenset[str] = frozenset({"remember", "delete_order"})

SIDE_CHAT_INSTRUCTION = (
    "你正在右侧「侧边追问（Side Question / /btw）」独立分支中与用户讨论。\n"
    "1. 上下文隔离规则（严格遵守）：\n"
    "   - 你可以看到上方主对话的完整历史与摘要作为背景参考；\n"
    "   - 本侧边分支单独保存，**绝不会写回或修改主对话历史**；\n"
    "   - 本侧边分支为只读讨论模式，已物理禁用 `remember` 及所有高危写操作工具，**不会也不允许写入长期记忆**。\n"
    "2. 工具使用：如有需要，你可以使用当前允许的只读工具（例如 `search_memory` 检索过往笔记、`query_order`、`web_search` 等）辅助回答。\n"
    "3. 回答风格：紧扣用户引用片段和追问，简洁、准确、直接给出结论。"
)


class ToolEvent(BaseModel):
    """单次工具调用及审核审计记录（方便入口层展示或记录 Trace）。"""

    cycle: int
    tool_name: str
    arguments: dict
    approved: bool
    success: bool
    summary: str


class TurnResult(BaseModel):
    """单轮对话执行完后的结构化返回值。"""

    session_id: str
    reply: str
    cycles: int
    stopped_reason: str  # "completed" | "max_cycles"
    snapshot_id: str
    model_name: str = ""
    tool_events: list[ToolEvent] = Field(default_factory=list)


class SideTurnResult(BaseModel):
    """侧边追问单轮执行完后的结构化返回值。"""

    side_id: str
    parent_session_id: str
    reply: str
    cycles: int
    stopped_reason: str  # "completed" | "max_cycles"
    model_name: str = ""
    tool_events: list[ToolEvent] = Field(default_factory=list)


class AgentService:
    """应用层编排服务：对外作为 CLI / Web 入口的唯一调用门面。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        session_store: SessionStore,
        tool_reviewer: ToolReviewer,
        tools: Iterable[Tool] = (),
        instruction: str = DEFAULT_INSTRUCTION,
        user_id: str = "default",
        memory_store: MemoryStore | None = None,
        customization_store: CustomizationStore | None = None,
        customization_token_budget: int = 20000,
        side_chat_store: SideChatStore | None = None,
        side_chat_blocked_tools: Iterable[str] | None = None,
        context_compressor: ContextCompressor | None = None,
        max_cycles: int = 8,
        max_repeat_calls: int = 2,
        max_history_messages: int = 40,
        compress_trigger_messages: int = 24,
        compress_trigger_chars: int = 12000,
        compress_keep_recent: int = 10,
        memory_recent_days: int = 2,
    ) -> None:
        self.model_client = model_client
        self.session_store = session_store
        self.tool_reviewer = tool_reviewer
        self._tools: dict[str, Tool] = {t.spec.name: t for t in tools}
        self.instruction = instruction
        self.user_id = user_id
        self.memory_store = memory_store
        self.customization_store = customization_store
        self.customization_token_budget = max(100, customization_token_budget)
        self.side_chat_store = side_chat_store
        self.side_chat_blocked_tools: set[str] = (
            set(side_chat_blocked_tools)
            if side_chat_blocked_tools is not None
            else set(DEFAULT_SIDE_CHAT_BLOCKED_TOOLS)
        )
        self.context_compressor = context_compressor
        self.max_cycles = max_cycles
        self.max_repeat_calls = max_repeat_calls
        self.max_history_messages = max_history_messages
        self.compress_trigger_messages = compress_trigger_messages
        self.compress_trigger_chars = compress_trigger_chars
        self.compress_keep_recent = compress_keep_recent
        self.memory_recent_days = memory_recent_days

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        title: str = "",
        user_id: str | None = None,
    ) -> Session:
        owner = user_id or self.user_id
        session = (
            Session(id=session_id, user_id=owner, title=title)
            if session_id
            else Session(user_id=owner, title=title)
        )
        await self.session_store.save(session)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        return await self.session_store.get(session_id)

    async def list_sessions(self) -> list[Session]:
        return await self.session_store.list_sessions()

    def side_chat_allowed_tools(self) -> dict[str, Tool]:
        """侧边追问允许的只读工具集合：物理级排除 remember 与所有 DANGEROUS 高危工具。"""
        return {
            name: tool
            for name, tool in self._tools.items()
            if name not in self.side_chat_blocked_tools and tool.spec.risk_level != ToolRiskLevel.DANGEROUS
        }

    @staticmethod
    def _split_memory_sections(memory_context: str) -> list[tuple[str, str]]:
        """把 memory_context 按 `### 标题` 拆成逐项条目，方便预算面板展示明细。"""
        text = memory_context.strip()
        if not text:
            return []
        sections: list[tuple[str, str]] = []
        current_title = "长期记忆"
        current_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("### "):
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = line[4:].strip() or "长期记忆"
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))
        return [(title, body) for title, body in sections if body]

    @staticmethod
    def _compose_governed_instruction(base_instruction: str, governed: GovernedCustomizations) -> str:
        sections: list[str] = []
        if base_instruction.strip():
            sections.append(base_instruction.strip())
        if governed.rules_text.strip():
            sections.append(f"## 本地规则\n{governed.rules_text.strip()}")
        if governed.skills_text.strip():
            sections.append(
                "## 可用技能（渐进式披露：仅在任务匹配时调用 `load_skill` 工具读取完整 SKILL.md）\n"
                f"{governed.skills_text.strip()}"
            )
        return "\n\n".join(sections)

    async def _govern_customizations(
        self,
        *,
        user_id: str,
        tools_map: dict[str, Tool],
    ) -> GovernedCustomizations:
        """收集 memory / rules / skills / builtin_tools / mcp_tools，经过预算治理器裁剪与降级。"""
        memory_context = await self._load_memory_context(user_id)
        rules = []
        skills = []
        if self.customization_store is not None:
            try:
                rules = await self.customization_store.load_rules(user_id=user_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Customizations] 读取本地规则失败，已忽略: %s", exc)
            try:
                skills = await self.customization_store.load_skills(user_id=user_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Customizations] 读取本地技能失败，已忽略: %s", exc)

        items: list[CustomizationItem] = []

        # 1. 长期记忆明细
        for idx, (title, body) in enumerate(self._split_memory_sections(memory_context), start=1):
            items.append(
                CustomizationItem(
                    id=f"memory:{idx}",
                    name=title,
                    category=CustomizationCategory.MEMORY,
                    content=body,
                    source_path=f"{user_id}/memory",
                )
            )

        # 2. 本地规则（支持超预算降级为路径指针）
        for rule in rules:
            items.append(
                CustomizationItem(
                    id=rule.id,
                    name=rule.name,
                    category=CustomizationCategory.RULES,
                    content=f"### 规则：{rule.name}\n{rule.content}",
                    description=rule.description,
                    source_path=rule.source_path,
                )
            )

        # 3. 本地技能（渐进式披露：仅摘要 name + description）
        for skill in skills:
            items.append(
                CustomizationItem(
                    id=skill.id,
                    name=skill.name,
                    category=CustomizationCategory.SKILLS,
                    content=f"- `{skill.name}`: {skill.description}（按需调用 `load_skill` 读取完整说明）",
                    description=skill.description,
                    source_path=skill.source_path,
                )
            )

        # 4. 工具（无本地 skills 时不暴露 load_skill，避免浪费 token）
        for name, tool in tools_map.items():
            if name == "load_skill" and not skills:
                continue
            spec = tool.spec
            is_mcp = name.startswith("mcp_") or bool(getattr(tool, "is_mcp", False))
            schema_text = json.dumps(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
                ensure_ascii=False,
            )
            items.append(
                CustomizationItem(
                    id=f"tool:{spec.name}",
                    name=spec.name,
                    category=CustomizationCategory.MCP_TOOLS if is_mcp else CustomizationCategory.BUILTIN_TOOLS,
                    content=schema_text,
                    description=spec.description,
                    source_path="mcp" if is_mcp else "builtin",
                )
            )

        governor = CustomizationGovernor(budget_tokens=self.customization_token_budget)
        return governor.govern(items, user_id=user_id)

    async def get_customization_usage(
        self,
        *,
        user_id: str | None = None,
    ) -> CustomizationBudgetReport:
        """返回当前用户的 Customizations Token 预算汇总与逐项明细报告。"""
        owner = user_id or self.user_id
        governed = await self._govern_customizations(user_id=owner, tools_map=self._tools)
        return governed.report

    async def get_side_chat(self, side_id: str) -> SideChat | None:
        if self.side_chat_store is None:
            return None
        return await self.side_chat_store.get(side_id)

    async def list_side_chats(self, parent_session_id: str) -> list[SideChat]:
        if self.side_chat_store is None:
            return []
        return await self.side_chat_store.list_by_parent(parent_session_id)

    async def delete_side_chat(self, side_id: str) -> bool:
        if self.side_chat_store is None:
            return False
        return await self.side_chat_store.delete(side_id)

    async def _save_side_chat(self, side_chat: SideChat) -> None:
        if self.side_chat_store is not None:
            await self.side_chat_store.save(side_chat)

    async def run_side_turn(
        self,
        question: str,
        *,
        parent_session_id: str | None = None,
        side_id: str | None = None,
        quote: str = "",
        seed_history: list[tuple[str, str]] = (),
        user_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> SideTurnResult:
        """非流式侧边追问入口：复用 run_side_turn_stream，保证只读工具与独立分支落盘一致。"""
        done_payload: dict[str, Any] | None = None
        async for ev in self.run_side_turn_stream(
            question,
            parent_session_id=parent_session_id,
            side_id=side_id,
            quote=quote,
            seed_history=seed_history,
            user_id=user_id,
            provider=provider,
            model=model,
        ):
            if ev["event"] == "done":
                done_payload = ev["data"]
        if done_payload is None:
            raise RuntimeError("侧边追问未返回完成事件")
        return SideTurnResult.model_validate(done_payload)

    async def run_side_turn_stream(
        self,
        question: str,
        *,
        parent_session_id: str | None = None,
        side_id: str | None = None,
        quote: str = "",
        seed_history: list[tuple[str, str]] = (),
        user_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式执行侧边追问分支（仿 Antigravity /btw）：
        - 只读可见主会话历史与摘要，绝不调用 session_store.save(parent)；
        - 物理级屏蔽 remember 与高危工具，仅保留只读工具；
        - 侧边记录单独落盘到 SideChatStore（.data/<user>/side_chats/YYYY/MM/DD/<side_id>.json）。
        """
        parent_id = (parent_session_id or "").strip() or "standalone"
        parent = await self.session_store.get(parent_id) if parent_session_id else None
        owner = user_id or (parent.user_id if parent else self.user_id)

        side_chat = await self.side_chat_store.get(side_id) if (side_id and self.side_chat_store) else None
        if side_chat is not None:
            if parent_session_id and side_chat.parent_session_id != parent_id:
                raise ValueError(f"侧边分支 {side_chat.id} 不属于主会话 {parent_id}")
            if quote.strip():
                side_chat.quote = quote.strip()[:2000]
            if not side_chat.title:
                side_chat.title = question[:40]
        else:
            side_chat = SideChat(
                id=side_id or SideChat(parent_session_id=parent_id).id,
                parent_session_id=parent_id,
                user_id=owner,
                title=question[:40],
                quote=quote.strip()[:2000],
            )
            for role_name, content_text in seed_history[-20:]:
                clean_text = content_text.strip()
                if not clean_text:
                    continue
                side_chat.append(Message.user(clean_text) if role_name == "user" else Message.assistant(clean_text))

        side_chat.append(Message.user(question))
        await self._save_side_chat(side_chat)

        yield {
            "event": "session",
            "data": {"side_id": side_chat.id, "parent_session_id": side_chat.parent_session_id},
        }

        allowed_tools = self.side_chat_allowed_tools()
        tool_specs = [t.spec for t in allowed_tools.values()]
        guard = LoopGuard(max_repeat_calls=self.max_repeat_calls)
        tool_events: list[ToolEvent] = []
        last_model_name = ""

        # 只读提取主对话历史与滚动摘要（绝不修改 parent）
        parent_summary = parent.latest_summary.content.strip() if (parent and parent.latest_summary) else ""
        parent_messages = list(parent.messages[parent.summarized_until :]) if parent else []

        instruction_parts = [SIDE_CHAT_INSTRUCTION]
        if side_chat.quote:
            instruction_parts.append(f"## 用户引用的片段\n> {side_chat.quote}")
        side_instruction = "\n\n".join(instruction_parts)

        # 审核器需要的轻量只读 Session 视图（仅用于审核上下文，永不落盘到 SessionStore）
        review_session = Session(id=side_chat.id, user_id=side_chat.user_id, title=side_chat.title)

        for cycle in range(1, self.max_cycles + 1):
            governed = await self._govern_customizations(user_id=side_chat.user_id, tools_map=allowed_tools)
            effective_side_instruction = self._compose_governed_instruction(side_instruction, governed)
            governed_tool_specs = [
                allowed_tools[name].spec for name in governed.active_tool_names if name in allowed_tools
            ]
            combined_messages = [*parent_messages, *side_chat.messages]

            logger.info(
                "[SideChat %s (parent=%s) | Cycle %d/%d] 流式请求模型决策 (provider=%s, model=%s)...",
                side_chat.id,
                side_chat.parent_session_id,
                cycle,
                self.max_cycles,
                provider or "default",
                model or "default",
            )
            context = AgentContext(
                session_id=side_chat.id,
                user_id=side_chat.user_id,
                provider=provider,
                model=model,
                instruction=effective_side_instruction,
                summary=parent_summary,
                memory_context=governed.memory_text,
                messages=combined_messages,
                tools=governed_tool_specs,
                cycle=cycle,
                max_cycles=self.max_cycles,
                max_history_messages=self.max_history_messages,
            )

            stream_fn = getattr(self.model_client, "generate_stream", None)
            if callable(stream_fn):
                queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

                async def _on_delta(chunk: str) -> None:
                    await queue.put(("delta", chunk))

                async def _produce() -> None:
                    try:
                        resp = await stream_fn(context, on_delta=_on_delta)
                        await queue.put(("response", resp))
                    except BaseException as exc:  # noqa: BLE001
                        await queue.put(("error", exc))

                producer = asyncio.create_task(_produce())
                response = None
                try:
                    while True:
                        kind, payload = await queue.get()
                        if kind == "delta":
                            yield {"event": "delta", "data": {"cycle": cycle, "content": payload}}
                        elif kind == "response":
                            response = payload
                            break
                        elif kind == "error":
                            raise payload
                finally:
                    if not producer.done():
                        producer.cancel()
                        try:
                            await producer
                        except asyncio.CancelledError:
                            pass
            else:
                response = await self.model_client.generate(context)
                if response.content and not response.tool_calls:
                    yield {"event": "delta", "data": {"cycle": cycle, "content": response.content}}

            assert response is not None
            if response.model_name:
                last_model_name = response.model_name
                side_chat.metadata["last_model"] = response.model_name

            side_chat.append(
                Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    model_name=response.model_name or None,
                )
            )
            await self._save_side_chat(side_chat)

            if not response.tool_calls:
                result = SideTurnResult(
                    side_id=side_chat.id,
                    parent_session_id=side_chat.parent_session_id,
                    reply=response.content,
                    cycles=cycle,
                    stopped_reason="completed",
                    model_name=last_model_name,
                    tool_events=tool_events,
                )
                yield {"event": "done", "data": result.model_dump(mode="json")}
                return

            for tool_call in response.tool_calls:
                tool_result = await self._handle_tool_call(
                    review_session,
                    tool_call,
                    guard,
                    allowed_tools=allowed_tools,
                )
                side_chat.append(Message.tool(tool_result))
                await self._save_side_chat(side_chat)

                ev = ToolEvent(
                    cycle=cycle,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    approved=tool_result.approved,
                    success=tool_result.success,
                    summary=tool_result.output if tool_result.success else (tool_result.error or ""),
                )
                tool_events.append(ev)
                yield {"event": "tool", "data": ev.model_dump(mode="json")}

        limit_reply = f"侧边追问已达到最大循环次数上限（{self.max_cycles} 轮），自动结束本轮。"
        side_chat.append(Message.assistant(content=limit_reply, model_name=last_model_name or None))
        await self._save_side_chat(side_chat)
        yield {"event": "delta", "data": {"cycle": self.max_cycles, "content": limit_reply}}
        result = SideTurnResult(
            side_id=side_chat.id,
            parent_session_id=side_chat.parent_session_id,
            reply=limit_reply,
            cycles=self.max_cycles,
            stopped_reason="max_cycles",
            model_name=last_model_name,
            tool_events=tool_events,
        )
        yield {"event": "done", "data": result.model_dump(mode="json")}

    async def get_context_usage(
        self,
        session_id: str,
        *,
        model: str | None = None,
    ) -> ContextUsageReport:
        """计算指定会话当前的上下文占用、旧工具折叠效果、压缩进度与真实 token 统计。"""
        session = await self.session_store.get(session_id)
        if session is None:
            raise KeyError(f"会话不存在: {session_id}")

        governed = await self._govern_customizations(user_id=session.user_id, tools_map=self._tools)
        effective_instruction = self._compose_governed_instruction(self.instruction, governed)
        governed_tools = [self._tools[name].spec for name in governed.active_tool_names if name in self._tools]
        memory_context = governed.memory_text
        latest_summary = session.latest_summary.content if session.latest_summary else ""
        active_messages = list(session.messages[session.summarized_until :])
        active_model = (model or str(session.metadata.get("last_model") or "")).strip()

        context = AgentContext(
            session_id=session.id,
            user_id=session.user_id,
            instruction=effective_instruction,
            summary=latest_summary,
            memory_context=memory_context,
            messages=active_messages,
            tools=governed_tools,
            max_cycles=self.max_cycles,
            max_history_messages=self.max_history_messages,
        )
        composed_system = context.composed_instruction()
        windowed = context.windowed_messages()

        active_chars = sum(len(m.content) for m in active_messages)
        folded_chars = sum(len(m.content) for m in windowed)
        total_context_chars = len(composed_system) + folded_chars

        estimated_tokens = estimate_tokens(composed_system) + sum(
            estimate_tokens(m.content) + 4 for m in windowed
        )

        last_usage = session.metadata.get("last_usage") if isinstance(session.metadata.get("last_usage"), dict) else {}
        cum_usage = (
            session.metadata.get("cumulative_usage")
            if isinstance(session.metadata.get("cumulative_usage"), dict)
            else {}
        )
        last_prompt = last_usage.get("prompt_tokens") if isinstance(last_usage.get("prompt_tokens"), int) else None
        last_completion = (
            last_usage.get("completion_tokens") if isinstance(last_usage.get("completion_tokens"), int) else None
        )
        last_total = last_usage.get("total_tokens") if isinstance(last_usage.get("total_tokens"), int) else None

        usage_source = "reported" if last_prompt is not None else "estimated"
        window_tokens = _infer_context_window(active_model)
        basis_tokens = last_prompt if last_prompt is not None else estimated_tokens
        window_ratio = round(basis_tokens / window_tokens, 6) if window_tokens else None

        msg_ratio = round(len(active_messages) / self.compress_trigger_messages, 4)
        char_ratio = round(active_chars / self.compress_trigger_chars, 4)
        will_compress = (
            self.context_compressor is not None
            and len(active_messages) > self.compress_keep_recent
            and (
                len(active_messages) > self.compress_trigger_messages
                or active_chars > self.compress_trigger_chars
            )
        )

        return ContextUsageReport(
            session_id=session.id,
            user_id=session.user_id,
            model_name=active_model,
            total_messages=len(session.messages),
            summarized_until=session.summarized_until,
            active_messages=len(active_messages),
            windowed_messages=len(windowed),
            summary_count=len(session.summaries),
            instruction_chars=len(self.instruction.strip()),
            memory_chars=len(memory_context.strip()),
            summary_chars=len(latest_summary.strip()),
            active_messages_chars=active_chars,
            folded_messages_chars=folded_chars,
            total_context_chars=total_context_chars,
            estimated_context_tokens=estimated_tokens,
            context_window_tokens=window_tokens,
            context_window_ratio=window_ratio,
            last_prompt_tokens=last_prompt,
            last_completion_tokens=last_completion,
            last_total_tokens=last_total,
            cumulative_prompt_tokens=int(cum_usage.get("prompt_tokens", 0)),
            cumulative_completion_tokens=int(cum_usage.get("completion_tokens", 0)),
            cumulative_total_tokens=int(cum_usage.get("total_tokens", 0)),
            model_calls=int(cum_usage.get("calls", 0)),
            usage_source=usage_source,
            compress_trigger_messages=self.compress_trigger_messages,
            compress_trigger_chars=self.compress_trigger_chars,
            compress_keep_recent=self.compress_keep_recent,
            compress_message_ratio=msg_ratio,
            compress_char_ratio=char_ratio,
            will_compress_next_turn=will_compress,
        )

    async def rollback_session(self, session_id: str, snapshot_id: str) -> SessionSnapshot:
        session = await self.session_store.get(session_id)
        if session is None:
            raise KeyError(f"会话不存在: {session_id}")
        target = session.rollback_to(snapshot_id)
        await self.session_store.save(session)
        logger.warning(
            "[Session %s] 已回滚至快照 %s (messages=%d, summaries=%d)",
            session.id,
            target.id,
            target.message_count,
            len(session.summaries),
        )
        return target

    async def run_turn(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> TurnResult:
        # 1. 加载或新建会话
        session = await self.session_store.get(session_id) if session_id else None
        if session is None:
            session = Session(
                id=session_id or Session().id,
                user_id=user_id or self.user_id,
                title=user_input[:40],
            )
        else:
            if user_id:
                session.user_id = user_id
            if not session.title:
                session.title = user_input[:40]

        # 2. 保存本轮开始前的快照，方便出错回滚；追加用户消息并立即持久化
        snapshot = session.create_snapshot(label=f"before_turn:{user_input[:20]}")
        session.append(Message.user(user_input))
        await self.session_store.save(session)

        guard = LoopGuard(max_repeat_calls=self.max_repeat_calls)
        tool_specs = [t.spec for t in self._tools.values()]
        tool_events: list[ToolEvent] = []
        last_model_name = ""

        # 3. 核心决策循环：按需压缩 -> 扩展预算治理 -> 模型推理 -> 工具审核 -> 执行/拒绝回传 -> 持续保存
        for cycle in range(1, self.max_cycles + 1):
            await self._maybe_compress_context(session)
            governed = await self._govern_customizations(user_id=session.user_id, tools_map=self._tools)
            effective_instruction = self._compose_governed_instruction(self.instruction, governed)
            governed_tool_specs = [
                self._tools[name].spec for name in governed.active_tool_names if name in self._tools
            ]
            active_messages = list(session.messages[session.summarized_until :])
            latest_summary = session.latest_summary.content if session.latest_summary else ""

            logger.info(
                "[Session %s | Cycle %d/%d] 请求模型决策 (provider=%s, model=%s)...",
                session.id,
                cycle,
                self.max_cycles,
                provider or "default",
                model or "default",
            )
            context = AgentContext(
                session_id=session.id,
                user_id=session.user_id,
                provider=provider,
                model=model,
                instruction=effective_instruction,
                summary=latest_summary,
                memory_context=governed.memory_text,
                messages=active_messages,
                tools=governed_tool_specs,
                cycle=cycle,
                max_cycles=self.max_cycles,
                max_history_messages=self.max_history_messages,
            )

            response = await self.model_client.generate(context)
            if response.model_name:
                last_model_name = response.model_name
                session.metadata["last_model"] = response.model_name
            _record_session_usage(session, response.usage)

            # 持续保存助手消息（即使本轮包含工具调用，也在执行前落盘）
            session.append(
                Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    model_name=response.model_name or None,
                )
            )
            await self.session_store.save(session)

            # 模型未提出工具调用 -> 给出最终回复，本轮结束
            if not response.tool_calls:
                logger.info("[Session %s | Cycle %d] 模型给出最终回复，本轮结束。", session.id, cycle)
                return TurnResult(
                    session_id=session.id,
                    reply=response.content,
                    cycles=cycle,
                    stopped_reason="completed",
                    snapshot_id=snapshot.id,
                    model_name=last_model_name,
                    tool_events=tool_events,
                )

            # 模型提出工具调用 -> 逐个审核并把执行结果或拒绝原因交回模型
            for tool_call in response.tool_calls:
                result = await self._handle_tool_call(session, tool_call, guard)
                session.append(Message.tool(result))
                await self.session_store.save(session)

                tool_events.append(
                    ToolEvent(
                        cycle=cycle,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        approved=result.approved,
                        success=result.success,
                        summary=result.output if result.success else (result.error or ""),
                    )
                )

        # 4. 达到循环次数上限 -> 安全结束并落盘
        limit_reply = f"已达到最大循环次数上限（{self.max_cycles} 轮），本轮决策自动结束。你可以继续补充指令接着聊。"
        session.append(Message.assistant(content=limit_reply, model_name=last_model_name or None))
        await self.session_store.save(session)
        logger.warning("[Session %s] 达到最大循环次数上限 (%d)，强制结束本轮。", session.id, self.max_cycles)

        return TurnResult(
            session_id=session.id,
            reply=limit_reply,
            cycles=self.max_cycles,
            stopped_reason="max_cycles",
            snapshot_id=snapshot.id,
            model_name=last_model_name,
            tool_events=tool_events,
        )

    async def run_turn_stream(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式执行单轮决策循环：实时产出 session / delta / tool / done 事件，并保持持续落盘与快照一致。"""
        session = await self.session_store.get(session_id) if session_id else None
        if session is None:
            session = Session(
                id=session_id or Session().id,
                user_id=user_id or self.user_id,
                title=user_input[:40],
            )
        else:
            if user_id:
                session.user_id = user_id
            if not session.title:
                session.title = user_input[:40]

        snapshot = session.create_snapshot(label=f"before_turn:{user_input[:20]}")
        session.append(Message.user(user_input))
        await self.session_store.save(session)

        yield {
            "event": "session",
            "data": {"session_id": session.id, "snapshot_id": snapshot.id},
        }

        guard = LoopGuard(max_repeat_calls=self.max_repeat_calls)
        tool_specs = [t.spec for t in self._tools.values()]
        tool_events: list[ToolEvent] = []
        last_model_name = ""

        for cycle in range(1, self.max_cycles + 1):
            await self._maybe_compress_context(session)
            governed = await self._govern_customizations(user_id=session.user_id, tools_map=self._tools)
            effective_instruction = self._compose_governed_instruction(self.instruction, governed)
            governed_tool_specs = [
                self._tools[name].spec for name in governed.active_tool_names if name in self._tools
            ]
            active_messages = list(session.messages[session.summarized_until :])
            latest_summary = session.latest_summary.content if session.latest_summary else ""

            logger.info(
                "[Session %s | Cycle %d/%d] 流式请求模型决策 (provider=%s, model=%s)...",
                session.id,
                cycle,
                self.max_cycles,
                provider or "default",
                model or "default",
            )
            context = AgentContext(
                session_id=session.id,
                user_id=session.user_id,
                provider=provider,
                model=model,
                instruction=effective_instruction,
                summary=latest_summary,
                memory_context=governed.memory_text,
                messages=active_messages,
                tools=governed_tool_specs,
                cycle=cycle,
                max_cycles=self.max_cycles,
                max_history_messages=self.max_history_messages,
            )

            stream_fn = getattr(self.model_client, "generate_stream", None)
            if callable(stream_fn):
                queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

                async def _on_delta(chunk: str) -> None:
                    await queue.put(("delta", chunk))

                async def _produce() -> None:
                    try:
                        resp = await stream_fn(context, on_delta=_on_delta)
                        await queue.put(("response", resp))
                    except BaseException as exc:  # noqa: BLE001
                        await queue.put(("error", exc))

                producer = asyncio.create_task(_produce())
                response = None
                try:
                    while True:
                        kind, payload = await queue.get()
                        if kind == "delta":
                            yield {"event": "delta", "data": {"cycle": cycle, "content": payload}}
                        elif kind == "response":
                            response = payload
                            break
                        elif kind == "error":
                            raise payload
                finally:
                    if not producer.done():
                        producer.cancel()
                        try:
                            await producer
                        except asyncio.CancelledError:
                            pass
            else:
                response = await self.model_client.generate(context)
                if response.content and not response.tool_calls:
                    yield {"event": "delta", "data": {"cycle": cycle, "content": response.content}}

            assert response is not None
            if response.model_name:
                last_model_name = response.model_name
                session.metadata["last_model"] = response.model_name
            _record_session_usage(session, response.usage)

            session.append(
                Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    model_name=response.model_name or None,
                )
            )
            await self.session_store.save(session)

            if not response.tool_calls:
                logger.info("[Session %s | Cycle %d] 流式输出完成，本轮结束。", session.id, cycle)
                result = TurnResult(
                    session_id=session.id,
                    reply=response.content,
                    cycles=cycle,
                    stopped_reason="completed",
                    snapshot_id=snapshot.id,
                    model_name=last_model_name,
                    tool_events=tool_events,
                )
                yield {"event": "done", "data": result.model_dump(mode="json")}
                return

            for tool_call in response.tool_calls:
                tool_result = await self._handle_tool_call(session, tool_call, guard)
                session.append(Message.tool(tool_result))
                await self.session_store.save(session)

                ev = ToolEvent(
                    cycle=cycle,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    approved=tool_result.approved,
                    success=tool_result.success,
                    summary=tool_result.output if tool_result.success else (tool_result.error or ""),
                )
                tool_events.append(ev)
                yield {"event": "tool", "data": ev.model_dump(mode="json")}

        limit_reply = f"已达到最大循环次数上限（{self.max_cycles} 轮），本轮决策自动结束。你可以继续补充指令接着聊。"
        session.append(Message.assistant(content=limit_reply, model_name=last_model_name or None))
        await self.session_store.save(session)
        yield {"event": "delta", "data": {"cycle": self.max_cycles, "content": limit_reply}}
        result = TurnResult(
            session_id=session.id,
            reply=limit_reply,
            cycles=self.max_cycles,
            stopped_reason="max_cycles",
            snapshot_id=snapshot.id,
            model_name=last_model_name,
            tool_events=tool_events,
        )
        yield {"event": "done", "data": result.model_dump(mode="json")}

    async def _load_memory_context(self, user_id: str) -> str:
        if self.memory_store is None:
            return ""
        try:
            return await self.memory_store.load_context(
                user_id=user_id,
                recent_days=self.memory_recent_days,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Memory] 加载用户 %s 的长期记忆失败，已忽略: %s", user_id, exc)
            return ""

    async def _maybe_compress_context(self, session: Session) -> None:
        if self.context_compressor is None:
            return

        covered = session.summarized_until
        active = session.messages[covered:]
        if len(active) <= self.compress_keep_recent:
            return

        active_chars = sum(len(m.content) for m in active)
        if len(active) <= self.compress_trigger_messages and active_chars <= self.compress_trigger_chars:
            return

        split_idx = len(session.messages) - self.compress_keep_recent
        # 对齐 tool 调用边界：避免切在 Role.TOOL 前面导致 assistant(tool_calls) 与 tool 结果分离
        while split_idx > covered and session.messages[split_idx].role == Role.TOOL:
            split_idx -= 1

        if split_idx <= covered:
            return

        to_compress = session.messages[covered:split_idx]
        prev_summary = session.latest_summary.content if session.latest_summary else ""
        try:
            new_summary = (
                await self.context_compressor.summarize(
                    previous_summary=prev_summary,
                    messages=to_compress,
                )
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Context] 生成滚动摘要失败，本轮暂沿用原历史窗口: %s", exc)
            return

        if not new_summary:
            return

        session.add_summary(covered_until=split_idx, content=new_summary)
        await self.session_store.save(session)
        logger.info(
            "[Session %s] 已触发上下文压缩: covered_until=%d (本次压缩 %d 条，保留最近 %d 条)",
            session.id,
            split_idx,
            len(to_compress),
            len(session.messages) - split_idx,
        )

    async def _handle_tool_call(
        self,
        session: Session,
        tool_call: ToolCall,
        guard: LoopGuard,
        *,
        allowed_tools: dict[str, Tool] | None = None,
    ) -> ToolResult:
        registry = self._tools if allowed_tools is None else allowed_tools
        tool = registry.get(tool_call.name)
        if tool is None:
            if allowed_tools is not None and tool_call.name in self._tools:
                logger.warning("[SideChat] 物理拦截未授权写入/记忆工具: %s", tool_call.name)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    success=False,
                    approved=False,
                    error=(
                        f"侧边问答为只读讨论模式，已禁止调用 `{tool_call.name}`"
                        "（不允许写入主会话或长期记忆）。"
                    ),
                )
            logger.warning("[Tool] 模型请求了未注册的工具: %s", tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                approved=False,
                error=f"未注册的工具 `{tool_call.name}`",
            )

        # 审核前置守卫：拦截死循环重复调用或 A-B-A-B 震荡
        guard_res = guard.check_and_record(tool_call)
        if not guard_res.allowed:
            logger.warning("[Guard] 拦截工具 %s: %s", tool_call.name, guard_res.reason)
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                approved=False,
                error=guard_res.reason,
            )

        # 工具审核端口：规则审核 / 人工审批 / 风控模型审核
        decision = await self.tool_reviewer.review(session, tool.spec, tool_call)
        if not decision.approved:
            logger.warning(
                "[Review:%s] 拒绝工具 %s(%s): %s",
                decision.reviewer,
                tool_call.name,
                tool_call.arguments,
                decision.reason,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                approved=False,
                error=decision.reason or "审核未通过",
            )

        logger.info("[Tool] 审核通过，执行 %s(%s)", tool_call.name, tool_call.arguments)
        try:
            output = await tool.execute(tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=True,
                approved=True,
                output=output,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Tool] 执行 %s 异常", tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                approved=True,
                error=str(exc),
            )
