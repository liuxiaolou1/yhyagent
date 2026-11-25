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
import logging
from collections.abc import AsyncIterator
from typing import Any, Iterable

from pydantic import BaseModel, Field

from app.domain import (
    AgentContext,
    ContextUsageReport,
    LoopGuard,
    Message,
    Role,
    Session,
    SessionSnapshot,
    Tool,
    ToolCall,
    ToolResult,
    estimate_tokens,
)

from .ports import ContextCompressor, MemoryStore, ModelClient, SessionStore, ToolReviewer

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

        memory_context = await self._load_memory_context(session.user_id)
        latest_summary = session.latest_summary.content if session.latest_summary else ""
        active_messages = list(session.messages[session.summarized_until :])
        active_model = (model or str(session.metadata.get("last_model") or "")).strip()

        context = AgentContext(
            session_id=session.id,
            user_id=session.user_id,
            instruction=self.instruction,
            summary=latest_summary,
            memory_context=memory_context,
            messages=active_messages,
            tools=[t.spec for t in self._tools.values()],
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

        # 3. 核心决策循环：按需压缩 -> 注入记忆 -> 模型推理 -> 工具审核 -> 执行/拒绝回传 -> 持续保存
        for cycle in range(1, self.max_cycles + 1):
            await self._maybe_compress_context(session)
            memory_context = await self._load_memory_context(session.user_id)
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
                instruction=self.instruction,
                summary=latest_summary,
                memory_context=memory_context,
                messages=active_messages,
                tools=tool_specs,
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
            memory_context = await self._load_memory_context(session.user_id)
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
                instruction=self.instruction,
                summary=latest_summary,
                memory_context=memory_context,
                messages=active_messages,
                tools=tool_specs,
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
    ) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
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
