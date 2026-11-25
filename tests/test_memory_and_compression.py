"""长期记忆（按用户/日期存本地）与上下文压缩（折叠 + 滚动摘要 + 回滚联动）单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.adapters.context import HeuristicContextCompressor, LLMContextCompressor
from app.adapters.llm import OpenAIModelClient
from app.adapters.review import PolicyToolReviewer
from app.adapters.storage import (
    InMemorySessionStore,
    JsonFileSessionStore,
    LocalFileMemoryStore,
)
from app.adapters.tools import RememberTool, SearchMemoryTool
from app.application import AgentService
from app.domain import (
    AgentContext,
    MemoryScope,
    Message,
    ModelResponse,
    Role,
    Session,
    ToolCall,
    ToolResult,
)


class RecordingModelClient:
    """记录每次收到的 AgentContext，并可按脚本返回工具调用或文本回复。"""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.contexts: list[AgentContext] = []
        self._responses = list(responses or [])

    async def generate(self, context: AgentContext) -> ModelResponse:
        self.contexts.append(context.model_copy(deep=True))
        if self._responses:
            return self._responses.pop(0)
        last_user = next((m.content for m in reversed(context.messages) if m.role == Role.USER), "")
        return ModelResponse(content=f"已收到：{last_user}")


class TestLocalMemoryAndCompression(unittest.IsolatedAsyncioTestCase):
    async def test_session_file_store_layout_by_user_and_date_with_legacy_compat(self) -> None:
        """验证会话按 <base>/<user>/sessions/YYYY/MM/DD/<session_id>.json 落盘，且兼容旧路径与用户隔离。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            created = datetime(2026, 11, 25, 10, 30, tzinfo=timezone.utc)

            alice_store = JsonFileSessionStore(base, user_id="alice")
            bob_store = JsonFileSessionStore(base, user_id="bob")

            session = Session(id="sess_alice", user_id="alice", title="测试会话", created_at=created)
            session.append(Message.user("你好"))
            await alice_store.save(session)

            local_dt = created.astimezone()
            expected = (
                base
                / "alice"
                / "sessions"
                / f"{local_dt.year:04d}"
                / f"{local_dt.month:02d}"
                / f"{local_dt.day:02d}"
                / "sess_alice.json"
            )
            self.assertTrue(expected.exists())
            self.assertIsNotNone(await alice_store.get("sess_alice"))
            # 用户隔离：bob 查不到 alice 的会话
            self.assertIsNone(await bob_store.get("sess_alice"))

            # 兼容读取旧版 .data/sessions/<id>.json（默认用户）
            legacy_dir = base / "sessions"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_session = Session(id="legacy_01", title="旧版会话")
            (legacy_dir / "legacy_01.json").write_text(legacy_session.model_dump_json(), encoding="utf-8")

            default_store = JsonFileSessionStore(base, user_id="default")
            loaded_legacy = await default_store.get("legacy_01")
            self.assertIsNotNone(loaded_legacy)
            assert loaded_legacy is not None
            self.assertEqual(loaded_legacy.title, "旧版会话")
            listed_ids = [s.id for s in await default_store.list_sessions()]
            self.assertIn("legacy_01", listed_ids)

    async def test_local_file_memory_store_by_user_and_date_and_tools(self) -> None:
        """验证长期记忆按用户和日期落盘（MEMORY.md + YYYY/MM/DD.md），并支持跨会话注入与检索。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            mem_store = LocalFileMemoryStore(base, user_id="alice")

            day1 = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
            day2 = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)

            await mem_store.remember("偏好简洁中文回复", user_id="alice", scope=MemoryScope.PROFILE, recorded_at=day1)
            # PROFILE 重复写入应自动去重
            await mem_store.remember("偏好简洁中文回复", user_id="alice", scope=MemoryScope.PROFILE, recorded_at=day2)
            await mem_store.remember("服务器 10.3.13.148 已连上 SSH", user_id="alice", scope=MemoryScope.DAILY, recorded_at=day2)

            profile_file = base / "alice" / "memory" / "MEMORY.md"
            local_day2 = day2.astimezone()
            daily_file = (
                base
                / "alice"
                / "memory"
                / f"{local_day2.year:04d}"
                / f"{local_day2.month:02d}"
                / f"{local_day2.day:02d}.md"
            )
            self.assertTrue(profile_file.exists())
            self.assertTrue(daily_file.exists())
            self.assertEqual(profile_file.read_text(encoding="utf-8").count("偏好简洁中文回复"), 1)

            # 关键词检索
            matches = await mem_store.search("10.3.13.148", user_id="alice")
            self.assertEqual(len(matches), 1)
            self.assertIn("SSH", matches[0].content)

            # 通过 AgentService 跑工具写入 + 新会话自动注入长期记忆
            model = RecordingModelClient(
                responses=[
                    ModelResponse(
                        content="帮你记住这条配置",
                        tool_calls=[
                            ToolCall(
                                name="remember",
                                arguments={"content": "本地开发优先用文件存储", "scope": "profile"},
                            )
                        ],
                    ),
                    ModelResponse(content="已经记下来了。"),
                    ModelResponse(content="新会话也读到了你的长期记忆。"),
                ]
            )
            service = AgentService(
                model_client=model,
                session_store=InMemorySessionStore(),
                tool_reviewer=PolicyToolReviewer(),
                tools=[
                    RememberTool(mem_store, user_id="alice"),
                    SearchMemoryTool(mem_store, user_id="alice"),
                ],
                user_id="alice",
                memory_store=mem_store,
            )

            turn1 = await service.run_turn("记住：本地开发优先用文件存储", session_id="s1")
            self.assertEqual(turn1.stopped_reason, "completed")
            self.assertEqual(turn1.tool_events[0].tool_name, "remember")
            self.assertTrue(turn1.tool_events[0].success)

            # 开启一个全新会话 s2，应自动在 context.memory_context 中拿到刚刚记住的内容
            await service.run_turn("我之前的偏好是什么？", session_id="s2")
            last_ctx = model.contexts[-1]
            self.assertIn("本地开发优先用文件存储", last_ctx.memory_context)
            self.assertIn("偏好简洁中文回复", last_ctx.memory_context)

            # 验证 OpenAI 序列化把长期记忆拼进了 system prompt
            serialized = OpenAIModelClient._serialize_messages(last_ctx)
            self.assertEqual(serialized[0]["role"], "system")
            self.assertIn("用户长期记忆", serialized[0]["content"])
            self.assertIn("本地开发优先用文件存储", serialized[0]["content"])

    async def test_context_compression_tool_folding_and_rollback_invalidation(self) -> None:
        """验证旧工具输出折叠、滚动摘要触发（含工具边界对齐）及快照回滚时自动作废超前摘要。"""
        # 1. 验证旧工具输出折叠不破坏最近 2 条工具结果，也不修改原始消息
        long_output = "A" * 1000
        msgs = [
            Message.user("开始"),
            Message.assistant(tool_calls=[ToolCall(id="c1", name="t1")]),
            Message.tool(ToolResult(tool_call_id="c1", tool_name="t1", success=True, output=long_output)),
            Message.assistant(tool_calls=[ToolCall(id="c2", name="t2")]),
            Message.tool(ToolResult(tool_call_id="c2", tool_name="t2", success=True, output=long_output)),
            Message.assistant(tool_calls=[ToolCall(id="c3", name="t3")]),
            Message.tool(ToolResult(tool_call_id="c3", tool_name="t3", success=True, output=long_output)),
        ]
        ctx = AgentContext(
            session_id="fold_test",
            instruction="系统指令",
            messages=msgs,
            max_old_tool_chars=200,
            keep_recent_tool_messages=2,
        )
        windowed = ctx.windowed_messages()
        self.assertIn("已折叠", windowed[2].content)
        self.assertEqual(windowed[4].content, long_output)
        self.assertEqual(windowed[6].content, long_output)
        self.assertEqual(msgs[2].content, long_output)  # 原消息未被篡改

        # 2. 验证滚动摘要触发 + 回滚联动清理摘要
        store = InMemorySessionStore()
        model = RecordingModelClient()
        service = AgentService(
            model_client=model,
            session_store=store,
            tool_reviewer=PolicyToolReviewer(),
            context_compressor=HeuristicContextCompressor(),
            compress_trigger_messages=6,
            compress_keep_recent=2,
        )

        snapshots: list[str] = []
        for idx in range(1, 6):
            res = await service.run_turn(f"第 {idx} 轮需求：处理任务 #{idx}", session_id="comp_sess")
            snapshots.append(res.snapshot_id)

        saved = await store.get("comp_sess")
        assert saved is not None
        # 原始消息完整保留（5 轮 * 2 条 = 10 条）
        self.assertEqual(len(saved.messages), 10)
        # 至少触发过 1 次滚动摘要
        self.assertGreaterEqual(len(saved.summaries), 1)
        self.assertGreater(saved.summarized_until, 0)
        # 最后一轮模型拿到的上下文已包含摘要且只发未压缩的尾部消息
        last_ctx = model.contexts[-1]
        self.assertIn("任务 #1", last_ctx.summary)
        self.assertLess(len(last_ctx.messages), len(saved.messages))

        # 验证回滚到第 2 轮之前（snapshot[1]，此时 message_count=2）时，后续摘要自动失效
        await service.rollback_session("comp_sess", snapshots[1])
        rolled = await store.get("comp_sess")
        assert rolled is not None
        self.assertEqual(len(rolled.messages), 2)
        self.assertEqual(len(rolled.summaries), 0)

    async def test_llm_context_compressor_and_tool_boundary_alignment(self) -> None:
        """验证 LLMContextCompressor 正常调用模型生成摘要，且切分点不会拆散 assistant(tool_calls)+tool。"""
        summary_model = RecordingModelClient(
            responses=[ModelResponse(content="- 目标与约束：查询订单\n- 已完成操作与关键结果：已查 ORD-9")]
        )
        compressor = LLMContextCompressor(model_client=summary_model)

        store = InMemorySessionStore()
        # 构造一段在切分点附近包含 assistant(tool_calls) + tool 的历史
        tc = ToolCall(id="call_align", name="query_order", arguments={"order_id": "ORD-9"})
        seed = Session(id="align_sess")
        seed.append(Message.user("u1"))
        seed.append(Message.assistant("a1"))
        seed.append(Message.user("u2"))
        seed.append(Message.assistant("准备查订单", tool_calls=[tc]))
        seed.append(Message.tool(ToolResult(tool_call_id="call_align", tool_name="query_order", success=True, output="ok")))
        seed.append(Message.assistant("查完了"))
        await store.save(seed)

        chat_model = RecordingModelClient()
        service = AgentService(
            model_client=chat_model,
            session_store=store,
            tool_reviewer=PolicyToolReviewer(),
            context_compressor=compressor,
            compress_trigger_messages=5,
            compress_keep_recent=3,
        )

        # 追加第 7 条消息后：len=7, keep_recent=3 -> 初始 split_idx=4（恰好落在 Role.TOOL）
        # 对齐后应回退到 index=3（assistant(tool_calls)），确保 remaining[0] 不是孤立的 Role.TOOL
        await service.run_turn("继续下一步", session_id="align_sess")

        saved = await store.get("align_sess")
        assert saved is not None
        self.assertEqual(len(saved.summaries), 1)
        self.assertEqual(saved.summaries[0].covered_until, 3)
        self.assertIn("ORD-9", saved.summaries[0].content)
        # 确认主模型收到的消息序列开头不是孤立的 tool 消息
        sent_ctx = chat_model.contexts[-1]
        self.assertNotEqual(sent_ctx.messages[0].role, Role.TOOL)
        self.assertEqual(sent_ctx.messages[0].role, Role.ASSISTANT)
        self.assertEqual(len(sent_ctx.messages[0].tool_calls), 1)

    async def test_search_memory_tool_returns_json(self) -> None:
        """验证 SearchMemoryTool 返回合法 JSON 且支持空查询返回最近记录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalFileMemoryStore(Path(tmpdir), user_id="u1")
            await store.remember("使用 qwen-flash 做默认模型", user_id="u1", scope=MemoryScope.PROFILE)
            tool = SearchMemoryTool(store, user_id="u1")
            raw = await tool.execute({"query": "qwen-flash", "limit": 5})
            data = json.loads(raw)
            self.assertEqual(len(data["matches"]), 1)
            self.assertEqual(data["matches"][0]["source"], "MEMORY.md")


if __name__ == "__main__":
    unittest.main()
