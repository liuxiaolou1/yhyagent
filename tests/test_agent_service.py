"""三层架构核心测试：覆盖工具审核/拒绝回传、持续保存续聊、快照回滚、循环上限与防震荡守卫。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.adapters.llm import MockModelClient
from app.adapters.review import PolicyToolReviewer
from app.adapters.storage import InMemorySessionStore, JsonFileSessionStore
from app.adapters.tools import DeleteOrderTool, QueryOrderTool
from app.application import AgentService
from app.config import InfraSettings, ProviderEndpoint, Settings, ToolSettings
from app.domain import AgentContext, LoopGuard, ModelResponse, Role, Session, ToolCall


class SpySessionStore(InMemorySessionStore):
    """记录 save 调用次数，用于验证运行过程中是否持续保存每条消息。"""

    def __init__(self) -> None:
        super().__init__()
        self.save_count = 0

    async def save(self, session: Session) -> None:
        self.save_count += 1
        await super().save(session)


class AlwaysToolCallModel:
    """始终不断发起不同参数工具调用的模型，用于测试 max_cycles 上限保护。"""

    async def generate(self, context: AgentContext) -> ModelResponse:
        return ModelResponse(
            content=f"第 {context.cycle} 轮继续查",
            tool_calls=[
                ToolCall(name="query_order", arguments={"order_id": f"ORD-{context.cycle}"})
            ],
        )


class RepeatToolCallModel:
    """反复提出完全相同工具调用的模型，用于验证 LoopGuard 拦截并回传拒绝原因。"""

    async def generate(self, context: AgentContext) -> ModelResponse:
        tool_msgs = [m for m in context.messages if m.role == Role.TOOL]
        if any("[REJECTED]" in m.content for m in tool_msgs):
            return ModelResponse(content="检测到重复调用被拦截，我直接总结结束。")
        return ModelResponse(
            content="再查一次同一个订单",
            tool_calls=[ToolCall(name="query_order", arguments={"order_id": "ORD-SAME"})],
        )


class TestAgentBackendArchitecture(unittest.IsolatedAsyncioTestCase):
    async def test_review_reject_then_adjust_and_continuous_save(self) -> None:
        """验证核心闭环：高危工具先审被拒 -> 拒绝原因交回模型 -> 改用安全工具 -> 最终回复 + 持续保存。"""
        store = SpySessionStore()
        service = AgentService(
            model_client=MockModelClient(),
            session_store=store,
            tool_reviewer=PolicyToolReviewer(interactive=False),
            tools=[QueryOrderTool(), DeleteOrderTool()],
            max_cycles=6,
        )

        result = await service.run_turn("帮我处理异常订单 ORD-1001", session_id="s_demo")

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.cycles, 3)
        self.assertEqual(len(result.tool_events), 2)
        # 第 1 个工具（delete_order）被审核拒绝
        self.assertEqual(result.tool_events[0].tool_name, "delete_order")
        self.assertFalse(result.tool_events[0].approved)
        # 第 2 个工具（query_order）审核通过并执行成功
        self.assertEqual(result.tool_events[1].tool_name, "query_order")
        self.assertTrue(result.tool_events[1].approved)
        self.assertTrue(result.tool_events[1].success)

        # 验证过程中持续保存（User + 3 Assistant + 2 Tool = 6 次保存）
        self.assertEqual(store.save_count, 6)

        # 验证相同 session_id 可以接着聊并保留历史消息
        turn2 = await service.run_turn("帮我继续跟进", session_id="s_demo")
        self.assertEqual(turn2.stopped_reason, "completed")
        self.assertIn("历史消息", turn2.reply)
        saved = await store.get("s_demo")
        assert saved is not None
        self.assertEqual(len(saved.messages), 8)  # 6 + User + Assistant

    async def test_file_store_persistence_and_snapshot_rollback(self) -> None:
        """验证本地文件持久化跨实例续聊，以及按 snapshot_id 回滚到上一轮之前。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = JsonFileSessionStore(Path(tmpdir))
            service1 = AgentService(
                model_client=MockModelClient(),
                session_store=store1,
                tool_reviewer=PolicyToolReviewer(),
                tools=[QueryOrderTool(), DeleteOrderTool()],
            )
            turn1 = await service1.run_turn("第一轮处理订单", session_id="file_sess")

            # 用新的 Store 实例模拟进程重启后续聊
            store2 = JsonFileSessionStore(Path(tmpdir))
            service2 = AgentService(
                model_client=MockModelClient(),
                session_store=store2,
                tool_reviewer=PolicyToolReviewer(),
                tools=[QueryOrderTool(), DeleteOrderTool()],
            )
            turn2 = await service2.run_turn("第二轮追问", session_id="file_sess")
            self.assertEqual(turn2.stopped_reason, "completed")

            # 回滚到第二轮开始前的快照
            await service2.rollback_session("file_sess", turn2.snapshot_id)
            rolled = await store2.get("file_sess")
            assert rolled is not None
            self.assertEqual(len(rolled.messages), 6)  # 仅保留第一轮的 6 条消息
            self.assertNotEqual(turn1.snapshot_id, turn2.snapshot_id)

    async def test_max_cycles_stops_loop_safely(self) -> None:
        """验证达到循环次数上限时安全终止并落盘。"""
        store = InMemorySessionStore()
        service = AgentService(
            model_client=AlwaysToolCallModel(),
            session_store=store,
            tool_reviewer=PolicyToolReviewer(),
            tools=[QueryOrderTool()],
            max_cycles=3,
        )

        result = await service.run_turn("一直查下去", session_id="s_limit")
        self.assertEqual(result.stopped_reason, "max_cycles")
        self.assertEqual(result.cycles, 3)
        self.assertEqual(len(result.tool_events), 3)

    async def test_loop_guard_blocks_duplicate_and_oscillation(self) -> None:
        """验证 LoopGuard 拦截连续重复调用与 A->B->A->B 震荡。"""
        store = InMemorySessionStore()
        service = AgentService(
            model_client=RepeatToolCallModel(),
            session_store=store,
            tool_reviewer=PolicyToolReviewer(),
            tools=[QueryOrderTool()],
            max_cycles=6,
            max_repeat_calls=2,
        )
        result = await service.run_turn("测试防重复", session_id="s_repeat")
        self.assertEqual(result.stopped_reason, "completed")
        self.assertTrue(result.tool_events[0].approved)
        self.assertFalse(result.tool_events[1].approved)
        self.assertIn("重复调用", result.tool_events[1].summary)

        # 独立验证 A -> B -> A -> B 震荡检测
        guard = LoopGuard(max_repeat_calls=3)
        a = ToolCall(name="edit", arguments={"v": 1})
        b = ToolCall(name="revert", arguments={"v": 0})
        self.assertTrue(guard.check_and_record(a).allowed)
        self.assertTrue(guard.check_and_record(b).allowed)
        self.assertTrue(guard.check_and_record(a).allowed)
        fourth = guard.check_and_record(b)
        self.assertFalse(fourth.allowed)
        self.assertIn("震荡", fourth.reason)

    def test_settings_masked_summary_hides_secrets(self) -> None:
        """验证脱敏配置摘要不会泄露任何密钥。"""
        secret = "sk-secret-do-not-leak-999"
        settings = Settings(
            providers={
                "default": ProviderEndpoint(
                    name="default",
                    api_key=secret,
                    base_url="https://example.com/v1",
                    default_model="qwen-flash",
                )
            },
            infra=InfraSettings(database_url=f"mysql+aiomysql://root:{secret}@127.0.0.1:3306/db"),
            tools=ToolSettings(tavily_api_key=secret),
        )
        self.assertNotIn(secret, str(settings.masked_summary()))


if __name__ == "__main__":
    unittest.main()
