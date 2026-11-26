"""侧边独立分支问答（仿 Antigravity /btw）测试：
验证只读可见主会话历史、绝不污染主会话、允许只读工具、物理屏蔽 remember/高危工具，以及独立分支按用户/日期落盘。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.adapters.review import PolicyToolReviewer
from app.adapters.storage import (
    JsonFileSessionStore,
    JsonFileSideChatStore,
    LocalFileMemoryStore,
)
from app.adapters.tools import DeleteOrderTool, QueryOrderTool, RememberTool, SearchMemoryTool
from app.adapters.web.server import create_app
from app.application import AgentService
from app.bootstrap import build_agent_service
from app.config import Settings
from app.domain import AgentContext, MemoryScope, ModelResponse, Role, ToolCall


class ScriptedSideChatModel:
    """记录收到的 AgentContext，并按脚本测试只读工具执行与 remember 物理拦截。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.contexts: list[AgentContext] = []
        self._responses = list(responses)

    async def generate(self, context: AgentContext) -> ModelResponse:
        self.contexts.append(context.model_copy(deep=True))
        if self._responses:
            return self._responses.pop(0)
        return ModelResponse(content="默认侧边回复", model_name="scripted/test")


class TestSideChatBranchMode(unittest.IsolatedAsyncioTestCase):
    async def test_side_chat_reads_parent_blocks_remember_allows_readonly_and_persists_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            session_store = JsonFileSessionStore(base, user_id="yhy")
            side_store = JsonFileSideChatStore(base, user_id="yhy")
            memory_store = LocalFileMemoryStore(base, user_id="yhy")

            # 预先写入一条长期记忆，方便测试只读 search_memory
            await memory_store.remember("部署服务器 IP 是 10.3.13.148", user_id="yhy", scope=MemoryScope.PROFILE)

            # 1. 先跑一轮主会话
            main_model = ScriptedSideChatModel(
                responses=[ModelResponse(content="主对话结论：订单 ORD-1001 暂不删除。", model_name="scripted/main")]
            )
            tools = [
                QueryOrderTool(),
                DeleteOrderTool(),
                RememberTool(memory_store, user_id="yhy"),
                SearchMemoryTool(memory_store, user_id="yhy"),
            ]
            service = AgentService(
                model_client=main_model,
                session_store=session_store,
                side_chat_store=side_store,
                tool_reviewer=PolicyToolReviewer(),
                tools=tools,
                user_id="yhy",
                memory_store=memory_store,
            )
            await service.run_turn("帮我看一下订单 ORD-1001", session_id="main_01")

            main_before = await session_store.get("main_01")
            assert main_before is not None
            before_msg_count = len(main_before.messages)
            before_updated = main_before.updated_at

            # 2. 切换为测试侧边分支行为的脚本模型：
            #    - Cycle 1: 故意尝试违规调用 remember（验证即便模型强行调用也会被物理拦截）
            #    - Cycle 2: 改用只读工具 search_memory + query_order（验证只读工具正常执行）
            #    - Cycle 3: 输出最终回复
            side_model = ScriptedSideChatModel(
                responses=[
                    ModelResponse(
                        content="尝试把侧边讨论记入长期记忆",
                        tool_calls=[
                            ToolCall(
                                name="remember",
                                arguments={"content": "不应写入的侧边内容", "scope": "profile"},
                            )
                        ],
                        model_name="scripted/side",
                    ),
                    ModelResponse(
                        content="改用只读工具检索记忆并查订单",
                        tool_calls=[
                            ToolCall(name="search_memory", arguments={"query": "10.3.13.148"}),
                            ToolCall(name="query_order", arguments={"order_id": "ORD-1001"}),
                        ],
                        model_name="scripted/side",
                    ),
                    ModelResponse(
                        content="侧边追问完成：已参考主对话和只读查询结果回答。",
                        model_name="scripted/side",
                    ),
                ]
            )
            service.model_client = side_model

            res = await service.run_side_turn(
                "刚才主对话里的订单和服务器信息帮我核对一下",
                parent_session_id="main_01",
                quote="订单 ORD-1001 暂不删除",
            )

            self.assertEqual(res.stopped_reason, "completed")
            self.assertEqual(res.parent_session_id, "main_01")
            self.assertTrue(res.side_id.startswith("side_"))

            # 验证发给模型的工具清单中：包含只读工具，绝不含 remember 与 delete_order
            exposed_tools = {t.name for t in side_model.contexts[0].tools}
            self.assertIn("search_memory", exposed_tools)
            self.assertIn("query_order", exposed_tools)
            self.assertNotIn("remember", exposed_tools)
            self.assertNotIn("delete_order", exposed_tools)

            # 验证侧边上下文完整看到了主对话历史 + 引用片段
            seen_texts = [m.content for m in side_model.contexts[0].messages]
            self.assertTrue(any("ORD-1001" in t for t in seen_texts))
            self.assertIn("订单 ORD-1001 暂不删除", side_model.contexts[0].instruction)

            # 验证 Cycle 1 的 remember 被物理拒绝，Cycle 2 的只读工具全部成功
            self.assertEqual(res.tool_events[0].tool_name, "remember")
            self.assertFalse(res.tool_events[0].approved)
            self.assertIn("只读", res.tool_events[0].summary)
            self.assertEqual(res.tool_events[1].tool_name, "search_memory")
            self.assertTrue(res.tool_events[1].approved and res.tool_events[1].success)
            self.assertEqual(res.tool_events[2].tool_name, "query_order")
            self.assertTrue(res.tool_events[2].approved and res.tool_events[2].success)

            # 验证长期记忆文件没有被侧边聊天写入任何脏内容
            mem_matches = await memory_store.search("不应写入的侧边内容", user_id="yhy")
            self.assertEqual(mem_matches, [])

            # 验证主会话完全零污染（消息数、updated_at 均未变化，主会话列表不含侧边分支）
            main_after = await session_store.get("main_01")
            assert main_after is not None
            self.assertEqual(len(main_after.messages), before_msg_count)
            self.assertEqual(main_after.updated_at, before_updated)
            self.assertEqual([s.id for s in await session_store.list_sessions()], ["main_01"])

            # 验证侧边分支已按用户/日期独立落盘在 .data/yhy/side_chats/YYYY/MM/DD/<side_id>.json
            side_files = list((base / "yhy" / "side_chats").rglob(f"{res.side_id}.json"))
            self.assertEqual(len(side_files), 1)
            reloaded = await side_store.get(res.side_id)
            assert reloaded is not None
            self.assertEqual(reloaded.parent_session_id, "main_01")
            self.assertEqual(reloaded.quote, "订单 ORD-1001 暂不删除")
            self.assertTrue(any(m.role == Role.TOOL for m in reloaded.messages))

            # 在同一侧边分支续聊，继续落盘到同一个分支文件
            res2 = await service.run_side_turn(
                "再补充追问一句",
                parent_session_id="main_01",
                side_id=res.side_id,
            )
            self.assertEqual(res2.side_id, res.side_id)
            reloaded2 = await side_store.get(res.side_id)
            assert reloaded2 is not None
            self.assertGreater(len(reloaded2.messages), len(reloaded.messages))


class TestSideChatFastAPIEndpoints(unittest.TestCase):
    def test_side_chat_branch_crud_and_stream_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service, label = build_agent_service(
                Settings(),
                offline=True,
                storage="file",
                user_id="alice",
                data_dir=Path(tmpdir),
            )
            client = TestClient(create_app(service, label))

            # 先建主会话
            m = client.post("/api/sessions/main_api/messages", json={"content": "查一下订单 ORD-1001"})
            self.assertEqual(m.status_code, 200)
            main_count = len(m.json()["messages"])

            # 发起侧边分支 SSE 追问
            with client.stream(
                "POST",
                "/api/side-chat/stream",
                json={
                    "question": "这个订单状态代表什么？",
                    "quote": "PENDING_REVIEW",
                    "session_id": "main_api",
                },
            ) as resp:
                self.assertEqual(resp.status_code, 200)
                raw = "".join(resp.iter_text())

            done_data = None
            for frame in raw.split("\n\n"):
                if "event: done" in frame:
                    for line in frame.splitlines():
                        if line.startswith("data:"):
                            done_data = json.loads(line[5:].strip())
            assert done_data is not None
            side_id = done_data["side_id"]
            self.assertTrue(side_id.startswith("side_"))
            self.assertEqual(done_data["parent_session_id"], "main_api")

            # 查询该主会话下的侧边分支列表
            listed = client.get("/api/sessions/main_api/side-chats")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()), 1)
            self.assertEqual(listed.json()[0]["id"], side_id)

            # 读取侧边分支详情
            detail = client.get(f"/api/side-chats/{side_id}")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["quote"], "PENDING_REVIEW")

            # 确认主会话未受影响
            self.assertEqual(len(client.get("/api/sessions/main_api").json()["messages"]), main_count)
            self.assertEqual([s["id"] for s in client.get("/api/sessions").json()], ["main_api"])

            # 删除侧边分支
            deleted = client.delete(f"/api/side-chats/{side_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertTrue(deleted.json()["deleted"])
            self.assertEqual(client.get(f"/api/side-chats/{side_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
