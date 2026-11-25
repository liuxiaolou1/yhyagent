"""FastAPI Web 后端测试：验证前端静态资源、会话对话、错误格式、同源校验、回滚与长期记忆 API。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.adapters.web.server import build_local_hosts, create_app
from app.bootstrap import build_agent_service
from app.config import Settings


class TestFastAPIWebBackend(unittest.TestCase):
    def _make_client(
        self,
        tmpdir: str,
        *,
        user_id: str = "alice",
        allowed_hosts: set[str] | None = None,
    ) -> TestClient:
        settings = Settings()
        service, model_label = build_agent_service(
            settings,
            offline=True,
            storage="file",
            user_id=user_id,
            data_dir=Path(tmpdir),
        )
        app = create_app(service, model_label, allowed_hosts=allowed_hosts)
        return TestClient(app)

    def test_static_assets_and_status_and_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir)

            root = client.get("/")
            self.assertEqual(root.status_code, 200)
            self.assertIn("text/html", root.headers["content-type"])
            self.assertIn("MyAgent", root.text)
            self.assertIn("Content-Security-Policy", root.headers)

            css = client.get("/styles.css")
            self.assertEqual(css.status_code, 200)
            self.assertIn("text/css", css.headers["content-type"])

            js = client.get("/chat.js")
            self.assertEqual(js.status_code, 200)
            self.assertIn("javascript", js.headers["content-type"])

            status = client.get("/api/status")
            self.assertEqual(status.status_code, 200)
            status_body = status.json()
            self.assertEqual(status_body["model"], "mock/offline")
            self.assertTrue(status_body["offline"])
            self.assertEqual(status_body["user_id"], "alice")
            self.assertEqual(len(status_body["models"]), 1)
            self.assertEqual(status_body["models"][0]["provider"], "offline")

            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json(), {"status": "ok"})

            openapi = client.get("/openapi.json")
            self.assertEqual(openapi.status_code, 200)
            self.assertNotIn("Content-Security-Policy", openapi.headers)

    def test_chat_session_flow_and_rollback_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir)

            # 1. 发送第一轮消息（触发离线 Mock 审核拒绝 + 安全查询完整闭环）
            resp1 = client.post(
                "/api/sessions/web_01/messages",
                json={"content": "帮我处理异常订单 ORD-1001"},
            )
            self.assertEqual(resp1.status_code, 200)
            data1 = resp1.json()
            self.assertEqual(data1["id"], "web_01")
            self.assertEqual(data1["user_id"], "alice")
            self.assertEqual(len(data1["snapshots"]), 1)
            roles = [m["role"] for m in data1["messages"]]
            self.assertIn("user", roles)
            self.assertIn("tool", roles)
            self.assertIn("assistant", roles)
            self.assertTrue(any(m["intermediate"] for m in data1["messages"]))

            # 2. 会话列表与详情读取
            listed = client.get("/api/sessions")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()), 1)
            self.assertEqual(listed.json()[0]["id"], "web_01")

            detail = client.get("/api/sessions/web_01")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(len(detail.json()["messages"]), len(data1["messages"]))

            # 3. 发送第二轮消息，再回滚到第二轮之前
            resp2 = client.post(
                "/api/sessions/web_01/messages",
                json={"content": "继续总结一下"},
            )
            self.assertEqual(resp2.status_code, 200)
            data2 = resp2.json()
            self.assertEqual(len(data2["snapshots"]), 2)
            self.assertGreater(len(data2["messages"]), len(data1["messages"]))

            snap_id = data2["snapshots"][1]["id"]
            rolled = client.post(
                "/api/sessions/web_01/rollback",
                json={"snapshot_id": snap_id},
            )
            self.assertEqual(rolled.status_code, 200)
            self.assertEqual(len(rolled.json()["messages"]), len(data1["messages"]))

            # 4. 校验错误响应均使用前端约定的 {"error": "..."}
            missing = client.get("/api/sessions/not_exists")
            self.assertEqual(missing.status_code, 404)
            self.assertIn("error", missing.json())

            # 5. 验证会话上下文使用量接口（包含字符分布、估算 token 与压缩进度）
            usage_resp = client.get("/api/sessions/web_01/context-usage", params={"model": "gemini-4-neon"})
            self.assertEqual(usage_resp.status_code, 200)
            usage = usage_resp.json()
            self.assertEqual(usage["session_id"], "web_01")
            self.assertEqual(usage["total_messages"], len(data1["messages"]))
            self.assertGreater(usage["instruction_chars"], 0)
            self.assertGreater(usage["estimated_context_tokens"], 0)
            self.assertEqual(usage["context_window_tokens"], 1_000_000)
            self.assertIsNotNone(usage["context_window_ratio"])
            self.assertEqual(usage["usage_source"], "estimated")
            self.assertEqual(usage["compress_trigger_messages"], 24)
            self.assertFalse(usage["will_compress_next_turn"])

            missing_usage = client.get("/api/sessions/not_exists/context-usage")
            self.assertEqual(missing_usage.status_code, 404)
            self.assertIn("error", missing_usage.json())

            empty_msg = client.post("/api/sessions/web_01/messages", json={"content": "   "})
            self.assertEqual(empty_msg.status_code, 400)
            self.assertIn("error", empty_msg.json())

            bad_id = client.post("/api/sessions/bad%20id!/messages", json={"content": "hello"})
            self.assertEqual(bad_id.status_code, 400)
            self.assertIn("error", bad_id.json())

    def test_memory_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir, user_id="yhy")

            created = client.post(
                "/api/memories",
                json={"content": "前端由 Codex 编写，后端已切换为 FastAPI", "scope": "profile"},
            )
            self.assertEqual(created.status_code, 201)
            self.assertEqual(created.json()["source"], "MEMORY.md")

            searched = client.get("/api/memories", params={"query": "FastAPI"})
            self.assertEqual(searched.status_code, 200)
            items = searched.json()["items"]
            self.assertEqual(len(items), 1)
            self.assertIn("FastAPI", items[0]["content"])

    def test_local_host_and_origin_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hosts = build_local_hosts("127.0.0.1", 8000)
            client = self._make_client(tmpdir, allowed_hosts=hosts)

            ok = client.get("/api/status", headers={"Host": "127.0.0.1:8000"})
            self.assertEqual(ok.status_code, 200)

            bad_host = client.get("/api/status", headers={"Host": "evil.example.com"})
            self.assertEqual(bad_host.status_code, 403)
            self.assertIn("error", bad_host.json())

            bad_origin = client.get(
                "/api/status",
                headers={"Host": "127.0.0.1:8000", "Origin": "http://evil.example.com"},
            )
            self.assertEqual(bad_origin.status_code, 403)


if __name__ == "__main__":
    unittest.main()
