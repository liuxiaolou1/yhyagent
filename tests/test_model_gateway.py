"""多模型统一网关与前后端真实切模型测试（使用显式 httpx.MockTransport，不调用外部真实供应商）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.adapters.gateway import LocalFileModelRegistry, LocalFileUsageStore, ModelGateway
from app.adapters.review import PolicyToolReviewer
from app.adapters.storage import InMemoryMemoryStore, InMemorySessionStore
from app.adapters.web.server import create_app
from app.application import AgentService
from app.config import GatewaySettings, ProviderEndpoint, Settings


def _make_settings(*, rpm: int = 60) -> Settings:
    return Settings(
        providers={
            "default": ProviderEndpoint(
                name="default",
                api_key="sk-qwen-test",
                base_url="https://dashscope.example.com/v1",
                default_model="qwen-flash",
            ),
            "dashscope": ProviderEndpoint(
                name="dashscope",
                api_key="sk-qwen-test",
                base_url="https://dashscope.example.com/v1",
                default_model="qwen-flash",
            ),
            "deepseek": ProviderEndpoint(
                name="deepseek",
                api_key="sk-ds-test",
                base_url="https://api.deepseek.example.com/v1",
                default_model="deepseek-chat",
            ),
            "openai": ProviderEndpoint(
                name="openai",
                api_key="",
                base_url="https://api.openai.example.com/v1",
                default_model="gpt-4o-mini",
            ),
        },
        gateway=GatewaySettings(max_concurrent=2, requests_per_minute=rpm, max_output_tokens=2048),
    )


class TestModelGatewayAndSwitching(unittest.TestCase):
    def test_dynamic_model_switching_no_silent_fallback_and_usage_audit(self) -> None:
        """验证真实路由切换（千问 vs DeepSeek）、不存在模型诚实报错不兜底，以及用量日志不落盘正文。"""
        seen_requests: list[tuple[str, str, dict]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            seen_requests.append((str(request.url), request.headers.get("authorization", ""), body))

            model = body.get("model", "")
            if model == "deepseek-v4.1-flash":
                return httpx.Response(
                    400,
                    json={"error": {"message": "Model Not Exist: deepseek-v4.1-flash"}},
                )
            if model == "auth-fail-model":
                return httpx.Response(401, json={"error": {"message": "Invalid API key"}})
            if model == "connect-fail-model":
                raise httpx.ConnectError("mock connection refused")

            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"来自 {model} 的真实回复"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            usage_store = LocalFileUsageStore(base, user_id="yhy")
            mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            gateway = ModelGateway(
                settings=_make_settings(),
                default_provider="default",
                offline=False,
                usage_store=usage_store,
                http_client=mock_http,
            )
            service = AgentService(
                model_client=gateway,
                session_store=InMemorySessionStore(),
                tool_reviewer=PolicyToolReviewer(),
                user_id="yhy",
                memory_store=InMemoryMemoryStore(user_id="yhy"),
            )
            client = TestClient(create_app(service, gateway.default_label, gateway=gateway))

            # 1. /health/ready 与 /api/v1/models 不触发外部调用，也不泄露密钥
            ready = client.get("/health/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertFalse(ready.json()["provider_called"])
            self.assertEqual(len(seen_requests), 0)

            models_resp = client.get("/api/v1/models")
            self.assertEqual(models_resp.status_code, 200)
            models = {m["provider"]: m for m in models_resp.json()["models"]}
            self.assertTrue(models["dashscope"]["available"])
            self.assertTrue(models["deepseek"]["available"])
            self.assertFalse(models["openai"]["available"])
            self.assertNotIn("sk-qwen-test", models_resp.text)
            self.assertNotIn("sk-ds-test", models_resp.text)

            # 2. 前端选择 deepseek + deepseek-chat -> 真正打到 deepseek 域名与 key，绝不走千问
            secret_prompt = "这是秘密提示词_DO_NOT_LOG_TO_USAGE"
            r_ds = client.post(
                "/api/sessions/switch_01/messages",
                json={"content": secret_prompt, "provider": "deepseek", "model": "deepseek-chat"},
            )
            self.assertEqual(r_ds.status_code, 200)
            ds_data = r_ds.json()
            self.assertEqual(ds_data["last_model"], "deepseek/deepseek-chat")
            self.assertEqual(ds_data["messages"][-1]["model_name"], "deepseek/deepseek-chat")
            self.assertIn("deepseek-chat", ds_data["messages"][-1]["content"])
            self.assertIn("api.deepseek.example.com", seen_requests[-1][0])
            self.assertEqual(seen_requests[-1][1], "Bearer sk-ds-test")

            # 3. 同一会话切回千问（支持 qwen 别名 -> dashscope）
            r_qwen = client.post(
                "/api/sessions/switch_01/messages",
                json={"content": "换千问回答", "provider": "qwen", "model": "qwen-plus"},
            )
            self.assertEqual(r_qwen.status_code, 200)
            self.assertEqual(r_qwen.json()["last_model"], "dashscope/qwen-plus")
            self.assertIn("dashscope.example.com", seen_requests[-1][0])
            self.assertEqual(seen_requests[-1][1], "Bearer sk-qwen-test")

            # 3.5 查询本会话上下文使用量：应包含真实上报的 last_prompt_tokens 与 2 轮累计 token
            ctx_usage = client.get("/api/sessions/switch_01/context-usage")
            self.assertEqual(ctx_usage.status_code, 200)
            cu = ctx_usage.json()
            self.assertEqual(cu["usage_source"], "reported")
            self.assertEqual(cu["last_prompt_tokens"], 12)
            self.assertEqual(cu["last_completion_tokens"], 8)
            self.assertEqual(cu["last_total_tokens"], 20)
            self.assertEqual(cu["cumulative_prompt_tokens"], 24)
            self.assertEqual(cu["cumulative_completion_tokens"], 16)
            self.assertEqual(cu["cumulative_total_tokens"], 40)
            self.assertEqual(cu["model_calls"], 2)
            self.assertEqual(cu["context_window_tokens"], 128_000)
            self.assertAlmostEqual(cu["context_window_ratio"], round(12 / 128_000, 6))

            # 4. 选择上游不存在的模型（例如 deepseek-v4.1-flash）-> 诚实返回 422 MODEL_UNAVAILABLE，绝不兜底千问，并自动回滚孤立用户消息
            r_bad_model = client.post(
                "/api/sessions/switch_01/messages",
                json={"content": "用不存在的模型试试", "provider": "deepseek", "model": "deepseek-v4.1-flash"},
            )
            self.assertEqual(r_bad_model.status_code, 422)
            self.assertEqual(r_bad_model.json()["code"], "MODEL_UNAVAILABLE")
            self.assertIn("deepseek-v4.1-flash", r_bad_model.json()["error"])
            # 确认会话里没有残留这条失败的用户消息（仍是前面 2 轮共 4 条消息）
            sess_after = client.get("/api/sessions/switch_01").json()
            self.assertEqual(len(sess_after["messages"]), 4)

            # 5. 选择未配置 Key 的厂商（openai）-> 前置直接返回 503 MODEL_UNAVAILABLE，不发上游请求也不兜底
            before_count = len(seen_requests)
            r_unconf = client.post(
                "/api/sessions/switch_01/messages",
                json={"content": "用 openai", "provider": "openai", "model": "gpt-4o-mini"},
            )
            self.assertEqual(r_unconf.status_code, 503)
            self.assertEqual(r_unconf.json()["code"], "MODEL_UNAVAILABLE")
            self.assertEqual(len(seen_requests), before_count)

            # 6. 验证 401 鉴权失败与连接失败的诚实错误码
            r_401 = client.post(
                "/api/sessions/switch_01/messages",
                json={"content": "测 401", "provider": "deepseek", "model": "auth-fail-model"},
            )
            self.assertEqual(r_401.status_code, 502)
            self.assertEqual(r_401.json()["code"], "PROVIDER_AUTH_FAILED")

            r_conn = client.post(
                "/api/sessions/switch_01/messages",
                json={"content": "测断网", "provider": "deepseek", "model": "connect-fail-model"},
            )
            self.assertEqual(r_conn.status_code, 503)
            self.assertEqual(r_conn.json()["code"], "UPSTREAM_UNREACHABLE")

            # 7. 验证 /v1/chat/completions 透传、禁止透传 api_key，以及响应头
            r_reject_key = client.post(
                "/v1/chat/completions",
                json={
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "api_key": "sk-client-hack",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(r_reject_key.status_code, 422)
            self.assertEqual(r_reject_key.json()["code"], "INVALID_REQUEST")

            r_pass = client.post(
                "/v1/chat/completions",
                json={
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hello gateway"}],
                },
            )
            self.assertEqual(r_pass.status_code, 200)
            self.assertEqual(r_pass.headers["x-myagent-provider"], "deepseek")
            self.assertEqual(r_pass.headers["x-myagent-model"], "deepseek-chat")
            self.assertIn("x-myagent-call-id", r_pass.headers)

            # 8. 验证增量 usage API 与本地 JSONL 落盘（绝不含 prompt/回复正文，费用为 null/unknown）
            usage1 = client.get("/api/v1/usage", params={"after": 0}).json()
            self.assertGreaterEqual(len(usage1["items"]), 5)
            self.assertTrue(all(item["cost"] is None and item["cost_status"] == "unknown" for item in usage1["items"]))

            next_cursor = usage1["next_cursor"]
            usage2 = client.get("/api/v1/usage", params={"after": next_cursor}).json()
            self.assertEqual(usage2["items"], [])
            self.assertEqual(usage2["next_cursor"], next_cursor)

            usage_files = list((base / "yhy" / "gateway").rglob("usage.jsonl"))
            self.assertEqual(len(usage_files), 1)
            raw_jsonl = usage_files[0].read_text(encoding="utf-8")
            self.assertNotIn(secret_prompt, raw_jsonl)
            self.assertNotIn("真实回复", raw_jsonl)

    def test_gateway_rpm_rate_limit(self) -> None:
        """验证网关 RPM 频率上限返回 429 RATE_LIMITED。"""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-chat",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"total_tokens": 5},
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            gateway = ModelGateway(
                settings=_make_settings(rpm=1),
                default_provider="deepseek",
                offline=False,
                usage_store=LocalFileUsageStore(Path(tmpdir), user_id="rate_user"),
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            service = AgentService(
                model_client=gateway,
                session_store=InMemorySessionStore(),
                tool_reviewer=PolicyToolReviewer(),
                user_id="rate_user",
            )
            client = TestClient(create_app(service, gateway.default_label, gateway=gateway))

            first = client.post("/api/sessions/r1/messages", json={"content": "第 1 次"})
            self.assertEqual(first.status_code, 200)

            second = client.post("/api/sessions/r1/messages", json={"content": "第 2 次"})
            self.assertEqual(second.status_code, 429)
            self.assertEqual(second.json()["code"], "RATE_LIMITED")

    def test_add_persist_route_and_delete_custom_models(self) -> None:
        """验证后端新增已有厂商模型、新增自定义厂商端点、持久落盘不泄露密钥以及删除模型。"""
        seen: list[tuple[str, str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            seen.append((str(request.url), request.headers.get("authorization", ""), body.get("model", "")))
            return httpx.Response(
                200,
                json={
                    "model": body.get("model", ""),
                    "choices": [{"message": {"role": "assistant", "content": f"ok from {body.get('model')}"}}],
                    "usage": {"total_tokens": 10},
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            registry = LocalFileModelRegistry(base, user_id="yhy")
            gateway = ModelGateway(
                settings=_make_settings(),
                default_provider="default",
                offline=False,
                usage_store=LocalFileUsageStore(base, user_id="yhy"),
                model_registry=registry,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            service = AgentService(
                model_client=gateway,
                session_store=InMemorySessionStore(),
                tool_reviewer=PolicyToolReviewer(),
                user_id="yhy",
            )
            client = TestClient(create_app(service, gateway.default_label, gateway=gateway))

            # 1. 给已有厂商 deepseek 追加一个新模型 ID（复用 .env 中 deepseek 的 URL/Key）
            add_ds = client.post(
                "/api/v1/models",
                json={"provider": "deepseek", "model": "deepseek-reasoner"},
            )
            self.assertEqual(add_ds.status_code, 201)
            self.assertEqual(add_ds.json()["id"], "deepseek/deepseek-reasoner")
            self.assertTrue(add_ds.json()["available"])
            self.assertTrue(add_ds.json()["custom"])

            # 2. 新增自定义厂商 minimax（未提供 base_url 时应诚实返回 422）
            missing_url = client.post(
                "/api/v1/models",
                json={"provider": "minimax", "model": "MiniMax-M3.1-Flash"},
            )
            self.assertEqual(missing_url.status_code, 422)

            secret_mm_key = "sk-mm-super-secret-999"
            add_mm = client.post(
                "/api/v1/models",
                json={
                    "provider": "minimax",
                    "model": "MiniMax-M3.1-Flash",
                    "vendor_label": "MiniMax",
                    "base_url": "https://api.minimaxi.example.com/v1",
                    "api_key": secret_mm_key,
                },
            )
            self.assertEqual(add_mm.status_code, 201)
            self.assertEqual(add_mm.json()["id"], "minimax/MiniMax-M3.1-Flash")
            self.assertTrue(add_mm.json()["available"])
            self.assertNotIn(secret_mm_key, add_mm.text)

            # 确认 GET /api/v1/models 与 /api/status 都包含新模型且不泄露密钥
            models_resp = client.get("/api/v1/models")
            self.assertEqual(models_resp.status_code, 200)
            self.assertNotIn(secret_mm_key, models_resp.text)
            model_ids = [m["id"] for m in models_resp.json()["models"]]
            self.assertIn("deepseek/deepseek-reasoner", model_ids)
            self.assertIn("minimax/MiniMax-M3.1-Flash", model_ids)

            # 3. 真正用新注册的 minimax 模型发消息 -> 命中自定义 base_url 与 api_key
            chat = client.post(
                "/api/sessions/mm_01/messages",
                json={"content": "测试自定义厂商", "provider": "minimax", "model": "MiniMax-M3.1-Flash"},
            )
            self.assertEqual(chat.status_code, 200)
            self.assertEqual(chat.json()["last_model"], "minimax/MiniMax-M3.1-Flash")
            self.assertEqual(seen[-1][0], "https://api.minimaxi.example.com/v1/chat/completions")
            self.assertEqual(seen[-1][1], f"Bearer {secret_mm_key}")
            self.assertEqual(seen[-1][2], "MiniMax-M3.1-Flash")

            # 4. 模拟重启：新实例读取 .data/yhy/gateway/models.json 后依然存在
            reloaded = LocalFileModelRegistry(base, user_id="yhy")
            self.assertIsNotNone(reloaded.get("minimax/MiniMax-M3.1-Flash"))
            self.assertIsNotNone(reloaded.get("deepseek/deepseek-reasoner"))

            # 5. 删除自定义模型
            deleted = client.delete("/api/v1/models/minimax/MiniMax-M3.1-Flash")
            self.assertEqual(deleted.status_code, 200)
            self.assertTrue(deleted.json()["deleted"])
            # 再次删除返回 404
            deleted_again = client.delete("/api/v1/models/minimax/MiniMax-M3.1-Flash")
            self.assertEqual(deleted_again.status_code, 404)


if __name__ == "__main__":
    unittest.main()
