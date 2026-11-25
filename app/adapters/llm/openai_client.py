"""OpenAI 兼容模型适配器（原生支持 Tool Calling / Function Calling）。

兼容 DashScope (Qwen) / DeepSeek / GLM / Gemini OpenAI 兼容端点 / vLLM / Ollama。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import ProviderEndpoint, Settings
from app.domain import AgentContext, Message, ModelResponse, Role, ToolCall, ToolSpec


class OpenAIModelClient:
    """实现 application.ports.ModelClient 端口。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: str = "default",
        model: str | None = None,
        endpoint: ProviderEndpoint | None = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> None:
        if endpoint is not None:
            ep = endpoint
            self.timeout = timeout
        elif settings is not None:
            ep = settings.get_provider(provider)
            self.timeout = settings.llm_timeout
        else:
            ep = ProviderEndpoint(name="none")
            self.timeout = timeout

        self.provider = ep.name
        self.api_key = ep.api_key
        self.base_url = ep.base_url.rstrip("/")
        self.model = model or ep.default_model
        self.temperature = temperature

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_api_key_here" and self.base_url)

    async def generate(self, context: AgentContext) -> ModelResponse:
        if not self.is_configured:
            raise RuntimeError(f"提供方 '{self.provider}' 未配置有效的 API Key 或 Base URL。")

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": self._serialize_messages(context),
        }
        if context.tools:
            payload["tools"] = [self._serialize_tool(t) for t in context.tools]
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice_msg = data["choices"][0]["message"]
        content = choice_msg.get("content") or ""
        tool_calls = self._parse_tool_calls(choice_msg.get("tool_calls") or [])
        usage = data.get("usage") or {}

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            model_name=f"{self.provider}/{self.model}",
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        )

    @staticmethod
    def _serialize_messages(context: AgentContext) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        system_prompt = context.composed_instruction()
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in context.windowed_messages():
            result.append(OpenAIModelClient._serialize_single_message(msg))
        return result

    @staticmethod
    def _serialize_single_message(msg: Message) -> dict[str, Any]:
        if msg.role == Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id or "",
                "name": msg.tool_name or "",
                "content": msg.content,
            }
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        return {"role": msg.role.value, "content": msg.content}

    @staticmethod
    def _serialize_tool(spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for item in raw_calls:
            fn = item.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            parsed.append(
                ToolCall(
                    id=item.get("id") or ToolCall(name=fn.get("name", "unknown")).id,
                    name=fn.get("name", "unknown"),
                    arguments=args,
                )
            )
        return parsed
