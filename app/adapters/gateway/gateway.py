"""多模型统一网关（参照 D:/project/ai_patent/model-gateway）：

核心原则：
1. 凭证严格留在服务端（.env 与本地 models.json），拒绝在聊天请求体中透传 api_key / base_url；
2. 支持按 provider + model 实时动态路由，支持持久化的新增/删除模型，绝不静默跨模型兜底；
3. 诚实错误分类（401 鉴权 / 403 权限 / 404 模型不存在 / 网络不可达 / 超时未知结果）；
4. 并发信号量 + RPM 频率护栏 + 仅记录元数据与实际 usage 的增量审计日志；
5. 同时实现 application.ports.ModelClient（供 AgentService 调用）与 OpenAI 兼容透传（供 /v1/chat/completions）。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from app.adapters.llm.mock_client import MockModelClient
from app.adapters.llm.openai_client import OpenAIModelClient
from app.config import ProviderEndpoint, Settings
from app.domain import AgentContext, ModelResponse, ToolCall

from .errors import GatewayError, connection_category, extract_upstream_message, provider_failure
from .model_registry import CustomModelConfig, InMemoryModelRegistry
from .usage_store import InMemoryUsageStore, UsageRecord, normalize_usage

logger = logging.getLogger(__name__)

# 前端厂商简称 -> 后端 Settings.providers 标准键
PROVIDER_ALIASES: dict[str, str] = {
    "qwen": "dashscope",
    "tongyi": "dashscope",
    "aliyun": "dashscope",
    "dashscope": "dashscope",
    "deepseek": "deepseek",
    "glm": "glm",
    "zhipu": "glm",
    "gemini": "gemini",
    "google": "gemini",
    "openai": "openai",
    "default": "default",
}

# 后端标准键 -> 前端图标厂商标识与展示名
PROVIDER_UI_META: dict[str, tuple[str, str]] = {
    "dashscope": ("qwen", "通义千问"),
    "deepseek": ("deepseek", "DeepSeek"),
    "glm": ("glm", "智谱 GLM"),
    "gemini": ("gemini", "Google Gemini"),
    "openai": ("openai", "OpenAI"),
}

# 各厂商默认模型目录（旗舰排前，同时保留轻量低成本模型）：(model_id, display_name, icon_vendor, vendor_label)
FLAGSHIP_CATALOG: dict[str, list[tuple[str, str, str, str]]] = {
    "dashscope": [
        ("qwen3.8-flash", "Qwen 3.8 Flash", "qwen", "通义千问"),
        ("qwen4-max", "Qwen 4 Max", "qwen", "通义千问"),
        ("qwen3.8-max", "Qwen 3.8 Max", "qwen", "通义千问"),
        ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "deepseek", "DeepSeek · 百炼"),
        ("qwen-flash", "Qwen Flash", "qwen", "通义千问 · 轻量"),
    ],
    "deepseek": [
        ("deepseek-v4-pro", "DeepSeek V4 Pro", "deepseek", "DeepSeek 官方"),
        ("deepseek-flash", "DeepSeek Flash", "deepseek", "DeepSeek 官方"),
        ("deepseek-chat", "DeepSeek Chat", "deepseek", "DeepSeek 官方 · 经典"),
    ],
    "glm": [
        ("glm-5.4", "GLM-5.4", "glm", "智谱 GLM"),
        ("glm-5.3-flash", "GLM-5.3 Flash", "glm", "智谱 GLM"),
        ("glm-4-flash", "GLM-4 Flash", "glm", "智谱 GLM · 轻量"),
    ],
    "gemini": [
        ("gemini-4-neon", "Gemini 4 Neon", "gemini", "Google Gemini"),
        ("gemini-4-carbon", "Gemini 4 Carbon", "gemini", "Google Gemini"),
        ("gemini-3.8-flash", "Gemini 3.8 Flash", "gemini", "Google Gemini"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash", "gemini", "Google Gemini · 轻量"),
    ],
}


class ModelDescriptor(BaseModel):
    """对外暴露的模型目录条目（永不包含任何密钥）。"""

    id: str
    provider: str
    vendor: str
    vendor_label: str
    model: str
    name: str
    available: bool
    is_default: bool = False
    custom: bool = False


class ResolvedTarget(BaseModel):
    """本次调用解析后的确定目标。"""

    provider: str
    model: str
    endpoint: ProviderEndpoint | None = None
    offline: bool = False

    @property
    def label(self) -> str:
        if self.offline:
            return "mock/offline"
        return f"{self.provider}/{self.model}"


def canonical_provider(raw: str | None) -> str:
    if not raw:
        return ""
    key = raw.strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def normalize_custom_provider(raw: str) -> str:
    """将用户输入的厂商名规整为安全的 provider 标识。"""
    aliased = canonical_provider(raw)
    if aliased in PROVIDER_UI_META or aliased == "default":
        return aliased
    slug = re.sub(r"[^a-z0-9_-]+", "-", aliased).strip("-")[:32]
    if slug:
        return slug
    digest = hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()[:6]
    return f"custom-{digest}"


def _is_local_base_url(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def _endpoint_ready(ep: ProviderEndpoint) -> bool:
    if ep.is_configured:
        return True
    return bool(ep.base_url and _is_local_base_url(ep.base_url))


def infer_provider_from_model(model: str) -> str:
    lower = model.strip().lower()
    if lower.startswith(("qwen", "qwq")):
        return "dashscope"
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith(("glm", "chatglm")):
        return "glm"
    if lower.startswith("gemini"):
        return "gemini"
    if lower.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return ""


class ModelGateway:
    """多模型统一网关：实现 application.ports.ModelClient 端口，并对外提供目录、注册、用量与透传能力。"""

    def __init__(
        self,
        settings: Settings,
        *,
        default_provider: str = "default",
        offline: bool = False,
        usage_store: InMemoryUsageStore | None = None,
        model_registry: InMemoryModelRegistry | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.default_provider = canonical_provider(default_provider) or "default"
        self.offline = offline
        self.usage_store = usage_store or InMemoryUsageStore()
        self.model_registry = model_registry or InMemoryModelRegistry()
        self._injected_http_client = http_client
        self._mock_client = MockModelClient()
        self._semaphore = asyncio.Semaphore(settings.gateway.max_concurrent)

    @property
    def is_ready(self) -> bool:
        if self.offline:
            return True
        if any(ep.is_configured for ep in self.settings.providers.values()):
            return True
        return any(_endpoint_ready(self._effective_endpoint_for_custom(cfg)) for cfg in self.model_registry.list_all())

    @property
    def default_label(self) -> str:
        if self.offline:
            return "mock/offline"
        ep = self.settings.providers.get(self.default_provider)
        if ep and ep.is_configured:
            return f"{ep.name}/{ep.default_model}"
        return "mock/offline (未检测到有效 API Key)"

    def _effective_endpoint_for_custom(self, cfg: CustomModelConfig) -> ProviderEndpoint:
        base_ep = self.settings.providers.get(cfg.provider)
        sibling = self.model_registry.find_by_provider(cfg.provider)
        base_url = (
            cfg.base_url
            or (sibling.base_url if sibling else "")
            or (base_ep.base_url if base_ep else "")
        )
        api_key = (
            cfg.api_key
            or (sibling.api_key if sibling else "")
            or (base_ep.api_key if base_ep else "")
        )
        return ProviderEndpoint(
            name=cfg.provider,
            api_key=api_key,
            base_url=base_url,
            default_model=cfg.model,
        )

    def _descriptor_from_custom(self, cfg: CustomModelConfig) -> ModelDescriptor:
        ep = self._effective_endpoint_for_custom(cfg)
        default_vendor, default_label = PROVIDER_UI_META.get(cfg.provider, ("local", cfg.provider))
        return ModelDescriptor(
            id=cfg.id,
            provider=cfg.provider,
            vendor=cfg.vendor or default_vendor,
            vendor_label=cfg.vendor_label or default_label,
            model=cfg.model,
            name=cfg.name or cfg.model,
            available=_endpoint_ready(ep) and not self.offline,
            is_default=False,
            custom=True,
        )

    def add_model(
        self,
        *,
        provider: str,
        model: str,
        name: str = "",
        vendor_label: str = "",
        base_url: str = "",
        api_key: str = "",
    ) -> ModelDescriptor:
        """持久化新增（或更新）一个模型配置到服务端注册表，返回脱敏后的 ModelDescriptor。"""
        raw_provider = provider.strip()
        clean_model = model.strip()
        if not raw_provider:
            raise GatewayError(422, "INVALID_REQUEST", "厂商名称 provider 不能为空。")
        if not clean_model or len(clean_model) > 128 or "\n" in clean_model or "\r" in clean_model:
            raise GatewayError(422, "INVALID_REQUEST", "模型 ID（model）不能为空且不能包含换行符。")

        norm_provider = normalize_custom_provider(raw_provider)
        clean_base_url = base_url.strip().rstrip("/")
        if clean_base_url:
            parsed = urlsplit(clean_base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise GatewayError(
                    422,
                    "INVALID_REQUEST",
                    "Base URL 必须是以 http:// 或 https:// 开头的有效地址（例如 https://api.example.com/v1）。",
                )

        existing_base_ep = self.settings.providers.get(norm_provider)
        existing_custom_ep = self.model_registry.find_by_provider(norm_provider)
        if not clean_base_url and not (existing_base_ep and existing_base_ep.base_url) and not (
            existing_custom_ep and existing_custom_ep.base_url
        ):
            raise GatewayError(
                422,
                "INVALID_REQUEST",
                f"厂商 `{raw_provider}` 不在 .env 预置列表中，请提供 Base URL（如 https://.../v1）后再添加。",
                provider=norm_provider,
                model=clean_model,
            )

        default_vendor, default_label = PROVIDER_UI_META.get(norm_provider, ("local", raw_provider))
        model_id = f"{norm_provider}/{clean_model}"
        previous = self.model_registry.get(model_id)

        cfg = CustomModelConfig(
            id=model_id,
            provider=norm_provider,
            model=clean_model,
            name=name.strip() or clean_model,
            vendor=default_vendor,
            vendor_label=vendor_label.strip() or (previous.vendor_label if previous else "") or default_label,
            base_url=clean_base_url or (previous.base_url if previous else ""),
            api_key=api_key.strip() or (previous.api_key if previous else ""),
        )
        saved = self.model_registry.upsert(cfg)
        return self._descriptor_from_custom(saved)

    def delete_model(self, model_id: str) -> bool:
        """从持久化注册表中删除指定自定义模型。"""
        return self.model_registry.delete(model_id.strip())

    def list_models(self) -> list[ModelDescriptor]:
        """返回服务端配置的所有内置模型与持久化自定义模型（永不输出 API Key）。"""
        descriptors: list[ModelDescriptor] = []
        seen_ids: set[str] = set()

        if self.offline:
            descriptors.append(
                ModelDescriptor(
                    id="mock/offline",
                    provider="offline",
                    vendor="local",
                    vendor_label="离线演示",
                    model="offline",
                    name="离线模型",
                    available=True,
                    is_default=True,
                    custom=False,
                )
            )
            seen_ids.add("mock/offline")
        else:
            default_ep = self.settings.providers.get(self.default_provider)
            default_base = default_ep.base_url.rstrip("/") if default_ep else ""

            default_assigned = False
            ordered_names = ["dashscope", "deepseek", "glm", "gemini", "openai"]
            for name in ordered_names:
                ep = self.settings.providers.get(name)
                if ep is None:
                    continue
                vendor, vendor_label = PROVIDER_UI_META.get(name, ("local", name))
                is_default_provider = (
                    name == self.default_provider
                    or (
                        self.default_provider == "default"
                        and ep.is_configured
                        and ep.base_url.rstrip("/") == default_base
                    )
                )

                # 已配置厂商展开旗舰目录；未配置厂商保留一条占位供 /api/v1/models 可观测
                entries: list[tuple[str, str, str, str]] = []
                if ep.is_configured and name in FLAGSHIP_CATALOG:
                    entries.extend(FLAGSHIP_CATALOG[name])
                    if ep.default_model and all(m_id != ep.default_model for m_id, *_ in entries):
                        entries.append((ep.default_model, ep.default_model, vendor, vendor_label))
                else:
                    entries.append((ep.default_model, ep.default_model, vendor, vendor_label))

                for idx, (model_id, display_name, item_vendor, item_label) in enumerate(entries):
                    desc_id = f"{name}/{model_id}"
                    if desc_id in seen_ids:
                        continue
                    seen_ids.add(desc_id)
                    mark_default = is_default_provider and not default_assigned and idx == 0
                    if mark_default:
                        default_assigned = True
                    descriptors.append(
                        ModelDescriptor(
                            id=desc_id,
                            provider=name,
                            vendor=item_vendor,
                            vendor_label=item_label,
                            model=model_id,
                            name=display_name,
                            available=ep.is_configured,
                            is_default=mark_default,
                            custom=False,
                        )
                    )

            # 若 default 指向自定义 base_url 且未归入内置列表，也单独列出来
            if default_ep and default_ep.is_configured and not default_assigned:
                inferred = infer_provider_from_model(default_ep.default_model)
                vendor, vendor_label = PROVIDER_UI_META.get(inferred, ("local", "默认配置"))
                desc_id = f"default/{default_ep.default_model}"
                seen_ids.add(desc_id)
                descriptors.insert(
                    0,
                    ModelDescriptor(
                        id=desc_id,
                        provider="default",
                        vendor=vendor,
                        vendor_label=vendor_label,
                        model=default_ep.default_model,
                        name=default_ep.default_model,
                        available=True,
                        is_default=True,
                        custom=False,
                    ),
                )

        # 合并服务端持久化的自定义模型
        for cfg in self.model_registry.list_all():
            desc = self._descriptor_from_custom(cfg)
            if desc.id in seen_ids:
                descriptors = [desc if existing.id == desc.id else existing for existing in descriptors]
            else:
                seen_ids.add(desc.id)
                descriptors.append(desc)

        return descriptors

    def resolve_target(self, provider: str | None = None, model: str | None = None) -> ResolvedTarget:
        """严格解析本次请求的目标厂商与模型；指定了不可用厂商时直接报错，绝不静默兜底到千问。"""
        req_provider = canonical_provider(provider)
        req_model = (model or "").strip()

        if self.offline:
            if not req_provider or req_provider in ("offline", "local", "runtime"):
                return ResolvedTarget(provider="offline", model=req_model or "offline", offline=True)
            raise GatewayError(
                503,
                "MODEL_UNAVAILABLE",
                f"当前服务运行在离线演示模式（--offline），无法调用真实厂商 `{req_provider}`，请去掉 --offline 后重启。",
                provider=req_provider,
                model=req_model,
            )

        # 支持 model 字段写成 "<provider>/<model>" 形式
        if not req_provider and "/" in req_model:
            prefix, rest = req_model.split("/", 1)
            candidate = canonical_provider(prefix)
            if (
                candidate in self.settings.providers
                or self.model_registry.get(f"{candidate}/{rest.strip()}") is not None
                or self.model_registry.find_by_provider(candidate) is not None
            ):
                req_provider = candidate
                req_model = rest.strip()

        if not req_provider and req_model:
            req_provider = infer_provider_from_model(req_model)

        target_provider = req_provider or self.default_provider

        # 1. 优先查找持久化自定义模型注册表
        custom_exact = (
            self.model_registry.get(f"{target_provider}/{req_model}")
            if req_model
            else None
        )
        if custom_exact is not None:
            endpoint = self._effective_endpoint_for_custom(custom_exact)
            if not _endpoint_ready(endpoint):
                raise GatewayError(
                    503,
                    "MODEL_UNAVAILABLE",
                    f"自定义模型 `{custom_exact.id}` 缺少有效的 API Key 或 Base URL，无法调用（不会自动兜底到其他模型）。",
                    provider=custom_exact.provider,
                    model=custom_exact.model,
                )
            return ResolvedTarget(
                provider=custom_exact.provider,
                model=custom_exact.model,
                endpoint=endpoint,
                offline=False,
            )

        custom_provider_cfg = self.model_registry.find_by_provider(target_provider)
        if target_provider not in self.settings.providers:
            if custom_provider_cfg is not None:
                endpoint = self._effective_endpoint_for_custom(custom_provider_cfg)
                target_model = req_model or custom_provider_cfg.model
                if not _endpoint_ready(endpoint):
                    raise GatewayError(
                        503,
                        "MODEL_UNAVAILABLE",
                        f"自定义厂商 `{target_provider}` 缺少有效的 API Key 或 Base URL，无法调用 `{target_model}`。",
                        provider=target_provider,
                        model=target_model,
                    )
                return ResolvedTarget(
                    provider=target_provider,
                    model=target_model,
                    endpoint=endpoint,
                    offline=False,
                )

            supported = ", ".join(k for k in self.settings.providers if k != "default")
            raise GatewayError(
                422,
                "MODEL_UNAVAILABLE",
                f"未知模型厂商 `{target_provider}`（可选：{supported}，或先通过新增模型注册自定义厂商），网关不会自动兜底切换其他模型。",
                provider=target_provider,
                model=req_model,
            )

        endpoint = self.settings.providers[target_provider]
        target_model = req_model or endpoint.default_model
        if not _endpoint_ready(endpoint):
            raise GatewayError(
                503,
                "MODEL_UNAVAILABLE",
                f"厂商 `{target_provider}` 未配置有效的 API Key 或 Base URL，无法调用模型 `{target_model}`（不会自动兜底到其他模型）。",
                provider=target_provider,
                model=target_model,
            )

        return ResolvedTarget(provider=target_provider, model=target_model, endpoint=endpoint, offline=False)

    async def generate(self, context: AgentContext) -> ModelResponse:
        """实现 application.ports.ModelClient：经统一网关路由、限流与用量审计后返回 ModelResponse。"""
        target = self.resolve_target(context.provider, context.model)
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        if target.offline:
            req_hash = self._hash_payload({"session_id": context.session_id, "messages": len(context.messages)})
            record = await self._reserve_or_raise(
                user_id=context.user_id,
                session_id=context.session_id,
                request_id=request_id,
                request_hash=req_hash,
                provider="offline",
                model="offline",
            )
            response = await self._mock_client.generate(context)
            await self.usage_store.finish(record.id, "succeeded", usage=response.usage or None, upstream_status=200)
            return response

        payload: dict[str, Any] = {
            "model": target.model,
            "temperature": 0.2,
            "messages": OpenAIModelClient._serialize_messages(context),
        }
        if context.tools:
            payload["tools"] = [OpenAIModelClient._serialize_tool(t) for t in context.tools]
            payload["tool_choice"] = "auto"

        data, _headers = await self._dispatch_openai_chat(
            target=target,
            payload=payload,
            user_id=context.user_id,
            session_id=context.session_id,
            request_id=request_id,
        )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GatewayError(
                502,
                "OUTCOME_UNKNOWN",
                f"厂商 `{target.provider}` 返回结构异常（缺少 choices）。",
                provider=target.provider,
                model=target.model,
            )
        choice_msg = choices[0].get("message") or {}
        content = choice_msg.get("content") or ""
        tool_calls = OpenAIModelClient._parse_tool_calls(choice_msg.get("tool_calls") or [])
        usage = normalize_usage(data.get("usage")) or {}
        returned_model = str(data.get("model") or target.model)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            model_name=f"{target.provider}/{returned_model}",
            usage=usage,
        )

    async def generate_stream(
        self,
        context: AgentContext,
        *,
        on_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> ModelResponse:
        """流式调用模型：实时回调文本增量 on_delta，并在结束时返回完整 ModelResponse（含流式 tool_calls 组装）。"""
        target = self.resolve_target(context.provider, context.model)
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        async def _emit(text: str) -> None:
            if not text or on_delta is None:
                return
            res = on_delta(text)
            if inspect.isawaitable(res):
                await res

        if target.offline:
            req_hash = self._hash_payload({"session_id": context.session_id, "messages": len(context.messages), "stream": True})
            record = await self._reserve_or_raise(
                user_id=context.user_id,
                session_id=context.session_id,
                request_id=request_id,
                request_hash=req_hash,
                provider="offline",
                model="offline",
            )
            response = await self._mock_client.generate(context)
            if not response.tool_calls and response.content:
                step = 12
                for idx in range(0, len(response.content), step):
                    await _emit(response.content[idx : idx + step])
                    await asyncio.sleep(0)
            await self.usage_store.finish(record.id, "succeeded", usage=response.usage or None, upstream_status=200)
            return response

        payload: dict[str, Any] = {
            "model": target.model,
            "temperature": 0.2,
            "messages": OpenAIModelClient._serialize_messages(context),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if context.tools:
            payload["tools"] = [OpenAIModelClient._serialize_tool(t) for t in context.tools]
            payload["tool_choice"] = "auto"

        assert target.endpoint is not None
        endpoint = target.endpoint
        req_hash = self._hash_payload({"provider": target.provider, "model": target.model, "payload": payload})
        record = await self._reserve_or_raise(
            user_id=context.user_id,
            session_id=context.session_id,
            request_id=request_id,
            request_hash=req_hash,
            provider=target.provider,
            model=target.model,
        )

        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-ID": request_id,
        }
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"

        content_parts: list[str] = []
        tool_call_buffers: dict[int, dict[str, str]] = {}
        reported_usage: dict[str, int] = {}
        returned_model = target.model
        complete = False

        try:
            async with self._semaphore:
                if self._injected_http_client is not None:
                    stream_ctx = self._injected_http_client.stream("POST", url, json=payload, headers=headers)
                    async with stream_ctx as resp:
                        complete, returned_model = await self._consume_openai_sse(
                            resp,
                            target=target,
                            record_id=record.id,
                            content_parts=content_parts,
                            tool_call_buffers=tool_call_buffers,
                            reported_usage=reported_usage,
                            emit=_emit,
                        )
                else:
                    async with httpx.AsyncClient(timeout=self.settings.gateway.upstream_timeout) as client:
                        async with client.stream("POST", url, json=payload, headers=headers) as resp:
                            complete, returned_model = await self._consume_openai_sse(
                                resp,
                                target=target,
                                record_id=record.id,
                                content_parts=content_parts,
                                tool_call_buffers=tool_call_buffers,
                                reported_usage=reported_usage,
                                emit=_emit,
                            )

            tool_calls = self._finalize_streamed_tool_calls(tool_call_buffers)
            finished_ok = complete or bool(content_parts) or bool(tool_calls)
            await self.usage_store.finish(
                record.id,
                "succeeded" if finished_ok else "unknown",
                usage=reported_usage or None,
                error_code=None if finished_ok else "OUTCOME_UNKNOWN",
                upstream_status=200,
            )
            if not finished_ok:
                raise GatewayError(
                    502,
                    "OUTCOME_UNKNOWN",
                    f"厂商 `{target.provider}` 流式响应提前结束且未返回有效内容。",
                    provider=target.provider,
                    model=target.model,
                )

            return ModelResponse(
                content="".join(content_parts),
                tool_calls=tool_calls,
                model_name=f"{target.provider}/{returned_model}",
                usage=reported_usage,
            )
        except GatewayError:
            raise
        except httpx.ConnectError as exc:
            category = connection_category(exc)
            logger.warning(
                "gateway_stream_connect_failed request_id=%s provider=%s host=%s category=%s",
                request_id,
                target.provider,
                urlsplit(url).hostname,
                category,
            )
            await self.usage_store.finish(record.id, "failed", error_code="UPSTREAM_UNREACHABLE")
            raise GatewayError(
                503,
                "UPSTREAM_UNREACHABLE",
                f"无法连接到厂商 `{target.provider}`（类别：{category}），请检查本机网络或代理配置；网关未自动重试。",
                provider=target.provider,
                model=target.model,
            ) from exc
        except (httpx.HTTPError, ValueError, asyncio.CancelledError) as exc:
            await asyncio.shield(
                self.usage_store.finish(
                    record.id,
                    "unknown",
                    usage=reported_usage or None,
                    error_code="OUTCOME_UNKNOWN",
                )
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GatewayError(
                502,
                "OUTCOME_UNKNOWN",
                f"厂商 `{target.provider}`（模型 `{target.model}`）流式传输中断或响应异常，未自动重发。",
                provider=target.provider,
                model=target.model,
            ) from exc

    async def _consume_openai_sse(
        self,
        resp: httpx.Response,
        *,
        target: ResolvedTarget,
        record_id: str,
        content_parts: list[str],
        tool_call_buffers: dict[int, dict[str, str]],
        reported_usage: dict[str, int],
        emit: Callable[[str], Awaitable[None]],
    ) -> tuple[bool, str]:
        if resp.status_code != 200:
            await resp.aread()
            detail = extract_upstream_message(resp)
            failure = provider_failure(
                resp.status_code,
                provider=target.provider,
                model=target.model,
                upstream_detail=detail,
            )
            await self.usage_store.finish(
                record_id,
                "failed",
                error_code=failure.code,
                upstream_status=resp.status_code,
            )
            raise failure

        complete = False
        returned_model = target.model
        async for raw_line in resp.aiter_lines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if not value:
                continue
            if value == "[DONE]":
                complete = True
                break
            data = json.loads(value)
            if not isinstance(data, dict):
                raise ValueError("invalid provider SSE event")
            if data.get("error"):
                err_obj = data["error"]
                msg = err_obj.get("message", "") if isinstance(err_obj, dict) else str(err_obj)
                raise GatewayError(
                    502,
                    "UPSTREAM_ERROR",
                    f"厂商 `{target.provider}` 流式返回错误：{msg[:180]}",
                    provider=target.provider,
                    model=target.model,
                )
            if isinstance(data.get("model"), str) and data["model"].strip():
                returned_model = data["model"].strip()
            chunk_usage = normalize_usage(data.get("usage"))
            if chunk_usage:
                reported_usage.update(chunk_usage)

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                complete = True
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                content_parts.append(piece)
                await emit(piece)

            raw_calls = delta.get("tool_calls")
            if isinstance(raw_calls, list):
                for item in raw_calls:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("index", 0))
                    buf = tool_call_buffers.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if isinstance(item.get("id"), str) and item["id"]:
                        buf["id"] = item["id"]
                    fn = item.get("function")
                    if isinstance(fn, dict):
                        if isinstance(fn.get("name"), str) and fn["name"]:
                            buf["name"] += fn["name"]
                        if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                            buf["arguments"] += fn["arguments"]

        return complete, returned_model

    @staticmethod
    def _finalize_streamed_tool_calls(buffers: dict[int, dict[str, str]]) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for idx in sorted(buffers.keys()):
            item = buffers[idx]
            name = item.get("name") or "unknown"
            raw_args = item.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
                if not isinstance(args, dict):
                    args = {"_raw": raw_args}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            parsed.append(
                ToolCall(
                    id=item.get("id") or ToolCall(name=name).id,
                    name=name,
                    arguments=args,
                )
            )
        return parsed

    async def invoke_chat_passthrough(
        self,
        body: dict[str, Any],
        *,
        user_id: str = "default",
        session_id: str = "",
        request_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """供 FastAPI `/v1/chat/completions` 使用的 OpenAI 兼容透传入口。"""
        if "api_key" in body or "base_url" in body:
            raise GatewayError(422, "INVALID_REQUEST", "厂商密钥与路由地址为服务端配置，不允许在请求体中透传。")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise GatewayError(422, "INVALID_REQUEST", "messages 必须是非空数组。")
        if body.get("stream") is True:
            raise GatewayError(422, "INVALID_REQUEST", "当前轻量网关暂仅支持非流式请求（stream=false）。")

        max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not (
                1 <= max_tokens <= self.settings.gateway.max_output_tokens
            ):
                raise GatewayError(
                    422,
                    "BUDGET_EXCEEDED",
                    f"单次最大输出 token 超出网关上限（1~{self.settings.gateway.max_output_tokens}）。",
                )

        req_provider = body.get("provider") if isinstance(body.get("provider"), str) else None
        req_model = body.get("model") if isinstance(body.get("model"), str) else None
        target = self.resolve_target(req_provider, req_model)

        if target.offline:
            raise GatewayError(
                503,
                "MODEL_UNAVAILABLE",
                "当前服务处于离线模式，/v1/chat/completions 透传需要配置真实模型提供方。",
                provider="offline",
                model="offline",
            )

        payload = {k: v for k, v in body.items() if k != "provider"}
        payload["model"] = target.model
        return await self._dispatch_openai_chat(
            target=target,
            payload=payload,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
        )

    async def _reserve_or_raise(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        request_hash: str,
        provider: str,
        model: str,
    ) -> UsageRecord:
        record = await self.usage_store.reserve(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            request_hash=request_hash,
            provider=provider,
            model=model,
            rpm=self.settings.gateway.requests_per_minute,
        )
        if record.status == "rejected":
            raise GatewayError(
                429,
                record.error_code or "RATE_LIMITED",
                f"已触发网关每分钟调用频率上限（{self.settings.gateway.requests_per_minute} 次/分），请稍后重试。",
                provider=provider,
                model=model,
            )
        return record

    async def _dispatch_openai_chat(
        self,
        *,
        target: ResolvedTarget,
        payload: dict[str, Any],
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        assert target.endpoint is not None
        endpoint = target.endpoint
        req_hash = self._hash_payload({"provider": target.provider, "model": target.model, "payload": payload})
        record = await self._reserve_or_raise(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            request_hash=req_hash,
            provider=target.provider,
            model=target.model,
        )

        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        }
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"

        try:
            async with self._semaphore:
                if self._injected_http_client is not None:
                    resp = await self._injected_http_client.post(url, json=payload, headers=headers)
                else:
                    async with httpx.AsyncClient(timeout=self.settings.gateway.upstream_timeout) as client:
                        resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code != 200:
                detail = extract_upstream_message(resp)
                failure = provider_failure(
                    resp.status_code,
                    provider=target.provider,
                    model=target.model,
                    upstream_detail=detail,
                )
                await self.usage_store.finish(
                    record.id,
                    "failed",
                    error_code=failure.code,
                    upstream_status=resp.status_code,
                )
                raise failure

            data = resp.json()
            if not isinstance(data, dict) or "choices" not in data:
                raise ValueError("invalid provider response")

            usage = normalize_usage(data.get("usage"))
            await self.usage_store.finish(record.id, "succeeded", usage=usage, upstream_status=200)
            resp_headers = {
                "X-MyAgent-Call-ID": record.id,
                "X-MyAgent-Provider": target.provider,
                "X-MyAgent-Model": str(data.get("model") or target.model),
                "X-Request-ID": request_id,
            }
            return data, resp_headers
        except GatewayError:
            raise
        except httpx.ConnectError as exc:
            category = connection_category(exc)
            logger.warning(
                "gateway_connect_failed request_id=%s provider=%s host=%s category=%s",
                request_id,
                target.provider,
                urlsplit(url).hostname,
                category,
            )
            await self.usage_store.finish(record.id, "failed", error_code="UPSTREAM_UNREACHABLE")
            raise GatewayError(
                503,
                "UPSTREAM_UNREACHABLE",
                f"无法连接到厂商 `{target.provider}`（类别：{category}），请检查本机网络或代理配置；网关未自动重试。",
                provider=target.provider,
                model=target.model,
            ) from exc
        except (httpx.HTTPError, ValueError, asyncio.CancelledError) as exc:
            await asyncio.shield(
                self.usage_store.finish(record.id, "unknown", error_code="OUTCOME_UNKNOWN")
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GatewayError(
                502,
                "OUTCOME_UNKNOWN",
                f"厂商 `{target.provider}`（模型 `{target.model}`）请求超时或响应异常，结果未知，未自动重发。",
                provider=target.provider,
                model=target.model,
            ) from exc

    @staticmethod
    def _hash_payload(data: dict[str, Any]) -> str:
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
