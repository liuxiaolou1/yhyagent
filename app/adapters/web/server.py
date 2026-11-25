"""基于 FastAPI 的本地 Web 后端：同源提供前端静态资源、多模型网关、会话对话、快照回滚与本地长期记忆接口。"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from app.adapters.gateway import GatewayError, ModelDescriptor, ModelGateway, UsageRecord
from app.application import AgentService
from app.bootstrap import build_agent_service
from app.config import Settings
from app.domain import ContextUsageReport, MemoryEntry, MemoryScope, Session

logger = logging.getLogger(__name__)

ASSET_DIR = Path(__file__).with_name("static")
STATIC_ASSETS: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
}
NEW_SESSION_ID = re.compile(r"[\w-]+")
MAX_BODY_BYTES = 256 * 1024
DOCS_PATHS = ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")


class ApiError(Exception):
    """统一业务异常：返回前端 chat.js 约定的 {"error": "..."} 结构。"""

    def __init__(self, status_code: int, message: str, *, code: str = "API_ERROR") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


class StatusResponse(BaseModel):
    model: str
    offline: bool
    user_id: str
    default_provider: str = "default"
    models: list[ModelDescriptor] = Field(default_factory=list)


class SessionSummaryResponse(BaseModel):
    id: str
    title: str
    user_id: str = "default"
    message_count: int = 0
    summary_count: int = 0
    last_model: str = ""
    updated_at: datetime


class MessageItemResponse(BaseModel):
    role: str
    content: str
    tool_name: str | None = None
    model_name: str | None = None
    intermediate: bool = False


class SnapshotItemResponse(BaseModel):
    id: str
    label: str
    message_count: int
    created_at: datetime


class SessionDetailResponse(BaseModel):
    id: str
    title: str
    user_id: str = "default"
    last_model: str = ""
    summary: str = ""
    snapshots: list[SnapshotItemResponse] = Field(default_factory=list)
    messages: list[MessageItemResponse] = Field(default_factory=list)


class SendMessageRequest(BaseModel):
    content: str
    provider: str | None = None
    model: str | None = None


class RollbackRequest(BaseModel):
    snapshot_id: str


class RememberRequest(BaseModel):
    content: str
    scope: Literal["daily", "profile"] = "daily"


class MemoryItemResponse(BaseModel):
    user_id: str
    scope: str
    source: str
    content: str
    created_at: datetime


class MemoryListResponse(BaseModel):
    user_id: str
    items: list[MemoryItemResponse] = Field(default_factory=list)


class GatewayModelsResponse(BaseModel):
    models: list[ModelDescriptor] = Field(default_factory=list)


class AddModelRequest(BaseModel):
    provider: str
    model: str
    name: str = ""
    vendor_label: str = ""
    base_url: str = ""
    api_key: str = ""


class DeleteModelResponse(BaseModel):
    deleted: bool
    id: str


class GatewayUsageResponse(BaseModel):
    items: list[UsageRecord] = Field(default_factory=list)
    next_cursor: int = 0


def session_summary(session: Session) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        id=session.id,
        title=session.title or session.id,
        user_id=session.user_id,
        message_count=len(session.messages),
        summary_count=len(session.summaries),
        last_model=str(session.metadata.get("last_model") or ""),
        updated_at=session.updated_at,
    )


def session_detail(session: Session) -> SessionDetailResponse:
    return SessionDetailResponse(
        id=session.id,
        title=session.title or session.id,
        user_id=session.user_id,
        last_model=str(session.metadata.get("last_model") or ""),
        summary=session.latest_summary.content if session.latest_summary else "",
        snapshots=[
            SnapshotItemResponse(
                id=s.id,
                label=s.label,
                message_count=s.message_count,
                created_at=s.created_at,
            )
            for s in session.snapshots
        ],
        messages=[
            MessageItemResponse(
                role=message.role.value,
                content=message.content,
                tool_name=message.tool_name,
                model_name=message.model_name,
                intermediate=bool(message.tool_calls),
            )
            for message in session.messages
            if message.role.value != "system"
        ],
    )


def memory_item(entry: MemoryEntry) -> MemoryItemResponse:
    return MemoryItemResponse(
        user_id=entry.user_id,
        scope=entry.scope.value,
        source=entry.source,
        content=entry.content,
        created_at=entry.created_at,
    )


def build_local_hosts(host: str, port: int) -> set[str] | None:
    """仅监听本机回环地址时启用 Host/Origin 校验，防止跨站访问本地 Agent。"""
    normalized = host.strip().lower()
    if normalized not in ("127.0.0.1", "localhost"):
        return None
    hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
    if port == 80:
        hosts.update(("localhost", "127.0.0.1"))
    return hosts


def create_app(
    service: AgentService,
    model_label: str,
    *,
    gateway: ModelGateway | None = None,
    allowed_hosts: set[str] | None = None,
) -> FastAPI:
    """创建 FastAPI 应用实例，注入应用层 AgentService 与统一 ModelGateway。"""
    active_gateway = gateway or (
        service.model_client if isinstance(service.model_client, ModelGateway) else None
    )

    app = FastAPI(
        title="MyAgent API & Model Gateway",
        version="0.1.0",
        description=(
            "MyAgent 本地工作台后端：会话管理、工具审核、按用户/日期长期记忆与上下文压缩，"
            "以及参照 model-gateway 设计的统一多模型网关。"
        ),
    )
    app.state.service = service
    app.state.model_label = model_label
    app.state.gateway = active_gateway
    turn_lock = asyncio.Lock()

    @app.exception_handler(GatewayError)
    async def _handle_gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            {
                "error": exc.message,
                "code": exc.code,
                "provider": exc.provider,
                "model": exc.model,
                "upstream_status": exc.upstream_status,
                "request_id": getattr(request.state, "request_id", ""),
            },
            status_code=exc.status,
        )

    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            {
                "error": exc.message,
                "code": exc.code,
                "request_id": getattr(request.state, "request_id", ""),
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": "请输入有效的消息内容。"}, status_code=400)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Web 请求处理异常: %s", exc)
        return JSONResponse({"error": "读取失败，请稍后再试。"}, status_code=500)

    @app.middleware("http")
    async def _security_and_limits_middleware(request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = req_id

        # 1. Host 与同源校验（仅在配置了 allowed_hosts 时生效）
        if allowed_hosts is not None:
            host = request.headers.get("host", "").lower()
            origin = request.headers.get("origin")
            valid_host = host in allowed_hosts
            valid_origin = origin is None or origin in (f"http://{host}", f"https://{host}")
            if not (valid_host and valid_origin):
                return JSONResponse({"error": "请从 MyAgent 的本地地址访问。"}, status_code=403)

        # 2. 请求体大小保护（最大 256 KB）
        if request.method in ("POST", "PUT", "PATCH"):
            raw_len = request.headers.get("content-length")
            if raw_len is not None:
                try:
                    length = int(raw_len)
                except ValueError:
                    return JSONResponse({"error": "请求头 Content-Length 无效。"}, status_code=400)
                if length > MAX_BODY_BYTES:
                    return JSONResponse({"error": "请求内容过长，请精简后再发送。"}, status_code=413)

        response: Response = await call_next(request)
        response.headers.setdefault("X-Request-ID", req_id)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if request.url.path not in DOCS_PATHS:
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; frame-ancestors 'none'",
            )
        return response

    async def _get_session_safe(session_id: str) -> Session | None:
        try:
            session = await service.get_session(session_id)
        except ValueError:
            if not NEW_SESSION_ID.fullmatch(session_id):
                return None
            raise
        return session if session is not None and session.id == session_id else None

    # ---- 前端静态资源（保持与 Codex 前端路径完全一致） ----
    for route_path, (filename, media_type) in STATIC_ASSETS.items():

        def _make_static_handler(file_path: Path, content_type: str):
            async def _serve() -> FileResponse:
                if not file_path.exists():
                    raise ApiError(404, "找不到请求的页面。")
                return FileResponse(file_path, media_type=content_type)

            return _serve

        app.add_api_route(
            route_path,
            _make_static_handler(ASSET_DIR / filename, media_type),
            methods=["GET"],
            include_in_schema=False,
        )

    # ---- 健康检查与运行状态（兼容 model-gateway 探针） ----
    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"service_id": "myagent-gateway", "status": "ok", "version": "0.1.0"}

    @app.get("/health/ready")
    async def health_ready() -> JSONResponse:
        provider_ready = active_gateway.is_ready if active_gateway is not None else True
        return JSONResponse(
            status_code=200 if provider_ready else 503,
            content={
                "status": "ready" if provider_ready else "not_ready",
                "provider_configured": provider_ready,
                "provider_called": False,
            },
        )

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status() -> StatusResponse:
        models = active_gateway.list_models() if active_gateway is not None else []
        default_provider = active_gateway.default_provider if active_gateway is not None else "default"
        return StatusResponse(
            model=model_label,
            offline=model_label.startswith("mock/offline"),
            user_id=service.user_id,
            default_provider=default_provider,
            models=models,
        )

    # ---- 模型网关端点（参照 D:/project/ai_patent/model-gateway） ----
    @app.get("/api/v1/models", response_model=GatewayModelsResponse)
    async def gateway_models() -> GatewayModelsResponse:
        models = active_gateway.list_models() if active_gateway is not None else []
        return GatewayModelsResponse(models=models)

    @app.post("/api/v1/models", response_model=ModelDescriptor, status_code=201)
    async def gateway_add_model(payload: AddModelRequest) -> ModelDescriptor:
        if active_gateway is None:
            raise GatewayError(503, "MODEL_UNAVAILABLE", "当前服务未挂载 ModelGateway。")
        return active_gateway.add_model(
            provider=payload.provider,
            model=payload.model,
            name=payload.name,
            vendor_label=payload.vendor_label,
            base_url=payload.base_url,
            api_key=payload.api_key,
        )

    @app.delete("/api/v1/models/{model_id:path}", response_model=DeleteModelResponse)
    async def gateway_delete_model(model_id: str) -> DeleteModelResponse:
        if active_gateway is None:
            raise GatewayError(503, "MODEL_UNAVAILABLE", "当前服务未挂载 ModelGateway。")
        removed = active_gateway.delete_model(model_id)
        if not removed:
            raise ApiError(404, "未找到对应的自定义模型（内置 .env 模型不可删除）。")
        return DeleteModelResponse(deleted=True, id=model_id)

    @app.get("/api/v1/usage", response_model=GatewayUsageResponse)
    async def gateway_usage(
        after: int = Query(default=0, ge=0, description="增量游标，只返回 cursor > after 的记录"),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> GatewayUsageResponse:
        if active_gateway is None:
            return GatewayUsageResponse(items=[], next_cursor=after)
        items = await active_gateway.usage_store.rows(after=after, limit=limit)
        next_cursor = items[-1].cursor if items else after
        return GatewayUsageResponse(items=items, next_cursor=next_cursor)

    @app.post("/v1/chat/completions")
    async def gateway_chat_completions(request: Request) -> JSONResponse:
        if active_gateway is None:
            raise GatewayError(503, "MODEL_UNAVAILABLE", "当前服务未挂载 ModelGateway。")
        try:
            body: Any = await request.json()
        except ValueError as exc:
            raise GatewayError(422, "INVALID_REQUEST", "请求体必须是合法 JSON。") from exc
        if not isinstance(body, dict):
            raise GatewayError(422, "INVALID_REQUEST", "请求体必须是 JSON 对象。")

        data, resp_headers = await active_gateway.invoke_chat_passthrough(
            body,
            user_id=service.user_id,
            session_id=request.headers.get("X-MyAgent-Session-ID", ""),
            request_id=request.state.request_id,
        )
        return JSONResponse(content=data, headers=resp_headers)

    # ---- 会话与对话 API ----
    @app.get("/api/sessions", response_model=list[SessionSummaryResponse])
    async def list_sessions() -> list[SessionSummaryResponse]:
        sessions = await service.list_sessions()
        return [session_summary(s) for s in sessions]

    @app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
    async def get_session(session_id: str) -> SessionDetailResponse:
        session = await _get_session_safe(session_id)
        if session is None:
            raise ApiError(404, "会话不存在。")
        return session_detail(session)

    @app.get("/api/sessions/{session_id}/context-usage", response_model=ContextUsageReport)
    async def get_session_context_usage(
        session_id: str,
        model: str | None = Query(default=None, description="可选指定模型名以计算参考上下文窗口占比"),
    ) -> ContextUsageReport:
        session = await _get_session_safe(session_id)
        if session is None:
            raise ApiError(404, "会话不存在。")
        try:
            return await service.get_context_usage(session.id, model=model)
        except KeyError as exc:
            raise ApiError(404, "会话不存在。") from exc

    @app.post("/api/sessions/{session_id}/messages", response_model=SessionDetailResponse)
    async def post_message(session_id: str, payload: SendMessageRequest) -> SessionDetailResponse:
        prompt = payload.content.strip()
        if not prompt:
            raise ApiError(400, "请输入有效的消息内容。")

        if not NEW_SESSION_ID.fullmatch(session_id) and await _get_session_safe(session_id) is None:
            raise ApiError(400, "会话标识无效。")

        # 前置校验目标模型（如未配置或未知厂商，立即诚实报错，不污染会话也不兜底）
        if active_gateway is not None:
            active_gateway.resolve_target(payload.provider, payload.model)

        if turn_lock.locked():
            raise ApiError(409, "Agent 正在处理另一条消息，请稍后发送。")

        async with turn_lock:
            existing_before = await _get_session_safe(session_id)
            prev_msg_count = len(existing_before.messages) if existing_before else 0
            try:
                result = await service.run_turn(
                    prompt,
                    session_id=session_id,
                    provider=payload.provider,
                    model=payload.model,
                )
                session = await service.get_session(result.session_id)
            except GatewayError:
                # 若首轮上游直接拒单（如模型不存在/鉴权失败），自动回滚刚写入的孤立用户消息，方便前端改选模型重发
                current = await _get_session_safe(session_id)
                if (
                    current is not None
                    and len(current.messages) == prev_msg_count + 1
                    and current.snapshots
                ):
                    await service.rollback_session(session_id, current.snapshots[-1].id)
                raise
            except Exception as exc:
                logger.exception("处理会话消息失败: %s", exc)
                raise ApiError(502, "本轮处理未完成，请检查模型连接后重试。") from exc

        if session is None:
            raise ApiError(500, "处理完成后未找到会话。")
        return session_detail(session)

    @app.post("/api/sessions/{session_id}/rollback", response_model=SessionDetailResponse)
    async def rollback_session(session_id: str, payload: RollbackRequest) -> SessionDetailResponse:
        snapshot_id = payload.snapshot_id.strip()
        if not snapshot_id:
            raise ApiError(400, "请指定要回滚的快照 ID。")
        if turn_lock.locked():
            raise ApiError(409, "Agent 正在处理消息，请稍后再回滚。")

        async with turn_lock:
            try:
                await service.rollback_session(session_id, snapshot_id)
            except KeyError as exc:
                raise ApiError(404, str(exc.args[0]) if exc.args else "会话或快照不存在。") from exc
            session = await service.get_session(session_id)

        if session is None:
            raise ApiError(404, "会话不存在。")
        return session_detail(session)

    # ---- 长期记忆 API（按用户 + 日期本地文件） ----
    @app.get("/api/memories", response_model=MemoryListResponse)
    async def list_or_search_memories(
        query: str = Query(default="", description="检索关键词，留空返回最近记忆"),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> MemoryListResponse:
        if service.memory_store is None:
            return MemoryListResponse(user_id=service.user_id, items=[])
        entries = await service.memory_store.search(query, user_id=service.user_id, limit=limit)
        return MemoryListResponse(
            user_id=service.user_id,
            items=[memory_item(e) for e in entries],
        )

    @app.post("/api/memories", response_model=MemoryItemResponse, status_code=201)
    async def create_memory(payload: RememberRequest) -> MemoryItemResponse:
        if service.memory_store is None:
            raise ApiError(503, "当前服务未启用长期记忆存储。")
        content = payload.content.strip()
        if not content:
            raise ApiError(400, "记忆内容不能为空。")
        scope = MemoryScope.PROFILE if payload.scope == "profile" else MemoryScope.DAILY
        entry = await service.memory_store.remember(content, user_id=service.user_id, scope=scope)
        return memory_item(entry)

    return app


def create_default_app() -> FastAPI:
    """供 `uvicorn app.adapters.web.server:create_default_app --factory` 直接启动的默认应用工厂。"""
    settings = Settings.from_env()
    service, model_label = build_agent_service(settings)
    return create_app(service, model_label)
