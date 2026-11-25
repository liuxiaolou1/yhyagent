"""为命令行和 Web 入口装配 Agent 服务与统一模型网关。"""

from pathlib import Path

from app.adapters.context import HeuristicContextCompressor, LLMContextCompressor
from app.adapters.gateway import (
    InMemoryModelRegistry,
    InMemoryUsageStore,
    LocalFileModelRegistry,
    LocalFileUsageStore,
    ModelGateway,
)
from app.adapters.review import PolicyToolReviewer
from app.adapters.storage import (
    InMemoryMemoryStore,
    InMemorySessionStore,
    JsonFileSessionStore,
    LocalFileMemoryStore,
    MongoSessionStore,
)
from app.adapters.tools import build_default_tools
from app.application import AgentService, ContextCompressor, MemoryStore, SessionStore
from app.config import Settings


def build_agent_service(
    settings: Settings,
    *,
    offline: bool = False,
    provider: str = "default",
    storage: str = "file",
    user_id: str | None = None,
    approve_interactive: bool = False,
    data_dir: Path | str = Path(".data"),
) -> tuple[AgentService, str]:
    active_user = (user_id or settings.context.default_user_id or "default").strip() or "default"

    session_store: SessionStore
    memory_store: MemoryStore
    if storage == "memory":
        session_store = InMemorySessionStore()
        memory_store = InMemoryMemoryStore(user_id=active_user)
        usage_store = InMemoryUsageStore(user_id=active_user)
        model_registry = InMemoryModelRegistry(user_id=active_user)
    elif storage == "mongo":
        session_store = MongoSessionStore(
            mongo_url=settings.infra.mongo_url,
            db_name=settings.infra.mongo_db,
        )
        memory_store = LocalFileMemoryStore(base_dir=data_dir, user_id=active_user)
        usage_store = LocalFileUsageStore(base_dir=data_dir, user_id=active_user)
        model_registry = LocalFileModelRegistry(base_dir=data_dir, user_id=active_user)
    else:
        session_store = JsonFileSessionStore(base_dir=data_dir, user_id=active_user)
        memory_store = LocalFileMemoryStore(base_dir=data_dir, user_id=active_user)
        usage_store = LocalFileUsageStore(base_dir=data_dir, user_id=active_user)
        model_registry = LocalFileModelRegistry(base_dir=data_dir, user_id=active_user)

    has_any_provider = any(ep.is_configured for ep in settings.providers.values()) or bool(model_registry.list_all())
    use_offline = offline or not has_any_provider
    gateway = ModelGateway(
        settings=settings,
        default_provider=provider,
        offline=use_offline,
        usage_store=usage_store,
        model_registry=model_registry,
    )
    model_label = gateway.default_label if not (not offline and not has_any_provider) else "mock/offline (未检测到有效 API Key)"

    context_compressor: ContextCompressor
    if use_offline:
        context_compressor = HeuristicContextCompressor()
    else:
        context_compressor = LLMContextCompressor(model_client=gateway)

    service = AgentService(
        model_client=gateway,
        session_store=session_store,
        tool_reviewer=PolicyToolReviewer(interactive=approve_interactive),
        tools=build_default_tools(settings, memory_store=memory_store, user_id=active_user),
        user_id=active_user,
        memory_store=memory_store,
        context_compressor=context_compressor,
        max_cycles=settings.max_cycles,
        max_repeat_calls=settings.max_repeat_actions,
        compress_trigger_messages=settings.context.compress_trigger_messages,
        compress_trigger_chars=settings.context.compress_trigger_chars,
        compress_keep_recent=settings.context.compress_keep_recent,
        memory_recent_days=settings.context.memory_recent_days,
    )
    return service, model_label
