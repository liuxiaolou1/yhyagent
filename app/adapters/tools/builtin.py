"""内置示例工具、本地长期记忆工具与外部检索工具适配。

每个工具继承领域层 domain.Tool，对外声明 ToolSpec（含风险等级与 JSON Schema）。
你可以在这里继续添加自己的业务工具（如 MySQL 查询、Commerce API 调用、MinerU 文档解析等）。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.application.ports import MemoryStore
from app.config import Settings
from app.domain import MemoryScope, Tool, ToolRiskLevel, ToolSpec


class QueryOrderTool(Tool):
    """安全只读工具示例：查询订单状态。"""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="query_order",
            description="只读查询订单当前状态与基本信息。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单编号，例如 ORD-1001"},
                },
                "required": ["order_id"],
            },
            risk_level=ToolRiskLevel.SAFE,
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        order_id = str(arguments.get("order_id", "UNKNOWN"))
        # TODO: 替换为真实数据库或 COMMERCE_API_BASE_URL 查询
        return json.dumps(
            {
                "order_id": order_id,
                "status": "PENDING_REVIEW",
                "amount": 199.0,
                "note": "订单等待人工核验，无需删除",
            },
            ensure_ascii=False,
        )


class DeleteOrderTool(Tool):
    """高危写操作工具示例：用于演示 ToolReviewer 的先审后行与拒绝回传机制。"""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="delete_order",
            description="不可逆删除订单记录（高危操作，必须经过审核批准）。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "要删除的订单编号"},
                    "reason": {"type": "string", "description": "删除原因"},
                },
                "required": ["order_id"],
            },
            risk_level=ToolRiskLevel.DANGEROUS,
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        order_id = str(arguments.get("order_id", "UNKNOWN"))
        reason = str(arguments.get("reason", ""))
        return f"已删除订单 {order_id}（原因: {reason or '无'}）"


class RememberTool(Tool):
    """长期记忆写入工具：按用户和日期落盘到 MEMORY.md 或 YYYY/MM/DD.md。"""

    def __init__(self, memory_store: MemoryStore, *, user_id: str = "default") -> None:
        self._memory_store = memory_store
        self._user_id = user_id

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remember",
            description=(
                "将需要跨会话保留的用户偏好、固定事实或当天重要结论写入本地记忆文件。"
                "scope='profile' 写入长期 MEMORY.md，scope='daily' 按当天日期写入 YYYY/MM/DD.md。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "简洁明确的陈述句，例如“用户偏好使用中文和三层架构”",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["daily", "profile"],
                        "description": "daily 表示按日期记录，profile 表示写入长期画像 MEMORY.md",
                        "default": "daily",
                    },
                },
                "required": ["content"],
            },
            risk_level=ToolRiskLevel.SAFE,
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        content = str(arguments.get("content", "")).strip()
        if not content:
            raise ValueError("content 不能为空")
        raw_scope = str(arguments.get("scope", "daily")).strip().lower()
        scope = MemoryScope.PROFILE if raw_scope == "profile" else MemoryScope.DAILY
        entry = await self._memory_store.remember(
            content,
            user_id=self._user_id,
            scope=scope,
        )
        return json.dumps(
            {
                "saved": True,
                "user_id": entry.user_id,
                "scope": entry.scope.value,
                "source": entry.source,
                "content": entry.content,
            },
            ensure_ascii=False,
        )


class SearchMemoryTool(Tool):
    """长期记忆检索工具：按关键词扫描该用户本地 MEMORY.md 与历史日期笔记。"""

    def __init__(self, memory_store: MemoryStore, *, user_id: str = "default") -> None:
        self._memory_store = memory_store
        self._user_id = user_id

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_memory",
            description="在当前用户的本地长期记忆（MEMORY.md 与历史日期日志）中按关键词检索记录。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词（留空则返回最近的记忆条目）",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回条数（1-20）",
                        "default": 8,
                    },
                },
            },
            risk_level=ToolRiskLevel.SAFE,
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query", "")).strip()
        raw_limit = arguments.get("limit", 8)
        limit = max(1, min(int(raw_limit), 20))
        entries = await self._memory_store.search(query, user_id=self._user_id, limit=limit)
        if not entries:
            return json.dumps({"matches": [], "message": "未检索到相关记忆"}, ensure_ascii=False)
        return json.dumps(
            {
                "matches": [
                    {"source": e.source, "scope": e.scope.value, "content": e.content}
                    for e in entries
                ]
            },
            ensure_ascii=False,
        )


class TavilySearchTool(Tool):
    """联网搜索工具：直接对接 .env 中的 TAVILY_API_KEY。"""

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="通过 Tavily 联网检索最新资料与技术文档。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {"type": "integer", "description": "返回条数（1-5）", "default": 3},
                },
                "required": ["query"],
            },
            risk_level=ToolRiskLevel.MODERATE,
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query", "")).strip()
        max_results = int(arguments.get("max_results", 3))
        if not self._api_key:
            raise RuntimeError("TAVILY_API_KEY 未配置")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max(1, min(max_results, 5)),
                },
            )
            resp.raise_for_status()
            data = resp.json()

        items = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:300],
            }
            for r in data.get("results", [])
        ]
        return json.dumps(items, ensure_ascii=False)


def build_default_tools(
    settings: Settings,
    *,
    memory_store: MemoryStore | None = None,
    user_id: str = "default",
) -> list[Tool]:
    """按配置组装默认工具集合。"""
    tools: list[Tool] = [QueryOrderTool(), DeleteOrderTool()]
    if memory_store is not None:
        tools.append(RememberTool(memory_store, user_id=user_id))
        tools.append(SearchMemoryTool(memory_store, user_id=user_id))
    if settings.tools.tavily_api_key:
        tools.append(TavilySearchTool(settings.tools.tavily_api_key))
    return tools
