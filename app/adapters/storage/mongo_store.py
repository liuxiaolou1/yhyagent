"""MongoDB 会话存储适配器：对接 .env 中的 MONGO_URL 实现 SessionStore 端口。"""

from __future__ import annotations

import asyncio
from typing import Any

from app.domain import Session


class MongoSessionStore:
    """实现 application.ports.SessionStore 端口（懒加载 pymongo，未安装时不影响其他存储）。"""

    def __init__(self, mongo_url: str, db_name: str = "myagent", collection: str = "sessions") -> None:
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.collection_name = collection
        self._client: Any = None

    def _get_collection(self) -> Any:
        if not self.mongo_url:
            raise RuntimeError("MONGO_URL 未配置，无法使用 MongoSessionStore。")
        if self._client is None:
            try:
                from pymongo import MongoClient  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("请先安装 pymongo: uv pip install pymongo") from exc
            self._client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=3000)
        return self._client[self.db_name][self.collection_name]

    async def get(self, session_id: str) -> Session | None:
        def _query() -> dict[str, Any] | None:
            return self._get_collection().find_one({"_id": session_id})

        doc = await asyncio.to_thread(_query)
        if not doc:
            return None
        doc.pop("_id", None)
        return Session.model_validate(doc)

    async def save(self, session: Session) -> None:
        payload = session.model_dump(mode="json")
        payload["_id"] = session.id

        def _upsert() -> None:
            self._get_collection().replace_one({"_id": session.id}, payload, upsert=True)

        await asyncio.to_thread(_upsert)

    async def list_sessions(self) -> list[Session]:
        def _list() -> list[dict[str, Any]]:
            cursor = self._get_collection().find().sort("updated_at", -1).limit(50)
            return list(cursor)

        docs = await asyncio.to_thread(_list)
        sessions: list[Session] = []
        for doc in docs:
            doc.pop("_id", None)
            sessions.append(Session.model_validate(doc))
        return sessions
