"""内存会话存储适配器（适合单元测试或无状态临时运行）。"""

from __future__ import annotations

from app.domain import Session


class InMemorySessionStore:
    """实现 application.ports.SessionStore 端口。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        return session.model_copy(deep=True) if session else None

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session.model_copy(deep=True)

    async def list_sessions(self) -> list[Session]:
        return sorted(
            (s.model_copy(deep=True) for s in self._sessions.values()),
            key=lambda s: s.updated_at,
            reverse=True,
        )
