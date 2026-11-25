"""模型网关用量与调用审计存储（参照 model-gateway：仅记录元数据、请求哈希、状态与实际 usage，绝不落盘 prompt/正文）。

本地文件模式按用户 + 日期分层保存：
    .data/<user_id>/gateway/YYYY/MM/DD/usage.jsonl
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

CallStatus = Literal["running", "succeeded", "failed", "rejected", "unknown"]


def _sanitize_user(user_id: str) -> str:
    safe = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_"))
    return safe or "default"


def normalize_usage(raw: Any) -> dict[str, int] | None:
    """只保留合法非负整数字段；无有效上报时返回 None（usage_source='unknown'）。"""
    if not isinstance(raw, dict):
        return None
    allowed = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    }
    cleaned = {
        k: v
        for k, v in raw.items()
        if k in allowed and isinstance(v, int) and not isinstance(v, bool) and v >= 0
    }
    return cleaned or None


class UsageRecord(BaseModel):
    """单次模型调用审计与用量记录（费用未配置时恒为 None，绝不伪造为 0）。"""

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:12]}")
    cursor: int
    user_id: str = "default"
    session_id: str = ""
    request_id: str = ""
    request_hash: str = ""
    provider: str
    model: str
    protocol: str = "chat"
    status: CallStatus = "running"
    error_code: str | None = None
    upstream_status: int | None = None
    usage: dict[str, int] | None = None
    usage_source: Literal["reported", "unknown"] = "unknown"
    cost: dict[str, Any] | None = None
    cost_status: Literal["reported", "unknown"] = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryUsageStore:
    """内存版用量存储（适合单元测试与 --storage memory）。"""

    def __init__(self, *, user_id: str = "default") -> None:
        self.user_id = _sanitize_user(user_id)
        self._lock = asyncio.Lock()
        self._cursor = 0
        self._records: dict[str, UsageRecord] = {}

    async def reserve(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        request_hash: str,
        provider: str,
        model: str,
        protocol: str = "chat",
        rpm: int = 60,
    ) -> UsageRecord:
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(minutes=1)
            recent = sum(
                1
                for rec in self._records.values()
                if rec.user_id == user_id and rec.status != "rejected" and rec.created_at >= cutoff
            )
            self._cursor += 1
            rejected = recent >= rpm
            record = UsageRecord(
                cursor=self._cursor,
                user_id=_sanitize_user(user_id),
                session_id=session_id,
                request_id=request_id,
                request_hash=request_hash,
                provider=provider,
                model=model,
                protocol=protocol,
                status="rejected" if rejected else "running",
                error_code="RATE_LIMITED" if rejected else None,
                created_at=now,
                updated_at=now,
            )
            self._records[record.id] = record
            self._persist_record(record)
            return record.model_copy(deep=True)

    async def finish(
        self,
        call_id: str,
        status: CallStatus,
        *,
        usage: dict[str, int] | None = None,
        error_code: str | None = None,
        upstream_status: int | None = None,
    ) -> UsageRecord | None:
        async with self._lock:
            record = self._records.get(call_id)
            if record is None:
                return None
            self._cursor += 1
            normalized = normalize_usage(usage)
            record.cursor = self._cursor
            record.status = status
            record.error_code = error_code
            record.upstream_status = upstream_status
            record.usage = normalized
            record.usage_source = "reported" if normalized is not None else "unknown"
            record.updated_at = datetime.now(timezone.utc)
            self._persist_record(record)
            return record.model_copy(deep=True)

    async def rows(self, after: int = 0, *, limit: int = 100) -> list[UsageRecord]:
        async with self._lock:
            matched = [r.model_copy(deep=True) for r in self._records.values() if r.cursor > after]
            matched.sort(key=lambda r: r.cursor)
            return matched[:limit]

    def _persist_record(self, record: UsageRecord) -> None:
        _ = record


class LocalFileUsageStore(InMemoryUsageStore):
    """本地按用户 + 日期 JSONL 落盘的用量存储：.data/<user_id>/gateway/YYYY/MM/DD/usage.jsonl。"""

    def __init__(self, base_dir: Path | str = Path(".data"), *, user_id: str = "default") -> None:
        super().__init__(user_id=user_id)
        self.base_dir = Path(base_dir)
        self.gateway_root = self.base_dir / self.user_id / "gateway"
        self.gateway_root.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.gateway_root.exists():
            return
        for file in sorted(self.gateway_root.rglob("usage.jsonl")):
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = UsageRecord.model_validate_json(line)
                except Exception:  # noqa: BLE001
                    continue
                # 进程重启时，遗留 running 状态按 model-gateway 原则诚实标为 unknown
                if rec.status == "running":
                    rec.status = "unknown"
                    rec.error_code = rec.error_code or "OUTCOME_UNKNOWN"
                existing = self._records.get(rec.id)
                if existing is None or rec.cursor >= existing.cursor:
                    self._records[rec.id] = rec
                if rec.cursor > self._cursor:
                    self._cursor = rec.cursor

    def _usage_file_for(self, record: UsageRecord) -> Path:
        local_dt = record.created_at.astimezone()
        dated_dir = (
            self.base_dir
            / _sanitize_user(record.user_id)
            / "gateway"
            / f"{local_dt.year:04d}"
            / f"{local_dt.month:02d}"
            / f"{local_dt.day:02d}"
        )
        dated_dir.mkdir(parents=True, exist_ok=True)
        return dated_dir / "usage.jsonl"

    def _persist_record(self, record: UsageRecord) -> None:
        path = self._usage_file_for(record)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
