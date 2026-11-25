"""本地按用户持久化的自定义模型注册表（支持追加已有厂商模型或注册新的 OpenAI 兼容端点）。

本地文件落盘路径：
    .data/<user_id>/gateway/models.json

注意：api_key 仅保存在本地注册表服务端文件中，任何对外 API 响应都不会回传明文密钥。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


def _sanitize_user(user_id: str) -> str:
    safe = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_"))
    return safe or "default"


class CustomModelConfig(BaseModel):
    """单个持久化自定义模型配置（服务端内部结构）。"""

    id: str
    provider: str
    model: str
    name: str = ""
    vendor: str = "local"
    vendor_label: str = ""
    base_url: str = ""
    api_key: str = Field(default="", repr=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryModelRegistry:
    """内存模型注册表（适合单元测试与 --storage memory）。"""

    def __init__(self, *, user_id: str = "default") -> None:
        self.user_id = _sanitize_user(user_id)
        self._items: dict[str, CustomModelConfig] = {}

    def list_all(self) -> list[CustomModelConfig]:
        return [item.model_copy(deep=True) for item in self._items.values()]

    def get(self, model_id: str) -> CustomModelConfig | None:
        item = self._items.get(model_id)
        return item.model_copy(deep=True) if item else None

    def find_by_provider(self, provider: str) -> CustomModelConfig | None:
        for item in reversed(list(self._items.values())):
            if item.provider == provider and (item.base_url or item.api_key):
                return item.model_copy(deep=True)
        return None

    def upsert(self, config: CustomModelConfig) -> CustomModelConfig:
        self._items[config.id] = config.model_copy(deep=True)
        self._persist()
        return config.model_copy(deep=True)

    def delete(self, model_id: str) -> bool:
        removed = self._items.pop(model_id, None) is not None
        if removed:
            self._persist()
        return removed

    def _persist(self) -> None:
        pass


class LocalFileModelRegistry(InMemoryModelRegistry):
    """本地 JSON 文件持久化注册表：.data/<user_id>/gateway/models.json。"""

    def __init__(self, base_dir: Path | str = Path(".data"), *, user_id: str = "default") -> None:
        super().__init__(user_id=user_id)
        self.base_dir = Path(base_dir)
        self.file_path = self.base_dir / self.user_id / "gateway" / "models.json"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        if not isinstance(raw, list):
            return
        for entry in raw:
            try:
                cfg = CustomModelConfig.model_validate(entry)
            except Exception:  # noqa: BLE001
                continue
            self._items[cfg.id] = cfg

    def _persist(self) -> None:
        payload = [item.model_dump(mode="json") for item in self._items.values()]
        tmp = self.file_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.file_path)
