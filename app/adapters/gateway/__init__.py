"""多模型统一网关适配层（参照 D:/project/ai_patent/model-gateway）。"""

from .errors import GatewayError
from .gateway import ModelDescriptor, ModelGateway, ResolvedTarget, canonical_provider
from .model_registry import CustomModelConfig, InMemoryModelRegistry, LocalFileModelRegistry
from .usage_store import InMemoryUsageStore, LocalFileUsageStore, UsageRecord

__all__ = [
    "GatewayError",
    "ModelDescriptor",
    "ModelGateway",
    "ResolvedTarget",
    "canonical_provider",
    "CustomModelConfig",
    "InMemoryModelRegistry",
    "LocalFileModelRegistry",
    "InMemoryUsageStore",
    "LocalFileUsageStore",
    "UsageRecord",
]
