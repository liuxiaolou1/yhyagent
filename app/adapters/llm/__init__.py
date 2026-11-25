"""模型适配层：实现应用层 ModelClient 端口。"""

from .mock_client import MockModelClient
from .openai_client import OpenAIModelClient

__all__ = ["OpenAIModelClient", "MockModelClient"]
