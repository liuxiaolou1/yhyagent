"""项目全局配置：按分组读取 .env，集中管理多模型、外部中间件与循环预算参数。"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


def _load_dotenv(env_path: Path = Path(".env")) -> None:
    """零依赖读取本地 .env 文件（已存在的系统环境变量优先）。"""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class ProviderEndpoint(BaseModel):
    """单个 OpenAI 兼容模型提供方的连接配置。"""

    name: str
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_api_key_here" and self.base_url)


class InfraSettings(BaseModel):
    """外部存储与中间件地址（Mongo / Redis / Milvus / MySQL / OSS）。"""

    mongo_url: str = ""
    mongo_db: str = "myagent"
    redis_url: str = ""
    milvus_url: str = ""
    database_url: str = ""

    oss_endpoint: str = ""
    oss_bucket: str = ""
    oss_access_key_id: str = ""
    oss_access_key_secret: str = ""


class ToolSettings(BaseModel):
    """外部工具与检索服务配置。"""

    tavily_api_key: str = ""
    mcp_dashscope_base_url: str = ""
    mineru_api_key: str = ""
    github_token: str = ""
    commerce_api_base_url: str = ""


class LoopSettings(BaseModel):
    """循环决策链预算与防死循环守卫阈值。"""

    max_cycles: int = Field(default=12, ge=1)
    max_backtracks: int = Field(default=3, ge=0)
    max_repeat_actions: int = Field(default=2, ge=2)
    max_stagnant_steps: int = Field(default=3, ge=2)


class ContextSettings(BaseModel):
    """本地按用户/日期记忆、上下文滚动压缩与自定义扩展预算配置。"""

    default_user_id: str = "default"
    compress_trigger_messages: int = Field(default=24, ge=4)
    compress_trigger_chars: int = Field(default=12000, ge=500)
    compress_keep_recent: int = Field(default=10, ge=2)
    memory_recent_days: int = Field(default=2, ge=0)
    customization_token_budget: int = Field(default=20000, ge=100)


class GatewaySettings(BaseModel):
    """模型网关资源限额（参考 model-gateway：并发信号量、每分钟频率、超时）。"""

    max_concurrent: int = Field(default=4, ge=1)
    requests_per_minute: int = Field(default=60, ge=1)
    max_output_tokens: int = Field(default=8192, ge=1)
    upstream_timeout: float = Field(default=60.0, gt=0)


class Settings(BaseModel):
    # 默认模型配置（向后兼容旧字段名）
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-flash"
    llm_timeout: float = 60.0

    # 多模型提供方映射：可按节点分别指定（如 Planner 用 deepseek，Actor 用 dashscope/qwen-flash）
    providers: dict[str, ProviderEndpoint] = Field(default_factory=dict)

    # 外部组件 / 工具 / 循环预算 / 上下文与记忆 / 模型网关
    infra: InfraSettings = Field(default_factory=InfraSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    loop: LoopSettings = Field(default_factory=LoopSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)

    # 便捷属性，兼容旧代码直接访问 settings.max_cycles 等
    @property
    def max_cycles(self) -> int:
        return self.loop.max_cycles

    @property
    def max_backtracks(self) -> int:
        return self.loop.max_backtracks

    @property
    def max_repeat_actions(self) -> int:
        return self.loop.max_repeat_actions

    @property
    def max_stagnant_steps(self) -> int:
        return self.loop.max_stagnant_steps

    def get_provider(self, name: str = "default") -> ProviderEndpoint:
        if name not in self.providers:
            supported = ", ".join(sorted(self.providers.keys()))
            raise KeyError(f"未知模型提供方 '{name}'，可选: {supported}")
        return self.providers[name]

    @classmethod
    def from_env(cls, env_path: Path = Path(".env")) -> Settings:
        _load_dotenv(env_path)

        default_model = os.getenv("LLM_MODEL") or os.getenv("LLM_DEFAULT_MODEL") or "qwen3.8-flash"
        default_base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        default_api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""

        providers = {
            "default": ProviderEndpoint(
                name="default",
                api_key=default_api_key,
                base_url=default_base_url,
                default_model=default_model,
            ),
            "dashscope": ProviderEndpoint(
                name="dashscope",
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                default_model=os.getenv("LLM_DEFAULT_MODEL", "qwen3.8-flash"),
            ),
            "deepseek": ProviderEndpoint(
                name="deepseek",
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                default_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            ),
            "glm": ProviderEndpoint(
                name="glm",
                api_key=os.getenv("GLM_API_KEY", ""),
                base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
                default_model=os.getenv("GLM_MODEL", "glm-5.4"),
            ),
            "gemini": ProviderEndpoint(
                name="gemini",
                api_key=os.getenv("GEMINI_API_KEY", ""),
                base_url=os.getenv(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai",
                ),
                default_model=os.getenv("GEMINI_MODEL", "gemini-4-neon"),
            ),
            "openai": ProviderEndpoint(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                default_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            ),
        }

        infra = InfraSettings(
            mongo_url=os.getenv("MONGO_URL", ""),
            mongo_db=os.getenv("MONGO_DB", "myagent"),
            redis_url=os.getenv("REDIS_URL", ""),
            milvus_url=os.getenv("MILVUS_URL", ""),
            database_url=os.getenv("DATABASE_URL", ""),
            oss_endpoint=os.getenv("ALIOSS_ENDPOINT", ""),
            oss_bucket=os.getenv("ALIOSS_BUCKET_NAME") or os.getenv("ALIYUN_OSS_BUCKET", ""),
            oss_access_key_id=os.getenv("ALIOSS_ACCESS_KEY_ID") or os.getenv("ALIYUN_OSS_API_ID", ""),
            oss_access_key_secret=os.getenv("ALIOSS_ACCESS_KEY_SECRET") or os.getenv("ALIYUN_OSS_API_KEY", ""),
        )

        tools = ToolSettings(
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            mcp_dashscope_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL", ""),
            mineru_api_key=os.getenv("MINERU_API_KEY", ""),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            commerce_api_base_url=os.getenv("COMMERCE_API_BASE_URL", ""),
        )

        loop = LoopSettings(
            max_cycles=int(os.getenv("MAX_CYCLES", "12")),
            max_backtracks=int(os.getenv("MAX_BACKTRACKS", "3")),
            max_repeat_actions=int(os.getenv("MAX_REPEAT_ACTIONS", "2")),
            max_stagnant_steps=int(os.getenv("MAX_STAGNANT_STEPS", "3")),
        )

        context = ContextSettings(
            default_user_id=os.getenv("DEFAULT_USER_ID", "default"),
            compress_trigger_messages=int(os.getenv("COMPRESS_TRIGGER_MESSAGES", "24")),
            compress_trigger_chars=int(os.getenv("COMPRESS_TRIGGER_CHARS", "12000")),
            compress_keep_recent=int(os.getenv("COMPRESS_KEEP_RECENT", "10")),
            memory_recent_days=int(os.getenv("MEMORY_RECENT_DAYS", "2")),
            customization_token_budget=int(os.getenv("CUSTOMIZATION_TOKEN_BUDGET", "20000")),
        )

        llm_timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))
        gateway = GatewaySettings(
            max_concurrent=int(os.getenv("GATEWAY_MAX_CONCURRENT", "4")),
            requests_per_minute=int(os.getenv("GATEWAY_REQUESTS_PER_MINUTE", "60")),
            max_output_tokens=int(os.getenv("GATEWAY_MAX_OUTPUT_TOKENS", "8192")),
            upstream_timeout=float(os.getenv("GATEWAY_UPSTREAM_TIMEOUT_SECONDS", str(llm_timeout))),
        )

        return cls(
            llm_api_key=default_api_key,
            llm_base_url=default_base_url,
            llm_model=default_model,
            llm_timeout=llm_timeout,
            providers=providers,
            infra=infra,
            tools=tools,
            loop=loop,
            context=context,
            gateway=gateway,
        )

    def masked_summary(self) -> dict[str, dict[str, str]]:
        """返回完全脱敏的配置体检摘要（绝不输出明文密钥）。"""
        return {
            "providers": {
                name: (f"ready (model={ep.default_model})" if ep.is_configured else "not configured")
                for name, ep in self.providers.items()
            },
            "infra": {
                "mongo": self.infra.mongo_url or "not configured",
                "redis": self.infra.redis_url or "not configured",
                "milvus": self.infra.milvus_url or "not configured",
                "mysql": "configured" if self.infra.database_url else "not configured",
                "oss": f"{self.infra.oss_endpoint}/{self.infra.oss_bucket}"
                if self.infra.oss_endpoint
                else "not configured",
            },
            "tools": {
                "tavily": "configured" if self.tools.tavily_api_key else "not configured",
                "mcp_websearch": self.tools.mcp_dashscope_base_url or "not configured",
                "mineru": "configured" if self.tools.mineru_api_key else "not configured",
                "github": "configured" if self.tools.github_token else "not configured",
                "commerce_api": self.tools.commerce_api_base_url or "not configured",
            },
        }
