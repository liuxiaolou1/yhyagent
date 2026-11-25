"""模型网关诚实错误分类（参照 model-gateway：明确区分鉴权、权限、模型不可用、网络不可达与未知结果，绝不静默兜底）。"""

from __future__ import annotations

import socket
import ssl

import httpx


class GatewayError(Exception):
    """标准模型网关错误，携带 HTTP 状态码、稳定错误码与安全可读提示。"""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.provider = provider
        self.model = model
        self.upstream_status = upstream_status


def connection_category(error: BaseException) -> str:
    """在不泄露密钥/正文的前提下，对底层传输连接异常进行归类。"""
    chain: list[BaseException] = []
    cause: BaseException | None = error
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen and len(chain) < 8:
        seen.add(id(cause))
        chain.append(cause)
        cause = cause.__cause__ or cause.__context__
    if any(isinstance(item, ssl.SSLError) for item in chain):
        return "tls"
    if any(isinstance(item, socket.gaierror) for item in chain):
        return "dns"
    if any(isinstance(item, httpx.ProxyError) for item in chain):
        return "proxy"
    return "connect"


def extract_upstream_message(response: httpx.Response) -> str:
    """尽力从上游 JSON 错误体中提取简短错误说明（限制长度，避免泄露敏感信息）。"""
    try:
        data = response.json()
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"].strip()[:180]
    if isinstance(err, str):
        return err.strip()[:180]
    if isinstance(data.get("message"), str):
        return data["message"].strip()[:180]
    return ""


def provider_failure(
    status: int,
    *,
    provider: str,
    model: str,
    upstream_detail: str = "",
) -> GatewayError:
    """将上游 HTTP 状态码映射为诚实的 GatewayError。"""
    suffix = f"（上游提示：{upstream_detail}）" if upstream_detail else ""
    if status == 401:
        return GatewayError(
            502,
            "PROVIDER_AUTH_FAILED",
            f"厂商 `{provider}` 鉴权失败（401），请检查 .env 中的 API Key 或区域地址。{suffix}",
            provider=provider,
            model=model,
            upstream_status=status,
        )
    if status == 403:
        return GatewayError(
            502,
            "PROVIDER_PERMISSION_DENIED",
            f"厂商 `{provider}` 拒绝了当前账号或模型 `{model}` 的访问权限（403）。{suffix}",
            provider=provider,
            model=model,
            upstream_status=status,
        )
    if status == 404 or (status == 400 and any(k in upstream_detail.lower() for k in ("model", "not exist", "not found"))):
        return GatewayError(
            422,
            "MODEL_UNAVAILABLE",
            f"厂商 `{provider}` 不支持模型 `{model}` 或该模型不存在（HTTP {status}）。{suffix}",
            provider=provider,
            model=model,
            upstream_status=status,
        )
    if status == 429:
        return GatewayError(
            429,
            "UPSTREAM_RATE_LIMITED",
            f"厂商 `{provider}` 返回限流（429），请稍后重试。{suffix}",
            provider=provider,
            model=model,
            upstream_status=status,
        )
    return GatewayError(
        502,
        "UPSTREAM_ERROR",
        f"厂商 `{provider}`（模型 `{model}`）拒绝了本次请求（HTTP {status}）。{suffix}",
        provider=provider,
        model=model,
        upstream_status=status,
    )
