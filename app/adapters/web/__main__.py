"""启动 MyAgent 的 FastAPI 本地 Web 工作台。"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from app.bootstrap import build_agent_service
from app.config import Settings

from .server import build_local_hosts, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 MyAgent FastAPI 本地服务与网页")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--user", "-u", default=None, help="用户标识（默认读取 DEFAULT_USER_ID 或 default）")
    parser.add_argument("--offline", action="store_true", help="使用离线演示模型")
    parser.add_argument("--provider", default="default", help="选择已配置的模型提供方")
    parser.add_argument("--storage", choices=("file", "memory", "mongo"), default="file")
    args = parser.parse_args()

    settings = Settings.from_env()
    if not args.offline and args.provider not in settings.providers:
        parser.error(f"未知模型提供方，可选：{', '.join(settings.providers)}")

    logging.basicConfig(level=logging.INFO)
    service, model_label = build_agent_service(
        settings,
        offline=args.offline,
        provider=args.provider,
        storage=args.storage,
        user_id=args.user,
    )
    app = create_app(
        service,
        model_label,
        allowed_hosts=build_local_hosts(args.host, args.port),
    )

    print(
        f"MyAgent FastAPI 正在运行：http://{args.host}:{args.port}/ "
        f"（接口文档：http://{args.host}:{args.port}/docs | 用户：{service.user_id} | 模型：{model_label}）",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
