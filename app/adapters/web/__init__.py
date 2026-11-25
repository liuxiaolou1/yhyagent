"""MyAgent FastAPI Web 适配层入口。"""

from .server import create_app, create_default_app

__all__ = ["create_app", "create_default_app"]
