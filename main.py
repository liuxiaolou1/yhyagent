"""项目启动入口：转发至 CLI 适配器。"""

from app.adapters.cli import run_cli

if __name__ == "__main__":
    run_cli()
