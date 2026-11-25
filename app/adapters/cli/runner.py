"""命令行入口适配器：只负责参数解析、装配组合根与终端展示，不包含核心编排逻辑。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from app.application import AgentService, TurnResult
from app.bootstrap import build_agent_service as assemble_agent_service
from app.config import Settings


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MyAgent 三层架构 CLI（支持工具审核与会话续聊）")
    parser.add_argument("prompt", nargs="?", default="", help="单轮对话输入内容")
    parser.add_argument("--user", "-u", default=None, help="用户标识（默认读取 DEFAULT_USER_ID 或 default）")
    parser.add_argument("--session", "-s", default=None, help="指定 session_id 继续历史对话")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入多轮交互对话模式")
    parser.add_argument("--list-sessions", action="store_true", help="列出已保存的历史会话")
    parser.add_argument("--rollback", default=None, help="配合 --session 使用，回滚到指定快照 ID（如 cp_001）")
    parser.add_argument("--check-config", action="store_true", help="脱敏检查 .env 中模型与外部组件状态")
    parser.add_argument("--offline", action="store_true", help="使用离线 Mock 模型跑通审核与循环闭环")
    parser.add_argument(
        "--provider",
        default="default",
        choices=["default", "dashscope", "deepseek", "glm", "gemini"],
        help="选择模型提供方",
    )
    parser.add_argument(
        "--storage",
        default="file",
        choices=["file", "memory", "mongo"],
        help="选择会话存储实现（默认 file 本地持久化）",
    )
    parser.add_argument(
        "--approve-interactive",
        action="store_true",
        help="遇到高危工具时在终端交互式 y/N 审批（默认策略自动拒绝）",
    )
    return parser.parse_args()


def build_agent_service(settings: Settings, args: argparse.Namespace) -> tuple[AgentService, str]:
    return assemble_agent_service(
        settings,
        offline=args.offline,
        provider=args.provider,
        storage=args.storage,
        user_id=args.user,
        approve_interactive=args.approve_interactive,
    )


def _print_turn(result: TurnResult) -> None:
    if result.tool_events:
        print("\n--- 工具审核与执行轨迹 ---")
        for ev in result.tool_events:
            status = "APPROVED+OK" if (ev.approved and ev.success) else ("REJECTED" if not ev.approved else "ERROR")
            print(f"  [Cycle {ev.cycle}] {ev.tool_name}({ev.arguments}) -> {status}: {ev.summary}")
    print("\n--- 助手回复 ---")
    print(result.reply)
    print(
        f"\n[session_id={result.session_id} | snapshot={result.snapshot_id} | "
        f"cycles={result.cycles} | stopped={result.stopped_reason}]"
    )


async def _async_main(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if args.check_config:
        print(json.dumps(settings.masked_summary(), ensure_ascii=False, indent=2))
        return

    service, model_label = build_agent_service(settings, args)

    if args.list_sessions:
        sessions = await service.list_sessions()
        if not sessions:
            print("暂无历史会话。")
            return
        for s in sessions:
            print(
                f"- {s.id} | user={s.user_id} | msgs={len(s.messages)} | "
                f"summaries={len(s.summaries)} | snapshots={len(s.snapshots)} | title={s.title}"
            )
        return

    if args.rollback:
        if not args.session:
            raise SystemExit("使用 --rollback 时必须同时指定 --session <session_id>")
        snap = await service.rollback_session(args.session, args.rollback)
        print(f"会话 {args.session} 已回滚至快照 {snap.id}（剩余消息数: {snap.message_count}）")
        return

    print(f"=== MyAgent 启动 | 用户: {service.user_id} | 模型: {model_label} | 存储: {args.storage} ===")

    if args.interactive:
        session_id = args.session
        print("进入交互模式，输入 exit/quit 退出。")
        while True:
            user_input = (await asyncio.to_thread(input, "\n你> ")).strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print(f"已退出，下次可用 --session {session_id} 继续对话。")
                break
            result = await service.run_turn(user_input, session_id=session_id)
            session_id = result.session_id
            _print_turn(result)
        return

    prompt = args.prompt.strip() or "帮我处理异常订单 ORD-1001"
    result = await service.run_turn(prompt, session_id=args.session)
    _print_turn(result)


def run_cli() -> None:
    setup_logging()
    args = parse_args()
    asyncio.run(_async_main(args))
