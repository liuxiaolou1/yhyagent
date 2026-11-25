"""规则与人工确认工具审核器。

支持按工具风险等级分层审核：
- SAFE / MODERATE：默认自动放行（也可通过 blocked_tools 配置黑名单）
- DANGEROUS：默认拒绝并返回明确原因给模型；若开启 interactive=True，则在 CLI 中请求人工 y/N 审批。

TODO: 如需引入低成本小模型做语义风控审核，可在此新增 LLMToolReviewer 实现同一 ToolReviewer 端口。
"""

from __future__ import annotations

import asyncio

from app.domain import ReviewDecision, Session, ToolCall, ToolRiskLevel, ToolSpec


class PolicyToolReviewer:
    """实现 application.ports.ToolReviewer 端口。"""

    def __init__(
        self,
        *,
        interactive: bool = False,
        blocked_tools: set[str] | None = None,
    ) -> None:
        self.interactive = interactive
        self.blocked_tools = blocked_tools or set()

    async def review(
        self,
        session: Session,
        tool_spec: ToolSpec,
        tool_call: ToolCall,
    ) -> ReviewDecision:
        _ = session
        if tool_call.name in self.blocked_tools:
            return ReviewDecision(
                approved=False,
                reason=f"工具 `{tool_call.name}` 当前处于策略禁用名单中。",
                reviewer="policy",
            )

        if tool_spec.risk_level in (ToolRiskLevel.SAFE, ToolRiskLevel.MODERATE):
            return ReviewDecision(approved=True, reason="低/中风险工具自动放行", reviewer="policy")

        # 高危工具：非交互模式直接拒绝；交互模式请人工审批
        if not self.interactive:
            return ReviewDecision(
                approved=False,
                reason=(
                    f"工具 `{tool_call.name}` 属于高危操作 (risk={tool_spec.risk_level.value})，"
                    "自动策略拒绝直接执行，请改用安全只读工具或向用户申请明确授权。"
                ),
                reviewer="policy",
            )

        prompt = (
            f"\n[人工审核] 模型请求执行高危工具 `{tool_call.name}`\n"
            f"  参数: {tool_call.arguments}\n"
            f"  是否批准执行? [y/N]: "
        )
        answer = (await asyncio.to_thread(input, prompt)).strip().lower()
        if answer in ("y", "yes"):
            return ReviewDecision(approved=True, reason="人工确认通过", reviewer="human")
        return ReviewDecision(
            approved=False,
            reason="用户人工审批拒绝了本次高危工具调用。",
            reviewer="human",
        )
