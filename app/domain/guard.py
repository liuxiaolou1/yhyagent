"""领域防死循环守卫：检测连续重复工具调用与 A->B->A->B 周期震荡。"""

from __future__ import annotations

from pydantic import BaseModel

from .message import ToolCall


class GuardCheckResult(BaseModel):
    allowed: bool = True
    reason: str = ""


class LoopGuard:
    """单轮循环内的动作轨迹守卫，阻止模型反复调用相同工具或在两个动作间震荡。"""

    def __init__(self, *, max_repeat_calls: int = 2) -> None:
        self.max_repeat_calls = max(2, max_repeat_calls)
        self._fingerprints: list[str] = []

    def check_and_record(self, tool_call: ToolCall) -> GuardCheckResult:
        fp = tool_call.fingerprint()
        self._fingerprints.append(fp)

        # 1. 连续重复完全相同的工具与参数
        if len(self._fingerprints) >= self.max_repeat_calls:
            tail = self._fingerprints[-self.max_repeat_calls :]
            if len(set(tail)) == 1:
                return GuardCheckResult(
                    allowed=False,
                    reason=(
                        f"检测到连续 {self.max_repeat_calls} 次重复调用 `{tool_call.name}` 且参数完全一致，"
                        "已拦截以防死循环，请换一种方式或直接总结回复。"
                    ),
                )

        # 2. 周期震荡检测（A->B->A->B 或 A->B->C->A->B->C）
        for period in (2, 3):
            if len(self._fingerprints) >= period * 2:
                prev_win = self._fingerprints[-period * 2 : -period]
                curr_win = self._fingerprints[-period:]
                if prev_win == curr_win:
                    return GuardCheckResult(
                        allowed=False,
                        reason=f"检测到长度为 {period} 的工具调用震荡循环 (A-B-A-B 模式)，已自动拒绝本次调用。",
                    )

        return GuardCheckResult(allowed=True)
