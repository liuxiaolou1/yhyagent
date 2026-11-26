"""离线 Mock 模型适配器：用于零成本本地验证「工具提出 -> 审核拒绝 -> 换方案执行 -> 最终回复」闭环。"""

from __future__ import annotations

from app.domain import AgentContext, ModelResponse, Role, ToolCall


class MockModelClient:
    """实现 application.ports.ModelClient 端口的离线演示客户端。"""

    async def generate(self, context: AgentContext) -> ModelResponse:
        # 找出本轮（最后一条用户消息之后）已经发生的工具消息
        last_user_idx = -1
        for idx, msg in enumerate(context.messages):
            if msg.role == Role.USER:
                last_user_idx = idx

        user_messages = [m for m in context.messages if m.role == Role.USER]
        last_user_text = user_messages[-1].content if user_messages else ""
        turn_tool_msgs = [
            m for m in context.messages[last_user_idx + 1 :] if m.role == Role.TOOL
        ]

        # 无工具上下文（如侧边追问 / 纯问答模式）：直接返回离线文本回答
        if not context.tools:
            return ModelResponse(
                content=f"（离线侧边回答）关于「{last_user_text}」，已结合引用片段与主会话背景给出说明。",
                model_name="mock/offline",
            )

        # 多轮续聊演示：如果是第 2+ 轮对话，直接引用历史上下文作答
        if len(user_messages) > 1 and not turn_tool_msgs:
            return ModelResponse(
                content=(
                    f"（离线续聊模式）已读取本会话共 {len(context.messages)} 条历史消息。"
                    f"针对你的新问题「{last_user_text}」，我已在原有上下文基础上继续处理完成。"
                ),
                model_name="mock/offline",
            )

        # 第 1 步：故意提出一个高危工具调用，演示系统先审核并拦截
        if len(turn_tool_msgs) == 0:
            return ModelResponse(
                content="我先尝试直接清理问题订单以快速处理。",
                tool_calls=[
                    ToolCall(
                        name="delete_order",
                        arguments={"order_id": "ORD-1001", "reason": "尝试重置异常状态"},
                    )
                ],
                model_name="mock/offline",
            )

        # 第 2 步：收到审核拒绝原因后，改用安全的只读查询工具
        if len(turn_tool_msgs) == 1 and "[REJECTED]" in turn_tool_msgs[0].content:
            return ModelResponse(
                content="收到系统审核拒绝原因，改用只读工具先核查订单状态。",
                tool_calls=[
                    ToolCall(
                        name="query_order",
                        arguments={"order_id": "ORD-1001"},
                    )
                ],
                model_name="mock/offline",
            )

        # 第 3 步：拿到安全工具执行结果后，给出最终回复
        latest_tool_out = turn_tool_msgs[-1].content
        return ModelResponse(
            content=(
                f"已完成处理「{last_user_text}」。\n"
                f"- 删除操作因高危风险已被审核拦截，未产生脏写；\n"
                f"- 通过只读查询获取结果：{latest_tool_out}"
            ),
            model_name="mock/offline",
        )
