# MyAgent — 三层架构循环决策链 Agent 后端

基于**领域层 / 应用层 / 适配层**三层整洁架构（依赖反转，零 LangChain 依赖）搭建的 Agent 后端项目骨架。

## 运行流程

```text
用户输入 + session_id
   │
   ▼
[SessionStore] 加载历史会话 -> 记录本轮快照 -> 追加 User 消息并落盘
   │
   ▼
┌────────────── 循环决策链（受 max_cycles 上限保护） ──────────────┐
│  1. 组装 AgentContext（指令 + 历史消息 + 可用工具）               │
│  2. ModelClient.generate(context) -> 文本回复 或 ToolCalls       │
│  3. 立即保存 Assistant 消息到 SessionStore                       │
│  4. 若无 ToolCalls -> 返回最终回复，本轮结束                      │
│  5. 若有 ToolCalls -> 逐个执行：                                 │
│       LoopGuard 防死循环检查 -> ToolReviewer 审核                │
│         ├─ 审核通过 -> 执行工具 -> 写入 ToolResult 消息并落盘    │
│         └─ 审核拒绝 -> 不执行   -> 写入拒绝原因消息并落盘        │
│     交回模型继续下一轮决策                                       │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼（达到 max_cycles 或模型给出最终回复）
持续持久化完成，下次传入相同 session_id 即可接着聊（或按 snapshot_id 回滚）
```

## 三层目录结构

| 层 | 目录 | 职责 |
| --- | --- | --- |
| **领域层** | `app/domain/` | 零外部依赖。定义 `Session`（含快照回滚与滚动摘要 `ContextSummary`）、`MemoryEntry` / `MemoryScope`、`AgentContext`（含记忆/摘要拼装与旧工具输出折叠）、`Tool` / `ToolSpec` / `ReviewDecision`、`Message` / `ToolCall` / `ToolResult` 以及 `LoopGuard` 防震荡规则。 |
| **应用层** | `app/application/` | 定义 `ModelClient`、`SessionStore`、`MemoryStore`、`ContextCompressor`、`ToolReviewer` 端口协议（`ports.py`），由 `AgentService` 编排记忆加载、上下文压缩、模型调用、工具审核、持续保存与循环控制。 |
| **适配层** | `app/adapters/` | 对接外部实现：`llm/`（OpenAI 兼容多模型 Function Calling + 离线 Mock）、`storage/`（按用户/日期分层的本地 JSON 会话与 Markdown 记忆 / 内存 / MongoDB）、`context/`（LLM 与离线规则滚动摘要）、`tools/`（示例工具 + `remember`/`search_memory` + Tavily 搜索）、`review/`（规则与人工审批）、`cli/` 与本地 `web/` 入口。 |

```text
Myagent/
├── main.py                              # CLI 启动入口
├── app/
│   ├── bootstrap.py                     # CLI 和 Web 共用的服务装配
│   ├── config.py                        # 分组配置（多模型 / Mongo / Redis / Milvus / 工具）
│   ├── domain/
│   │   ├── message.py                   # Role, Message, ToolCall, ToolResult, ModelResponse
│   │   ├── tool.py                      # Tool, ToolSpec, ToolRiskLevel, ReviewDecision
│   │   ├── session.py                   # Session, SessionSnapshot（快照回滚）
│   │   ├── context.py                   # AgentContext（指令 + 历史窗口 + 工具列表）
│   │   └── guard.py                     # LoopGuard（重复调用与 A-B-A-B 震荡检测）
│   ├── application/
│   │   ├── ports.py                     # ModelClient, SessionStore, ToolReviewer 端口接口
│   │   └── agent_service.py             # AgentService 核心编排服务
│   └── adapters/
│       ├── llm/                         # openai_client.py, mock_client.py
│       ├── storage/                     # file_store.py, memory_store.py, mongo_store.py
│       ├── tools/                       # builtin.py（QueryOrder / DeleteOrder / TavilySearch）
│       ├── review/                      # policy_reviewer.py（策略与人工确认审核）
│       ├── cli/                         # runner.py（单轮/交互/续聊/回滚命令行入口）
│       └── web/                         # 本地会话 API 和聊天界面
└── tests/
    └── test_agent_service.py            # 核心编排、审核拒绝回传、持续保存续聊、回滚与守卫测试
```

## 快速开始

在项目目录启动基于 **FastAPI + Uvicorn** 的本地工作台，然后打开 [MyAgent](http://127.0.0.1:8000/)（交互式接口文档：[Swagger UI](http://127.0.0.1:8000/docs)）。

```powershell
py -m app.adapters.web --offline
# 或使用 uvicorn 工厂模式启动（支持 --reload）
uvicorn app.adapters.web.server:create_default_app --factory --reload --port 8000
```

输入框下方可以按厂商选择模型。模型菜单支持新增和删除自定义模型，模型列表由后端提供，浏览器记住当前选择。Web 与 CLI 默认按用户和日期落盘到本地 `.data/<user>/`：
- 会话历史（含快照与滚动摘要）：`.data/<user>/sessions/YYYY/MM/DD/<session_id>.json`（兼容读取旧 `.data/sessions/*.json`）
- 长期记忆：`.data/<user>/memory/MEMORY.md`（长期偏好）与 `.data/<user>/memory/YYYY/MM/DD.md`（每日记录）

可通过 `--user <用户名>`（或 `.env` 中的 `DEFAULT_USER_ID`）切换本地用户目录。使用已配置的模型时，去掉 `--offline` 后启动，再在输入框下方选择厂商和模型。`--provider deepseek` 等启动参数可指定默认提供方。

### 命令行

```powershell
# 1. 脱敏检查 .env 配置状态
py main.py --check-config

# 2. 离线体验「提出高危工具 -> 审核拒绝 -> 改用安全工具 -> 最终回复」完整闭环
py main.py --offline --session demo_01 "帮我处理异常订单 ORD-1001"

# 3. 用相同 session_id 接着聊（历史消息已持续保存在 .data/sessions/demo_01.json）
py main.py --offline --session demo_01 "那帮我总结一下刚才的结果"

# 4. 查看所有已保存会话 / 回滚到某个快照
py main.py --list-sessions
py main.py --session demo_01 --rollback cp_002

# 5. 调用真实模型（默认 .env 中的 qwen-flash，或切 --provider deepseek）
py main.py --session live_01 "帮我查一下订单 ORD-1001 的状态"
py main.py --provider deepseek --interactive

# 6. 切换 MongoDB 存储（需安装 pymongo 且本地 Mongo 已启动）
py main.py --storage mongo --session mongo_01 "测试 MongoDB 持久化"

# 7. 运行全部单元测试
py -m unittest discover -s tests -v
```
