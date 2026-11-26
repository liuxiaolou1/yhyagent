"""Customizations Token 预算监控、自动降级（路径指针 / 自动排除）与 load_skill 按需加载单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.adapters.review import PolicyToolReviewer
from app.adapters.storage import (
    InMemorySessionStore,
    InMemorySideChatStore,
    LocalFileCustomizationStore,
    LocalFileMemoryStore,
)
from app.adapters.tools import LoadSkillTool, QueryOrderTool, RememberTool
from app.adapters.web.server import create_app
from app.application import AgentService
from app.bootstrap import build_agent_service
from app.config import ContextSettings, Settings
from app.domain import (
    AgentContext,
    CustomizationCategory,
    CustomizationGovernor,
    CustomizationItem,
    CustomizationStatus,
    MemoryScope,
    ModelResponse,
    Tool,
    ToolCall,
    ToolRiskLevel,
    ToolSpec,
)


class DummyMcpTool(Tool):
    """用于验证 mcp_tools 分类与超大 MCP 扩展超预算自动排除的模拟工具。"""

    def __init__(self, name: str = "mcp_big_search", desc_repeat: int = 1) -> None:
        self._name = name
        self._desc = ("通过外部 MCP 服务检索大量文档并返回结构化字段。" * desc_repeat).strip()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=self._desc,
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            risk_level=ToolRiskLevel.SAFE,
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        return f"mcp:{arguments.get('q', '')}"


class CaptureModelClient:
    """记录每次发给模型的 AgentContext，支持脚本化返回工具调用。"""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.contexts: list[AgentContext] = []
        self._responses = list(responses or [])

    async def generate(self, context: AgentContext) -> ModelResponse:
        self.contexts.append(context.model_copy(deep=True))
        if self._responses:
            return self._responses.pop(0)
        return ModelResponse(content="已按治理后的上下文完成回复。")


class TestCustomizationsBudgetAndSkills(unittest.IsolatedAsyncioTestCase):
    async def test_normal_budget_breakdown_and_missing_dirs_zero_error(self) -> None:
        """1. 正常预算内各分类 token 统计与明细；且尚无 rules/skills 目录时零报错运行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            mem_store = LocalFileMemoryStore(base, user_id="alice")
            await mem_store.remember("用户偏好简洁中文回复", user_id="alice", scope=MemoryScope.PROFILE)
            cust_store = LocalFileCustomizationStore(base, user_id="alice")

            # 目录尚未创建时也应零报错返回空 rules/skills，并正确统计 memory 与 builtin_tools
            service = AgentService(
                model_client=CaptureModelClient(),
                session_store=InMemorySessionStore(),
                tool_reviewer=PolicyToolReviewer(),
                tools=[
                    QueryOrderTool(),
                    RememberTool(mem_store, user_id="alice"),
                    LoadSkillTool(cust_store, user_id="alice"),
                    DummyMcpTool("mcp_docs"),
                ],
                user_id="alice",
                memory_store=mem_store,
                customization_store=cust_store,
                customization_token_budget=20000,
            )
            empty_report = await service.get_customization_usage()
            by_cat = {c.category: c for c in empty_report.categories}
            self.assertEqual(by_cat[CustomizationCategory.RULES].count, 0)
            self.assertEqual(by_cat[CustomizationCategory.SKILLS].count, 0)
            self.assertGreater(by_cat[CustomizationCategory.MEMORY].tokens, 0)
            self.assertGreater(by_cat[CustomizationCategory.BUILTIN_TOOLS].tokens, 0)
            self.assertGreater(by_cat[CustomizationCategory.MCP_TOOLS].tokens, 0)
            # 无 skills 时 load_skill 自动不计入工具清单
            builtin_names = [i.name for i in by_cat[CustomizationCategory.BUILTIN_TOOLS].items]
            self.assertNotIn("load_skill", builtin_names)

            # 创建本地 rules/*.md 和 skills/*/SKILL.md
            rules_dir = base / "alice" / "rules"
            rules_dir.mkdir(parents=True, exist_ok=True)
            (rules_dir / "coding.md").write_text(
                "---\nname: coding-style\ndescription: 三层整洁架构规范\n---\n始终遵守依赖反转，零 LangChain 依赖。\n",
                encoding="utf-8",
            )
            skill_dir = base / "alice" / "skills" / "deploy-helper"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: deploy-helper\ndescription: 自动化部署检查清单\n---\n# 详细步骤\n" + ("步骤说明内容。" * 80),
                encoding="utf-8",
            )

            full_report = await service.get_customization_usage()
            self.assertFalse(full_report.over_budget_initially)
            self.assertEqual(full_report.demoted_count, 0)
            self.assertEqual(full_report.excluded_count, 0)
            self.assertEqual(full_report.budget_tokens, 20000)

            full_by_cat = {c.category: c for c in full_report.categories}
            self.assertEqual(full_by_cat[CustomizationCategory.RULES].count, 1)
            self.assertEqual(full_by_cat[CustomizationCategory.SKILLS].count, 1)
            # 渐进式披露：skills 仅统计 name + description，应远小于完整 SKILL.md 体积
            skill_item = full_by_cat[CustomizationCategory.SKILLS].items[0]
            self.assertEqual(skill_item.name, "deploy-helper")
            self.assertLess(skill_item.tokens, 100)
            # 有 skills 时 load_skill 自动激活
            builtin_names_after = [i.name for i in full_by_cat[CustomizationCategory.BUILTIN_TOOLS].items]
            self.assertIn("load_skill", builtin_names_after)

    async def test_over_budget_demotes_large_rules_and_excludes_oversized_extensions(self) -> None:
        """2. 超出预算时大体积 rule 自动降级为路径指针，仍超预算时自动排除超大扩展，并真实裁剪发给模型的上下文。"""
        # 2a. 纯领域治理器验证两阶段降级
        big_rule = CustomizationItem(
            id="rule:huge",
            name="huge-rule",
            category=CustomizationCategory.RULES,
            source_path="alice/rules/huge.md",
            description="超长架构规范文档",
            content="这是一段非常长的项目规则正文内容。" * 120,
        )
        big_mcp = CustomizationItem(
            id="tool:mcp_giant",
            name="mcp_giant",
            category=CustomizationCategory.MCP_TOOLS,
            description="超大 MCP 工具",
            content="超长工具描述与参数定义说明。" * 100,
        )
        small_skill = CustomizationItem(
            id="skill:quick",
            name="quick-skill",
            category=CustomizationCategory.SKILLS,
            description="轻量技能",
            content="- `quick-skill`: 轻量技能",
        )

        # 预算设为 120：big_rule 先降级为路径指针，big_mcp 依然超预算被自动排除，small_skill 保留
        governed = CustomizationGovernor(budget_tokens=120).govern(
            [big_rule, big_mcp, small_skill],
            user_id="alice",
        )
        self.assertTrue(governed.report.over_budget_initially)
        self.assertLessEqual(governed.report.used_tokens, 120)
        self.assertEqual(governed.report.demoted_count, 1)
        self.assertEqual(governed.report.excluded_count, 1)

        by_id = {item.id: item for cat in governed.report.categories for item in cat.items}
        self.assertEqual(by_id["rule:huge"].status, CustomizationStatus.DEMOTED_TO_POINTER)
        self.assertLess(by_id["rule:huge"].tokens, by_id["rule:huge"].original_tokens)
        self.assertIn("alice/rules/huge.md", governed.rules_text)
        self.assertNotIn("非常长的项目规则正文内容", governed.rules_text)

        self.assertEqual(by_id["tool:mcp_giant"].status, CustomizationStatus.EXCLUDED)
        self.assertEqual(by_id["tool:mcp_giant"].tokens, 0)
        self.assertGreater(by_id["tool:mcp_giant"].original_tokens, 0)
        self.assertNotIn("mcp_giant", governed.active_tool_names)
        self.assertEqual(by_id["skill:quick"].status, CustomizationStatus.INCLUDED)

        # 2b. 端到端验证 AgentService 组装上下文时严格执行降级与排除
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            rules_dir = base / "alice" / "rules"
            rules_dir.mkdir(parents=True, exist_ok=True)
            (rules_dir / "big_rule.md").write_text(
                "---\nname: big-rule\ndescription: 规范指针说明\n---\n" + ("详细条款内容不会全文注入。" * 80),
                encoding="utf-8",
            )
            cust_store = LocalFileCustomizationStore(base, user_id="alice")
            capture_model = CaptureModelClient()
            service = AgentService(
                model_client=capture_model,
                session_store=InMemorySessionStore(),
                tool_reviewer=PolicyToolReviewer(),
                tools=[QueryOrderTool(), DummyMcpTool("mcp_oversized", desc_repeat=60)],
                user_id="alice",
                customization_store=cust_store,
                customization_token_budget=260,
            )

            await service.run_turn("检查一下当前上下文裁剪", session_id="gov_s1")
            sent_ctx = capture_model.contexts[-1]
            self.assertIn("alice/rules/big_rule.md", sent_ctx.instruction)
            self.assertNotIn("详细条款内容不会全文注入", sent_ctx.instruction)
            sent_tool_names = [t.name for t in sent_ctx.tools]
            self.assertIn("query_order", sent_tool_names)
            self.assertNotIn("mcp_oversized", sent_tool_names)

    async def test_load_skill_tool_in_main_and_side_chat(self) -> None:
        """4. 验证 load_skill 按需加载完整 SKILL.md，且在只读侧边分支中也可用（同时继续拦截 remember）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            skill_dir = base / "alice" / "skills" / "release-check"
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_body = "---\nname: release-check\ndescription: 发版前自检清单\n---\n# Checklist\n1. 跑通全量测试\n2. 核对 .env\n"
            (skill_dir / "SKILL.md").write_text(skill_body, encoding="utf-8")

            cust_store = LocalFileCustomizationStore(base, user_id="alice")
            mem_store = LocalFileMemoryStore(base, user_id="alice")
            load_tool = LoadSkillTool(cust_store, user_id="alice")

            # 直接调用工具读取正文
            raw_res = await load_tool.execute({"name": "release-check"})
            parsed = json.loads(raw_res)
            self.assertTrue(parsed["found"])
            self.assertIn("跑通全量测试", parsed["content"])

            # 在侧边只读模式中通过模型调用 load_skill，验证只读可用且不污染主会话
            side_model = CaptureModelClient(
                responses=[
                    ModelResponse(
                        content="我先按需加载 release-check 技能说明。",
                        tool_calls=[ToolCall(name="load_skill", arguments={"name": "release-check"})],
                    ),
                    ModelResponse(content="发版前需要跑通全量测试并核对 .env。"),
                ]
            )
            service = AgentService(
                model_client=side_model,
                session_store=InMemorySessionStore(),
                side_chat_store=InMemorySideChatStore(user_id="alice"),
                tool_reviewer=PolicyToolReviewer(),
                tools=[
                    QueryOrderTool(),
                    RememberTool(mem_store, user_id="alice"),
                    load_tool,
                ],
                user_id="alice",
                memory_store=mem_store,
                customization_store=cust_store,
            )
            side_allowed = service.side_chat_allowed_tools()
            self.assertIn("load_skill", side_allowed)
            self.assertNotIn("remember", side_allowed)

            res = await service.run_side_turn("发版前要检查什么？", parent_session_id="p_1")
            self.assertEqual(res.stopped_reason, "completed")
            self.assertEqual(len(res.tool_events), 1)
            self.assertEqual(res.tool_events[0].tool_name, "load_skill")
            self.assertTrue(res.tool_events[0].approved)
            self.assertTrue(res.tool_events[0].success)
            self.assertIn("跑通全量测试", res.tool_events[0].summary)

    def test_customizations_usage_api_endpoint(self) -> None:
        """3. 验证 GET /api/customizations/usage 返回结构与降级字段。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            rules_dir = base / "yhy" / "rules"
            rules_dir.mkdir(parents=True, exist_ok=True)
            (rules_dir / "style.md").write_text(
                "---\nname: style-guide\ndescription: 代码风格说明\n---\n" + ("规则细则内容。" * 60),
                encoding="utf-8",
            )
            skill_dir = base / "yhy" / "skills" / "sql-audit"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: sql-audit\ndescription: SQL 慢查询审查技能\n---\n详细 SQL 审查步骤。",
                encoding="utf-8",
            )

            settings = Settings(context=ContextSettings(customization_token_budget=350))
            service, model_label = build_agent_service(
                settings,
                offline=True,
                storage="file",
                user_id="yhy",
                data_dir=base,
            )
            client = TestClient(create_app(service, model_label))

            resp = client.get("/api/customizations/usage")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["user_id"], "yhy")
            self.assertEqual(data["budget_tokens"], 350)
            self.assertLessEqual(data["used_tokens"], 350)
            self.assertIn("usage_ratio", data)
            self.assertIn("categories", data)

            cats = {c["category"]: c for c in data["categories"]}
            self.assertEqual(set(cats.keys()), {"memory", "rules", "skills", "builtin_tools", "mcp_tools"})
            self.assertEqual(cats["rules"]["items"][0]["status"], "demoted_to_pointer")
            self.assertEqual(cats["skills"]["count"], 1)
            self.assertEqual(cats["skills"]["items"][0]["name"], "sql-audit")
            self.assertGreaterEqual(data["demoted_count"], 1)


if __name__ == "__main__":
    unittest.main()
