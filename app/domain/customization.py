"""领域自定义扩展与 Token 预算治理模型（零外部 IO 依赖）。

参考 IDE Customizations Token 预算机制：
- 按分类统计 memory / rules / skills / builtin_tools / mcp_tools 的 token 占用；
- 超出预算时优先将大体积 rules 降级为文件路径指针（demoted_to_pointer）；
- 若仍超预算，则按体积从大到小自动排除非核心扩展（excluded），并在报告中如实呈现，绝不静默丢弃。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .context import estimate_tokens


class CustomizationCategory(StrEnum):
    MEMORY = "memory"
    RULES = "rules"
    SKILLS = "skills"
    BUILTIN_TOOLS = "builtin_tools"
    MCP_TOOLS = "mcp_tools"


class CustomizationStatus(StrEnum):
    INCLUDED = "included"
    DEMOTED_TO_POINTER = "demoted_to_pointer"
    EXCLUDED = "excluded"


CATEGORY_LABELS: dict[CustomizationCategory, str] = {
    CustomizationCategory.MEMORY: "Memory",
    CustomizationCategory.RULES: "Rules",
    CustomizationCategory.SKILLS: "Skills",
    CustomizationCategory.BUILTIN_TOOLS: "Builtin Tools",
    CustomizationCategory.MCP_TOOLS: "MCP Tools",
}

CATEGORY_ORDER: tuple[CustomizationCategory, ...] = (
    CustomizationCategory.MEMORY,
    CustomizationCategory.RULES,
    CustomizationCategory.SKILLS,
    CustomizationCategory.BUILTIN_TOOLS,
    CustomizationCategory.MCP_TOOLS,
)


class RuleEntry(BaseModel):
    """本地规则文件实体（例如 .data/<user>/rules/*.md）。"""

    id: str
    name: str
    source_path: str
    description: str = ""
    content: str = ""


class SkillSummary(BaseModel):
    """本地技能元数据（渐进式披露：仅提取 SKILL.md 的 name 与 description）。"""

    id: str
    name: str
    description: str
    source_path: str


class CustomizationItem(BaseModel):
    """进入预算治理引擎的单个自定义扩展条目。"""

    id: str
    name: str
    category: CustomizationCategory
    content: str = ""
    description: str = ""
    source_path: str = ""
    pointer_text: str = ""
    core: bool = False
    tokens: int = 0
    original_tokens: int = 0
    ratio: float = 0.0
    status: CustomizationStatus = CustomizationStatus.INCLUDED
    reason: str = ""


class CustomizationItemReport(BaseModel):
    """对外展示的单条扩展明细（不含冗长正文，保留完整状态与降级信息）。"""

    id: str
    name: str
    category: CustomizationCategory
    tokens: int
    original_tokens: int
    ratio: float
    status: CustomizationStatus
    reason: str = ""
    source_path: str = ""
    description: str = ""


class CustomizationCategorySummary(BaseModel):
    """按分类汇总的 token 统计与明细列表。"""

    category: CustomizationCategory
    label: str
    tokens: int = 0
    ratio: float = 0.0
    count: int = 0
    included_count: int = 0
    demoted_count: int = 0
    excluded_count: int = 0
    items: list[CustomizationItemReport] = Field(default_factory=list)


class CustomizationBudgetReport(BaseModel):
    """Customizations 总预算与分类占比报告（对应 GET /api/customizations/usage）。"""

    user_id: str = "default"
    used_tokens: int = 0
    budget_tokens: int = 20000
    usage_ratio: float = 0.0
    over_budget_initially: bool = False
    demoted_count: int = 0
    excluded_count: int = 0
    categories: list[CustomizationCategorySummary] = Field(default_factory=list)


class GovernedCustomizations(BaseModel):
    """预算治理后的运行态输出：包含实际应注入模型的文本、保留工具与完整审计报告。"""

    report: CustomizationBudgetReport
    memory_text: str = ""
    rules_text: str = ""
    skills_text: str = ""
    active_tool_names: list[str] = Field(default_factory=list)
    items: list[CustomizationItem] = Field(default_factory=list)


def build_rule_pointer(item: CustomizationItem) -> str:
    """为大体积规则生成轻量路径指针文本。"""
    if item.pointer_text.strip():
        return item.pointer_text.strip()
    desc = item.description.strip()
    desc_part = f" — {desc}" if desc else ""
    path_part = item.source_path or item.name
    return f"- [规则指针] `{item.name}`（文件：`{path_part}`）{desc_part}"


class CustomizationGovernor:
    """Customizations Token 预算治理器：统计、规则降级为路径指针与超大扩展自动排除。"""

    def __init__(self, *, budget_tokens: int = 20000) -> None:
        self.budget_tokens = max(1, budget_tokens)

    def govern(
        self,
        items: list[CustomizationItem],
        *,
        user_id: str = "default",
    ) -> GovernedCustomizations:
        working: list[CustomizationItem] = []
        for raw in items:
            orig = raw.original_tokens if raw.original_tokens > 0 else estimate_tokens(raw.content)
            working.append(
                raw.model_copy(
                    update={
                        "original_tokens": orig,
                        "tokens": orig,
                        "status": CustomizationStatus.INCLUDED,
                        "reason": "",
                    }
                )
            )

        total_tokens = sum(item.tokens for item in working)
        over_budget_initially = total_tokens > self.budget_tokens

        # 阶段 1：超出预算时，优先将大体积 rules 降级为路径指针
        if total_tokens > self.budget_tokens:
            rule_candidates = sorted(
                (i for i in working if i.category == CustomizationCategory.RULES),
                key=lambda x: x.tokens,
                reverse=True,
            )
            for item in rule_candidates:
                if total_tokens <= self.budget_tokens:
                    break
                pointer = build_rule_pointer(item)
                pointer_tokens = estimate_tokens(pointer)
                if pointer_tokens < item.tokens:
                    saved = item.tokens - pointer_tokens
                    item.content = pointer
                    item.tokens = pointer_tokens
                    item.status = CustomizationStatus.DEMOTED_TO_POINTER
                    item.reason = "超出预算，已降级为文件路径指针"
                    total_tokens -= saved

        # 阶段 2：若仍超出预算，按体积从大到小自动排除非核心扩展（不静默丢弃，报告中显式标记 excluded）
        if total_tokens > self.budget_tokens:
            exclusion_priority = {
                CustomizationCategory.MCP_TOOLS: 0,
                CustomizationCategory.RULES: 1,
                CustomizationCategory.SKILLS: 2,
                CustomizationCategory.MEMORY: 3,
                CustomizationCategory.BUILTIN_TOOLS: 4,
            }
            candidates = sorted(
                (i for i in working if not i.core and i.tokens > 0),
                key=lambda x: (-x.tokens, exclusion_priority.get(x.category, 99)),
            )
            for item in candidates:
                if total_tokens <= self.budget_tokens:
                    break
                total_tokens -= item.tokens
                item.tokens = 0
                item.content = ""
                item.status = CustomizationStatus.EXCLUDED
                item.reason = "超出预算，已自动排除"

        # 计算每项与每类占预算比例
        for item in working:
            item.ratio = round(item.tokens / self.budget_tokens, 6) if self.budget_tokens else 0.0

        categories: list[CustomizationCategorySummary] = []
        for cat in CATEGORY_ORDER:
            cat_items = [i for i in working if i.category == cat]
            cat_tokens = sum(i.tokens for i in cat_items)
            categories.append(
                CustomizationCategorySummary(
                    category=cat,
                    label=CATEGORY_LABELS[cat],
                    tokens=cat_tokens,
                    ratio=round(cat_tokens / self.budget_tokens, 6) if self.budget_tokens else 0.0,
                    count=len(cat_items),
                    included_count=sum(1 for i in cat_items if i.status == CustomizationStatus.INCLUDED),
                    demoted_count=sum(1 for i in cat_items if i.status == CustomizationStatus.DEMOTED_TO_POINTER),
                    excluded_count=sum(1 for i in cat_items if i.status == CustomizationStatus.EXCLUDED),
                    items=[
                        CustomizationItemReport(
                            id=i.id,
                            name=i.name,
                            category=i.category,
                            tokens=i.tokens,
                            original_tokens=i.original_tokens,
                            ratio=i.ratio,
                            status=i.status,
                            reason=i.reason,
                            source_path=i.source_path,
                            description=i.description,
                        )
                        for i in cat_items
                    ],
                )
            )

        used_tokens = sum(i.tokens for i in working)
        report = CustomizationBudgetReport(
            user_id=user_id,
            used_tokens=used_tokens,
            budget_tokens=self.budget_tokens,
            usage_ratio=round(used_tokens / self.budget_tokens, 6) if self.budget_tokens else 0.0,
            over_budget_initially=over_budget_initially,
            demoted_count=sum(1 for i in working if i.status == CustomizationStatus.DEMOTED_TO_POINTER),
            excluded_count=sum(1 for i in working if i.status == CustomizationStatus.EXCLUDED),
            categories=categories,
        )

        memory_parts = [
            i.content.strip()
            for i in working
            if i.category == CustomizationCategory.MEMORY
            and i.status != CustomizationStatus.EXCLUDED
            and i.content.strip()
        ]
        rules_parts = [
            i.content.strip()
            for i in working
            if i.category == CustomizationCategory.RULES
            and i.status != CustomizationStatus.EXCLUDED
            and i.content.strip()
        ]
        skills_parts = [
            i.content.strip()
            for i in working
            if i.category == CustomizationCategory.SKILLS
            and i.status != CustomizationStatus.EXCLUDED
            and i.content.strip()
        ]
        active_tool_names = [
            i.name
            for i in working
            if i.category in (CustomizationCategory.BUILTIN_TOOLS, CustomizationCategory.MCP_TOOLS)
            and i.status != CustomizationStatus.EXCLUDED
        ]

        return GovernedCustomizations(
            report=report,
            memory_text="\n\n".join(memory_parts),
            rules_text="\n\n".join(rules_parts),
            skills_text="\n".join(skills_parts),
            active_tool_names=active_tool_names,
            items=working,
        )
