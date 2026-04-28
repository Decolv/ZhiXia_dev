"""英语考试辅导助手技能卡

提供专业的英语考试辅导功能：
- 考试规划：制定个性化备考计划
- 听力训练：提供听力材料和练习建议
- 长难句解析：分析复杂句子结构
- 词汇记忆：词汇学习策略和记忆方法
- 作文辅导：写作指导和范文分析

解耦设计：
- 技能卡通过 KnowledgeProvider 接口与知识卡交互
- 支持动态发现和绑定多种知识卡
- 无知识卡时提供降级功能
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

from zhixia.agent.tool import Tool, ToolRegistry
from zhixia.core.card_base import CardManifest, HostContext, SkillCard, KnowledgeProvider

if TYPE_CHECKING:
    from zhixia.core.card_base import ContentAwareKnowledgeCard

# ============================================================
# 自包含导入关键代码（所有 Skill 卡必须包含）
# ============================================================
_CARD_ROOT = Path(__file__).parent.resolve()
if str(_CARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_CARD_ROOT))

from tools.exam_planning_tool import ExamPlanningTool
from tools.listening_tool import ListeningTool
from tools.sentence_analysis_tool import SentenceAnalysisTool
from tools.vocabulary_tool import VocabularyTool
from tools.writing_tool import WritingTool
# 新增核心工具
from tools.exam_planner import ExamPlannerTool
from tools.listening_assistant import ListeningAssistantTool
from tools.long_sentence import LongSentenceTool
from tools.vocabulary_reviewer import VocabularyReviewerTool
from tools.writing_assistant import WritingAssistantTool
# ============================================================


class EnglishTutorSkill(SkillCard):
    """英语考试辅导助手技能卡。

    提供专业的英语考试辅导功能：
    - 考试准备计划器：智能安排备考计划，分析薄弱点
    - 听力辅助器：听力材料获取、播放、理解测试
    - 长难句助力器：长难句解析、语法讲解
    - 词汇复习器：滚动复习、定期检测
    - 作文辅导器：范文、思路、润色

    解耦特性：
    - 动态发现和绑定知识卡
    - 通过 KnowledgeProvider 接口获取内容
    - 支持多种知识卡灵活组合

    工具生命周期：
    - 创建：在 on_mount() 插卡时
    - 注册：在 on_mount() 插卡时
    - 注销：在 on_unmount() 拔卡时
    """

    # 需要的内容类型
    REQUIRED_CONTENT_TYPES = ["listening", "sentences", "writing", "vocabulary"]

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        self._tools_created = False
        self._knowledge_provider: Optional[KnowledgeProvider] = None

    def on_mount(self, host: HostContext) -> None:
        """插卡时：发现知识卡 -> 创建工具 -> 注册工具 -> 加载人设。"""
        if self._tools_created:
            return  # 防止重复注册

        llm_engine = host.llm_engine

        # 步骤1: 动态发现和绑定知识卡
        self._knowledge_provider = self._discover_knowledge_provider(host)
        if self._knowledge_provider:
            print(f"   [解耦] 已绑定知识卡，内容类型: {', '.join(getattr(self._knowledge_provider, 'content_types', []))}")
        else:
            print(f"   [解耦] 警告: 未找到合适的知识卡，工具将以降级模式运行")

        # 步骤2: 创建基础工具实例并注入LLM引擎（插卡时创建）
        exam_planning_tool = ExamPlanningTool(llm_engine=llm_engine)
        listening_tool = ListeningTool(llm_engine=llm_engine)
        sentence_analysis_tool = SentenceAnalysisTool(llm_engine=llm_engine)
        vocabulary_tool = VocabularyTool(llm_engine=llm_engine)
        writing_tool = WritingTool(llm_engine=llm_engine)

        # 步骤3: 创建核心工具实例，注入知识提供者
        exam_planner_tool = ExamPlannerTool(
            llm_engine=llm_engine,
            knowledge_provider=self._knowledge_provider
        )
        listening_assistant_tool = ListeningAssistantTool(
            llm_engine=llm_engine,
            knowledge_provider=self._knowledge_provider
        )
        long_sentence_tool = LongSentenceTool(
            llm_engine=llm_engine,
            knowledge_provider=self._knowledge_provider
        )
        vocabulary_reviewer_tool = VocabularyReviewerTool(
            llm_engine=llm_engine,
            knowledge_provider=self._knowledge_provider
        )
        writing_assistant_tool = WritingAssistantTool(
            llm_engine=llm_engine,
            knowledge_provider=self._knowledge_provider
        )

        # 步骤4: 注册基础工具到主机（插卡时注册）
        host.tool_registry.register(exam_planning_tool)
        host.tool_registry.register(listening_tool)
        host.tool_registry.register(sentence_analysis_tool)
        host.tool_registry.register(vocabulary_tool)
        host.tool_registry.register(writing_tool)

        # 步骤5: 注册核心工具到主机
        host.tool_registry.register(exam_planner_tool)
        host.tool_registry.register(listening_assistant_tool)
        host.tool_registry.register(long_sentence_tool)
        host.tool_registry.register(vocabulary_reviewer_tool)
        host.tool_registry.register(writing_assistant_tool)
        
        self.registered_tool_names = [
            # 基础工具
            "exam_planning",
            "listening_training",
            "sentence_analysis",
            "vocabulary_learning",
            "writing_coaching",
            # 核心工具
            "exam_planner",
            "listening_assistant",
            "long_sentence",
            "vocabulary_reviewer",
            "writing_assistant",
        ]

        self._tools_created = True

        # 步骤6: 加载人设
        persona = self._load_persona()
        if persona:
            host.persona_holder.set_overlay(persona, self.name)

        print(f"[MOUNT] 英语考试辅导助手技能卡已插入: {self.display_name}")
        print(f"   基础工具: exam_planning, listening_training, sentence_analysis, vocabulary_learning, writing_coaching")
        print(f"   核心工具: exam_planner, listening_assistant, long_sentence, vocabulary_reviewer, writing_assistant")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销工具 + 恢复人设。"""
        for tool_name in list(self.registered_tool_names):
            host.tool_registry.unregister(tool_name)
        self.registered_tool_names = []

        host.persona_holder.clear_overlay(self.name)

        self._tools_created = False
        self._knowledge_provider = None

        print(f"[UNMOUNT] 英语考试辅导助手技能卡已拔出: {self.display_name}")

    def _discover_knowledge_provider(self, host: HostContext) -> Optional[KnowledgeProvider]:
        """动态发现和绑定知识卡。

        通过 HostContext 查询已注册的知识卡，找到匹配的内容提供者。

        Args:
            host: 主机上下文

        Returns:
            匹配的知识提供者，如果没有则返回 None
        """
        # 从 knowledge_hub 获取已注册的知识卡
        # 注意：这里假设 knowledge_hub 有办法获取知识卡实例
        # 实际实现可能需要通过其他方式，如 host.config 或全局注册表

        # 方案1: 如果 knowledge_hub 提供了获取知识卡的方法
        if hasattr(host.knowledge_hub, '_retrievers'):
            for name, retriever in host.knowledge_hub._retrievers.items():
                # 检查是否是 ContentAwareKnowledgeCard
                if hasattr(retriever, 'content_types') and hasattr(retriever, 'supported_exams'):
                    # 检查是否提供所需的内容类型
                    provided_types = set(retriever.content_types)
                    required_types = set(self.REQUIRED_CONTENT_TYPES)
                    if required_types.intersection(provided_types):
                        return retriever

        # 方案2: 通过配置或其他方式获取
        # 这里可以根据实际架构调整

        return None

    def get_tools(self) -> ToolRegistry:
        """获取工具列表预览（仅用于元数据展示，不创建实际工具实例）。"""
        registry = ToolRegistry()
        return registry

    def get_persona(self) -> str:
        return self._load_persona() or ""

    def _load_persona(self) -> str:
        persona_path = self.card_root / "persona.json"
        if not persona_path.exists():
            return ""
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("persona", "")
        except Exception:
            return ""
