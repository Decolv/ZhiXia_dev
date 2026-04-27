"""卡片基类接口 —— Skill 卡与 Knowledge 卡的契约定义

每张卡是一个独立的 Python 包，必须提供：
1. manifest.json — 卡片元数据（名称、版本、类型、作者）
2. card.py — 包含 Card 子类实现
3. （可选）其他资源文件

卡片加载器通过 importlib 动态导入 card.py，实例化 Card 类。
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from zhixia.agent.tool import ToolRegistry
from zhixia.core.user_profile import UserProfile
from zhixia.display.base import DisplayOutput
from zhixia.llm.rag.base import RAGContext, RAGRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentConfigurator — Agent 配置接口
# ---------------------------------------------------------------------------

class AgentConfigurator:
    """Agent 配置器接口。

    卡片可通过此接口定义 Agent 类型和行为，实现不同卡片使用不同 Agent 策略。
    例如：
    - 新生助手卡 → ToolCallingAgent
    - 复杂推理卡 → ReActAgent
    - 简单问答卡 → 直接 LLM
    
    线程安全：所有读写操作使用锁保护。
    """

    def __init__(self) -> None:
        self._agent_type: str = "react"  # 默认 ReAct
        self._max_iterations: int = 5
        self._early_stopping_method: str = "raise"
        self._enabled_tools: Optional[List[str]] = None  # None=全部启用
        self._custom_system_prompt: Optional[str] = None
        self._lock = threading.Lock()

    def set_agent_type(self, agent_type: str) -> None:
        """设置 Agent 类型: 'react', 'tool_calling', 'direct_llm'。"""
        with self._lock:
            self._agent_type = agent_type

    def set_max_iterations(self, max_iter: int) -> None:
        """设置最大工具调用迭代次数。"""
        with self._lock:
            self._max_iterations = max_iter

    def set_early_stopping_method(self, method: str) -> None:
        """设置提前停止方法: 'force', 'raise'。"""
        with self._lock:
            self._early_stopping_method = method

    def set_enabled_tools(self, tool_names: Optional[List[str]]) -> None:
        """设置启用的工具列表，None 表示启用所有注册的工具。"""
        with self._lock:
            self._enabled_tools = tool_names

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """设置自定义 system prompt（None 则使用 persona_holder）。"""
        with self._lock:
            self._custom_system_prompt = prompt

    def get_config(self) -> Dict[str, Any]:
        """获取 Agent 配置。"""
        with self._lock:
            return {
                "agent_type": self._agent_type,
                "max_iterations": self._max_iterations,
                "early_stopping_method": self._early_stopping_method,
                "enabled_tools": self._enabled_tools,
                "custom_system_prompt": self._custom_system_prompt,
            }

    def clear(self) -> None:
        """重置为默认配置。"""
        with self._lock:
            self._agent_type = "react"
            self._max_iterations = 5
            self._early_stopping_method = "raise"
            self._enabled_tools = None
            self._custom_system_prompt = None


# ---------------------------------------------------------------------------
# ResponsePostProcessor — 响应后处理器接口
# ---------------------------------------------------------------------------

class ResponsePostProcessor(ABC):
    """响应后处理器接口。

    卡片可注册此处理器，在 AI 生成响应后执行自定义逻辑。
    例如：
    - 导航卡 → 解析 __NAV_DATA__ 并展示导航界面
    - 音乐卡 → 解析音乐指令并播放
    - 提醒卡 → 设置定时提醒

    主机不关心具体处理逻辑，仅负责调用。
    """

    @abstractmethod
    def process(self, response_text: str) -> Tuple[str, bool]:
        """处理响应。

        Args:
            response_text: 原始响应文本

        Returns:
            (cleaned_text, is_handled)
            - cleaned_text: 清理后的响应文本
            - is_handled: 是否已处理（如果已处理，主机可能跳过默认显示）
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """处理器名称，用于日志和注销。"""
        ...


# ---------------------------------------------------------------------------
# CardManifest — 卡片元数据
# ---------------------------------------------------------------------------

@dataclass
class CardManifest:
    """卡片元数据，对应卡片根目录的 manifest.json。

    示例 manifest.json:
        {
            "name": "hnu_freshman",
            "display_name": "湖南大学新生助手",
            "version": "1.0.0",
            "type": "skill",
            "author": "ZhiXia Team",
            "description": "为湖南大学新生提供校园导航、专业查询、生活指导",
            "entrypoint": "card.py",
            "dependencies": [],
            "min_host_version": "0.2.0"
        }
    """

    name: str
    display_name: str
    version: str
    type: str  # "skill" | "knowledge"
    author: str = ""
    description: str = ""
    entrypoint: str = "card.py"  # 相对卡片根目录的入口文件
    dependencies: List[str] = field(default_factory=list)
    min_host_version: str = "0.1.0"

    @classmethod
    def load(cls, card_root: Path) -> Optional[CardManifest]:
        """从卡片根目录加载 manifest.json。"""
        manifest_path = card_root / "manifest.json"
        if not manifest_path.exists():
            logger.warning("卡片 manifest 不存在: %s", manifest_path)
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as exc:
            logger.error("解析 manifest 失败: %s", exc)
            return None


# ---------------------------------------------------------------------------
# HostContext — 主机上下文
# ---------------------------------------------------------------------------

@dataclass
class HostContext:
    """卡片与主机交互的上下文对象。

    插卡时，HostOrchestrator 构造此对象传给 card.on_mount()，
    卡片通过此对象注册工具、人设、知识等。
    
    线程安全：
    - response_processors 列表使用锁保护
    """

    # 工具注册表（Skill 卡在此注册工具）
    tool_registry: ToolRegistry

    # 人设管理器（Skill 卡可覆盖 system prompt）
    persona_holder: "PersonaHolder"

    # 知识检索器（Knowledge 卡可注册/扩展 RAG）
    knowledge_hub: "KnowledgeHub"

    # 用户画像（技能卡维护的用户特征和偏好）
    user_profile: Optional[UserProfile] = None

    # 显示输出（卡片可推送自定义显示内容）
    display: Optional[DisplayOutput] = None

    # Agent 配置器（Skill 卡可自定义 Agent 类型和行为）
    agent_configurator: AgentConfigurator = field(default_factory=AgentConfigurator)

    # 响应后处理器列表（卡片可注册自定义响应处理逻辑）
    response_processors: List["ResponsePostProcessor"] = field(default_factory=list)

    # 主机配置
    config: Optional[Any] = None

    # LLM引擎（工具需要调用LLM智能生成答案）
    llm_engine: Optional[Any] = None

    # 卡片根目录（卡片可读取自己的资源文件）
    card_root: Optional[Path] = None

    def __post_init__(self) -> None:
        self._processors_lock = threading.RLock()

    def register_response_processor(self, processor: "ResponsePostProcessor") -> None:
        """注册响应后处理器。"""
        with self._processors_lock:
            self.response_processors.append(processor)
        logger.info("响应后处理器已注册: %s", processor.name)

    def unregister_response_processor(self, processor_name: str) -> None:
        """注销响应后处理器。"""
        with self._processors_lock:
            self.response_processors = [
                p for p in self.response_processors if p.name != processor_name
            ]
        logger.info("响应后处理器已注销: %s", processor_name)

    def __repr__(self) -> str:
        return (
            f"HostContext(tools={len(self.tool_registry.list_tools())}, "
            f"persona={self.persona_holder.current_persona[:20]}...)"
        )


# ---------------------------------------------------------------------------
# PersonaHolder — 人设管理器
# ---------------------------------------------------------------------------

class PersonaHolder:
    """管理人设（system prompt）的叠加与恢复。

    支持多层人设叠加：
        基础人设（主机默认）
        + Skill 卡人设1（插卡时叠加）
        + Skill 卡人设2（插卡时叠加）
        → 最终 system prompt

    拔卡后自动恢复到基础人设。
    """

    def __init__(self, base_persona: str) -> None:
        self.base_persona = base_persona
        self._overlays: List[Tuple[str, str]] = []

    @property
    def current_persona(self) -> str:
        if not self._overlays:
            return self.base_persona
        overlay_parts = [persona for _, persona in self._overlays]
        return "\n\n".join([self.base_persona] + overlay_parts)

    def set_overlay(self, persona: str, card_name: str) -> None:
        """Skill 卡挂载时调用：追加人设叠加。
        
        如果同一卡片名已存在，则替换其人设。
        """
        existing_idx = None
        for i, (name, _) in enumerate(self._overlays):
            if name == card_name:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self._overlays[existing_idx] = (card_name, persona)
            logger.info("人设已更新 [%s]: %s...", card_name, persona[:50])
        else:
            self._overlays.append((card_name, persona))
            logger.info("人设已追加 [%s]: %s...", card_name, persona[:50])

    def clear_overlay(self, card_name: Optional[str] = None) -> None:
        """Skill 卡卸载时调用：移除指定卡片的人设叠加。
        
        如果不指定 card_name，则清除所有叠加人设。
        """
        if card_name is None:
            self._overlays.clear()
            logger.info("所有人设叠加已清除")
        else:
            self._overlays = [
                (name, persona) for name, persona in self._overlays if name != card_name
            ]
            logger.info("人设叠加已移除 [%s]", card_name)


# ---------------------------------------------------------------------------
# KnowledgeHub — 知识管理中心
# ---------------------------------------------------------------------------

class KnowledgeHub:
    """管理知识卡提供的 RAG 检索器和多媒体资源。

    支持多张知识卡共存（知识并集），也支持替换。
    线程安全：所有公共方法使用 RLock 保护。
    """

    def __init__(self) -> None:
        self._retrievers: Dict[str, RAGRetriever] = {}
        self._assets: Dict[str, Dict[str, Path]] = {}
        self._lock = threading.RLock()

    def register_retriever(self, name: str, retriever: RAGRetriever) -> None:
        with self._lock:
            self._retrievers[name] = retriever
        logger.info("知识检索器已注册: %s", name)

    def unregister_retriever(self, name: str) -> None:
        with self._lock:
            if name in self._retrievers:
                del self._retrievers[name]
        logger.info("知识检索器已注销: %s", name)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """从所有已注册的知识检索器中查询。

        单个检索器失败不影响其他检索器的结果。
        """
        results = []
        with self._lock:
            retrievers_snapshot = dict(self._retrievers)

        for name, retriever in retrievers_snapshot.items():
            try:
                context = retriever.retrieve(query, top_k)
                if isinstance(context, RAGContext) and getattr(context, "chunks", None):
                    results.extend(context.chunks)
            except Exception as exc:
                logger.warning("知识检索失败 [%s]: %s", name, exc)
        return results

    def register_assets(self, name: str, assets: Dict[str, Path]) -> None:
        with self._lock:
            self._assets[name] = assets
        logger.info("资源已注册: %s (%d 个)", name, len(assets))

    def unregister_assets(self, name: str) -> None:
        with self._lock:
            if name in self._assets:
                del self._assets[name]
        logger.info("资源已注销: %s", name)

    def get_asset(self, name: str) -> Optional[Path]:
        """按名称查找资源（跨所有知识卡）。"""
        with self._lock:
            for card_assets in self._assets.values():
                if name in card_assets:
                    return card_assets[name]
        return None

    def clear_all(self) -> None:
        with self._lock:
            self._retrievers.clear()
            self._assets.clear()
        logger.info("KnowledgeHub 已清空")


# ---------------------------------------------------------------------------
# CardBase — 所有卡片的抽象基类
# ---------------------------------------------------------------------------

class CardBase(ABC):
    """所有卡片的抽象基类。

    子类必须实现：
        - on_mount(host_context): 插卡时注册到主机
        - on_unmount(host_context): 拔卡时清理痕迹

    子类可选择性覆盖：
        - get_manifest(): 返回 CardManifest
    """

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        self.manifest = manifest
        self.card_root = card_root
        self.registered_tool_names: List[str] = []

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def display_name(self) -> str:
        return self.manifest.display_name

    @abstractmethod
    def on_mount(self, host: HostContext) -> None:
        """插卡时调用。卡片在此注册自己的工具/人设/知识。"""
        ...

    @abstractmethod
    def on_unmount(self, host: HostContext) -> None:
        """拔卡时调用。卡片必须在此清理所有痕迹。"""
        ...

    def get_resource(self, relative_path: str) -> Path:
        """获取卡片内的资源文件路径。

        安全检查：防止路径穿越攻击。
        """
        target = (self.card_root / relative_path).resolve()
        root = self.card_root.resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"路径越界: {relative_path}")
        return target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.manifest.version})"


# ---------------------------------------------------------------------------
# SkillCard — 技能卡接口
# ---------------------------------------------------------------------------

class SkillCard(CardBase, ABC):
    """技能卡：提供 Tools + Agent 人设。

    示例实现:
        class HNUFreshmanSkill(SkillCard):
            def on_mount(self, host):
                host.tool_registry.register(CampusMapTool())
                host.persona_holder.set_overlay(self._load_persona(), self.name)
                host.agent_configurator.set_agent_type("tool_calling")

            def on_unmount(self, host):
                host.tool_registry.unregister("campus_map")
                host.persona_holder.clear_overlay(self.name)
    """

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """返回本卡提供的所有工具列表。"""
        ...

    @abstractmethod
    def get_persona(self) -> str:
        """返回本卡的 Agent 人设（system prompt）。"""
        ...


# ---------------------------------------------------------------------------
# KnowledgeCard — 知识卡接口
# ---------------------------------------------------------------------------

class KnowledgeCard(CardBase, ABC):
    """知识卡：提供 RAG 知识库 + 多媒体资源。

    示例实现:
        class HNUCampusKnowledge(KnowledgeCard):
            def on_mount(self, host):
                retriever = self._build_vector_store()
                host.knowledge_hub.register_retriever(self.name, retriever)
                host.knowledge_hub.register_assets(self.name, {
                    "campus_map": self.get_resource("maps/campus.png"),
                })

            def on_unmount(self, host):
                host.knowledge_hub.unregister_retriever(self.name)
                host.knowledge_hub.unregister_assets(self.name)
    """

    @abstractmethod
    def get_retriever(self) -> RAGRetriever:
        """返回本卡的知识检索器。"""
        ...

    @abstractmethod
    def get_assets(self) -> Dict[str, Path]:
        """返回资源映射 {asset_name: file_path}。"""
        ...

    def get_documents(self) -> List[Path]:
        """返回知识文档路径列表（默认扫描 docs/ 目录）。"""
        docs_dir = self.card_root / "docs"
        if not docs_dir.exists() or not docs_dir.is_dir():
            return []
        return sorted([p for p in docs_dir.iterdir() if p.is_file()])
