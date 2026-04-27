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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from zhixia.agent.tool import ToolRegistry
from zhixia.display.base import DisplayOutput
from zhixia.llm.rag.base import RAGRetriever

logger = logging.getLogger(__name__)


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
    """

    # 工具注册表（Skill 卡在此注册工具）
    tool_registry: ToolRegistry

    # 人设管理器（Skill 卡可覆盖 system prompt）
    persona_holder: "PersonaHolder"

    # 知识检索器（Knowledge 卡可注册/扩展 RAG）
    knowledge_hub: "KnowledgeHub"

    # 显示输出（卡片可推送自定义显示内容）
    display: Optional[DisplayOutput] = None

    # 主机配置
    config: Optional[Any] = None

    # 卡片根目录（卡片可读取自己的资源文件）
    card_root: Optional[Path] = None

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
        + Skill 卡人设（插卡时叠加）
        → 最终 system prompt

    拔卡后自动恢复到基础人设。
    """

    def __init__(self, base_persona: str) -> None:
        self.base_persona = base_persona
        self._overlay: Optional[str] = None
        self._card_name: Optional[str] = None

    @property
    def current_persona(self) -> str:
        if self._overlay:
            return self._overlay
        return self.base_persona

    def set_overlay(self, persona: str, card_name: str) -> None:
        """Skill 卡挂载时调用：覆盖人设。"""
        self._overlay = persona
        self._card_name = card_name
        logger.info("人设已叠加 [%s]: %s...", card_name, persona[:50])

    def clear_overlay(self) -> None:
        """Skill 卡卸载时调用：恢复基础人设。"""
        if self._card_name:
            logger.info("人设已恢复（移除 %s 的叠加）", self._card_name)
        self._overlay = None
        self._card_name = None


# ---------------------------------------------------------------------------
# KnowledgeHub — 知识管理中心
# ---------------------------------------------------------------------------

class KnowledgeHub:
    """管理知识卡提供的 RAG 检索器和多媒体资源。

    支持多张知识卡共存（知识并集），也支持替换。
    """

    def __init__(self) -> None:
        self._retrievers: Dict[str, RAGRetriever] = {}
        self._assets: Dict[str, Dict[str, Path]] = {}

    def register_retriever(self, name: str, retriever: RAGRetriever) -> None:
        self._retrievers[name] = retriever
        logger.info("知识检索器已注册: %s", name)

    def unregister_retriever(self, name: str) -> None:
        if name in self._retrievers:
            del self._retrievers[name]
            logger.info("知识检索器已注销: %s", name)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """从所有已注册的知识检索器中查询。"""
        results = []
        for name, retriever in self._retrievers.items():
            try:
                context = retriever.retrieve(query, top_k)
                if context and context.chunks:
                    results.extend(context.chunks)
            except Exception as exc:
                logger.warning("知识检索失败 [%s]: %s", name, exc)
        return results

    def register_assets(self, name: str, assets: Dict[str, Path]) -> None:
        self._assets[name] = assets
        logger.info("资源已注册: %s (%d 个)", name, len(assets))

    def unregister_assets(self, name: str) -> None:
        if name in self._assets:
            del self._assets[name]
            logger.info("资源已注销: %s", name)

    def get_asset(self, name: str) -> Optional[Path]:
        """按名称查找资源（跨所有知识卡）。"""
        for card_assets in self._assets.values():
            if name in card_assets:
                return card_assets[name]
        return None

    def clear_all(self) -> None:
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
        """获取卡片内的资源文件路径。"""
        return self.card_root / relative_path

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

            def on_unmount(self, host):
                host.tool_registry.unregister("campus_map")
                host.persona_holder.clear_overlay()
    """

    @abstractmethod
    def get_tools(self) -> ToolRegistry:
        """返回本卡提供的所有工具。"""
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
        if not docs_dir.exists():
            return []
        return sorted([p for p in docs_dir.iterdir() if p.is_file()])
