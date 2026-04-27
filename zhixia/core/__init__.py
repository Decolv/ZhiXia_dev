"""ZhiXia Core — 主机核心骨架

主机只保留基础框架，所有业务逻辑通过插卡载入：
- Skill 卡: 提供 Tools + Agent 人设
- Knowledge 卡: 提供 RAG 知识库 + 多媒体资源

核心组件:
    CardBase          — 卡片基类接口
    SkillCard         — 技能卡接口（Tools + Persona）
    KnowledgeCard     — 知识卡接口（RAG + Assets）
    CardManifest      — 卡片元数据
    CardLoader        — 热插拔加载器 + 痕迹清除
    HostContext       — 主机上下文（卡片与主机的交互桥梁）
    HostOrchestrator  — 主机编排器（动态组装 Agent）

插卡生命周期:
    1. 用户将卡放入槽位（文件系统目录）
    2. CardLoader.scan() 检测到新卡
    3. 调用 card.on_mount(host_context) → 注册工具/人设/知识
    4. HostOrchestrator 使用新配置运行
    5. 用户拔卡
    6. 调用 card.on_unmount(host_context) → 清除所有痕迹
    7. CardLoader 从 sys.modules 卸载卡片模块
"""

from zhixia.core.card_base import (
    CardBase,
    CardManifest,
    HostContext,
    KnowledgeCard,
    SkillCard,
)
from zhixia.core.card_loader import CardLoader, SlotWatcher
from zhixia.core.host_orchestrator import HostOrchestrator

__all__ = [
    "CardBase",
    "CardManifest",
    "SkillCard",
    "KnowledgeCard",
    "HostContext",
    "CardLoader",
    "SlotWatcher",
    "HostOrchestrator",
]
