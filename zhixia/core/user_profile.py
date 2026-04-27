"""用户画像管理模块

每个技能卡维护一个 zhixia.md 文件，用于记录用户特征和偏好。
系统在交互时加载此文件，提供个性化服务；交互后自动更新。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# zhixia.md 的默认模板
DEFAULT_PROFILE_TEMPLATE = """# 用户画像

> 此文件由知匣自动维护，用于记录用户特征和偏好，提供个性化服务。

## 基本信息
- 称呼: 
- 身份: 
- 偏好: 

## 对话历史摘要


## 用户特质


## 注意事项

"""


@dataclass
class UserProfile:
    """用户画像数据类。

    属性:
        card_root: 技能卡根目录
        file_path: zhixia.md 文件路径
        raw_content: 原始文件内容
        sections: 解析后的章节内容 {section_name: content}
        basic_info: 基本信息字典
        conversation_summary: 对话历史摘要列表
        user_traits: 用户特质列表
        notes: 注意事项列表
    """

    card_root: Path
    file_path: Path = field(init=False)
    raw_content: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    basic_info: Dict[str, str] = field(default_factory=dict)
    conversation_summary: List[str] = field(default_factory=list)
    user_traits: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.file_path = self.card_root / "zhixia.md"
        self._ensure_file()
        self.load()

    def _ensure_file(self) -> None:
        """确保 zhixia.md 文件存在。"""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(DEFAULT_PROFILE_TEMPLATE, encoding="utf-8")
            logger.info("创建用户画像文件: %s", self.file_path)

    def load(self) -> None:
        """从 zhixia.md 加载用户画像。"""
        try:
            self.raw_content = self.file_path.read_text(encoding="utf-8")
            self._parse()
            logger.debug("用户画像已加载: %s", self.file_path)
        except Exception as exc:
            logger.warning("加载用户画像失败: %s", exc)
            self.raw_content = DEFAULT_PROFILE_TEMPLATE
            self._parse()

    def _parse(self) -> None:
        """解析 zhixia.md 内容。"""
        # 按章节分割
        section_pattern = r'^## (.+)$'
        current_section = None
        current_content = []

        for line in self.raw_content.split('\n'):
            match = re.match(section_pattern, line)
            if match:
                if current_section:
                    self.sections[current_section] = '\n'.join(current_content).strip()
                current_section = match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            self.sections[current_section] = '\n'.join(current_content).strip()

        # 解析基本信息
        self._parse_basic_info()

        # 解析对话历史摘要
        self.conversation_summary = self._parse_list_section("对话历史摘要")

        # 解析用户特质
        self.user_traits = self._parse_list_section("用户特质")

        # 解析注意事项
        self.notes = self._parse_list_section("注意事项")

    def _parse_basic_info(self) -> None:
        """解析基本信息章节。"""
        basic_section = self.sections.get("基本信息", "")
        self.basic_info = {}
        for line in basic_section.split('\n'):
            line = line.strip().lstrip('- ')
            if ':' in line or '：' in line:
                key, _, value = line.partition(':')
                if not value:
                    key, _, value = line.partition('：')
                self.basic_info[key.strip()] = value.strip()

    def _parse_list_section(self, section_name: str) -> List[str]:
        """解析列表类型的章节。"""
        content = self.sections.get(section_name, "")
        items = []
        for line in content.split('\n'):
            line = line.strip().lstrip('- ')
            if line and not line.startswith('#'):
                items.append(line)
        return items

    def save(self) -> None:
        """将用户画像保存到 zhixia.md。"""
        self._ensure_file()
        try:
            content = self._build_content()
            self.file_path.write_text(content, encoding="utf-8")
            logger.debug("用户画像已保存: %s", self.file_path)
        except Exception as exc:
            logger.error("保存用户画像失败: %s", exc)

    def _build_content(self) -> str:
        """构建 zhixia.md 文件内容。"""
        lines = [
            "# 用户画像",
            "",
            "> 此文件由知匣自动维护，用于记录用户特征和偏好，提供个性化服务。",
            "",
            "## 基本信息",
        ]

        # 基本信息
        for key in ["称呼", "身份", "偏好"]:
            value = self.basic_info.get(key, "")
            lines.append(f"- {key}: {value}")

        lines.extend(["", "## 对话历史摘要", ""])
        for item in self.conversation_summary:
            lines.append(f"- {item}")

        lines.extend(["", "## 用户特质", ""])
        for item in self.user_traits:
            lines.append(f"- {item}")

        lines.extend(["", "## 注意事项", ""])
        for item in self.notes:
            lines.append(f"- {item}")

        lines.append("")
        return '\n'.join(lines)

    def to_prompt_text(self) -> str:
        """生成用于注入到 system prompt 的文本。"""
        parts = []

        if self.basic_info:
            filled_info = {k: v for k, v in self.basic_info.items() if v}
            if filled_info:
                parts.append("【用户基本信息】")
                for k, v in filled_info.items():
                    parts.append(f"- {k}: {v}")

        if self.user_traits:
            parts.append("【用户特质】")
            for trait in self.user_traits:
                parts.append(f"- {trait}")

        if self.notes:
            parts.append("【注意事项】")
            for note in self.notes:
                parts.append(f"- {note}")

        if self.conversation_summary:
            parts.append("【最近对话摘要】")
            for summary in self.conversation_summary[-3:]:  # 最近3条
                parts.append(f"- {summary}")

        if parts:
            return "\n".join(parts)
        return ""

    def update_basic_info(self, **kwargs: str) -> None:
        """更新基本信息。"""
        self.basic_info.update(kwargs)
        self.save()

    def add_trait(self, trait: str) -> None:
        """添加用户特质（去重）。"""
        if trait and trait not in self.user_traits:
            self.user_traits.append(trait)
            self.save()

    def add_note(self, note: str) -> None:
        """添加注意事项（去重）。"""
        if note and note not in self.notes:
            self.notes.append(note)
            self.save()

    def add_conversation_summary(self, summary: str) -> None:
        """添加对话摘要（保留最近10条）。"""
        if summary:
            self.conversation_summary.append(summary)
            # 保留最近10条
            if len(self.conversation_summary) > 10:
                self.conversation_summary = self.conversation_summary[-10:]
            self.save()
