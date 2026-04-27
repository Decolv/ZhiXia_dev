# ZhiXia 卡片开发规范

> 本文档定义 Skill 卡与 Knowledge 卡的开发标准，确保卡片与主机深度解耦、即插即用。

---

## 核心原则

### 1. 自包含原则
**卡片必须是自包含的。** 将卡片目录复制到任意机器的 `cards/slot_a/`（Skill）或 `cards/slot_b/`（Knowledge）后，无需修改任何代码即可运行。

### 2. 禁止硬编码项目包路径
**卡片内部模块间的导入，不得使用项目级包路径前缀。**

❌ 错误示例（深度耦合）：
```python
# 即使卡片被复制到 cards/slot_a/，这些导入仍会去找 skills/ 目录
from skills.my_skill.tools.my_tool import MyTool
from skills.my_skill.nav_processor import NavProcessor
```

✅ 正确示例（自包含）：
```python
import sys
from pathlib import Path

# 将卡片根目录加入 Python 路径，使导入基于卡片自身目录
_CARD_ROOT = Path(__file__).parent.resolve()
if str(_CARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_CARD_ROOT))

from tools.my_tool import MyTool
from nav_processor import NavProcessor
```

### 3. 允许依赖主机公共接口
卡片**可以且应当**依赖 `zhixia` 包提供的公共接口：

```python
from zhixia.agent.tool import Tool, ToolRegistry
from zhixia.core.card_base import SkillCard, HostContext, CardManifest
from zhixia.llm.rag.base import RAGRetriever
```

这些接口是主机与卡片之间的**契约**，版本兼容性通过 `manifest.json` 中的 `min_host_version` 声明。

---

## 目录结构规范

### Skill 卡结构

```
my_skill/                    # 卡片根目录（复制到 cards/slot_a/ 下）
├── manifest.json            # 卡片元数据（必填）
├── card.py                  # 卡片入口类（必填）
├── persona.json             # 人设配置（可选）
├── tools/                   # 工具目录（可选）
│   ├── __init__.py          # 保持为空或工具聚合
│   └── my_tool.py           # 具体工具实现
├── assets/                  # 资源文件（可选）
│   └── images/
└── README.md                # 卡片说明（可选）
```

### Knowledge 卡结构

```
my_knowledge/                # 卡片根目录（复制到 cards/slot_b/ 下）
├── manifest.json            # 卡片元数据（必填）
├── card.py                  # 卡片入口类（必填）
├── docs/                    # 知识文档（必填）
│   ├── topic_a.md
│   └── topic_b.md
├── maps/                    # 地图等资源（可选）
└── README.md                # 卡片说明（可选）
```

---

## manifest.json 规范

```json
{
    "name": "my_skill",
    "display_name": "我的技能",
    "version": "1.0.0",
    "type": "skill",
    "author": "Your Name",
    "description": "简短描述卡片功能",
    "entrypoint": "card.py",
    "dependencies": [],
    "min_host_version": "0.1.0"
}
```

| 字段 | 说明 | 必填 |
|------|------|------|
| `name` | 英文标识符（唯一） | 是 |
| `display_name` | 显示名称 | 是 |
| `version` | 语义化版本 | 是 |
| `type` | `"skill"` 或 `"knowledge"` | 是 |
| `author` | 作者 | 否 |
| `description` | 功能描述 | 否 |
| `entrypoint` | 入口文件名，默认 `card.py` | 否 |
| `dependencies` | 其他卡片依赖 | 否 |
| `min_host_version` | 最低主机版本要求 | 否 |

---

## Skill 卡开发指南

### 最小示例

```python
"""示例 Skill 卡 —— 天气查询"""

import json
import sys
from pathlib import Path
from typing import Optional

from zhixia.agent.tool import Tool, ToolRegistry
from zhixia.core.card_base import CardManifest, HostContext, SkillCard

# 自包含导入：将卡片根目录加入 sys.path
_CARD_ROOT = Path(__file__).parent.resolve()
if str(_CARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_CARD_ROOT))

from tools.weather_tool import WeatherTool


class WeatherSkill(SkillCard):
    """天气查询技能卡。"""

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        self._tools_created = False

    def on_mount(self, host: HostContext) -> None:
        if self._tools_created:
            return

        llm_engine = host.llm_engine

        # 创建并注册工具
        weather_tool = WeatherTool(llm_engine=llm_engine)
        host.tool_registry.register(weather_tool)
        self.registered_tool_names = ["weather_query"]

        self._tools_created = True

        # 加载人设
        persona = self._load_persona()
        if persona:
            host.persona_holder.set_overlay(persona, self.name)

        print(f"[MOUNT] Skill 卡已插入: {self.display_name}")

    def on_unmount(self, host: HostContext) -> None:
        for tool_name in list(self.registered_tool_names):
            host.tool_registry.unregister(tool_name)
        self.registered_tool_names = []
        host.persona_holder.clear_overlay(self.name)
        self._tools_created = False
        print(f"[UNMOUNT] Skill 卡已拔出: {self.display_name}")

    def get_tools(self) -> ToolRegistry:
        return ToolRegistry()

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
```

### 工具开发规范

```python
"""天气查询工具"""

from typing import Optional
from zhixia.agent.tool import Tool


class WeatherTool(Tool):
    """查询指定城市天气。"""

    def __init__(self, llm_engine=None):
        super().__init__(
            name="weather_query",
            description="查询指定城市的天气情况，如：今天北京天气怎么样？",
            func=self._query,
        )
        self._llm_engine = llm_engine

    def _query(self, city: str) -> str:
        # 实际实现可调用天气 API 或使用 LLM 生成
        if self._llm_engine:
            # 使用 LLM 生成模拟回答
            from zhixia.llm.base import LLMMessage
            messages = [
                LLMMessage(role="system", content="你是天气助手，简洁回答。"),
                LLMMessage(role="user", content=f"{city}今天天气怎么样？"),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=64)
        return f"{city}今天晴，25°C，适合出行。"
```

---

## Knowledge 卡开发指南

### 最小示例

```python
"""示例 Knowledge 卡 —— 公司知识库"""

from pathlib import Path
from typing import Dict, List, Optional

from zhixia.core.card_base import CardManifest, HostContext, KnowledgeCard
from zhixia.llm.rag.base import RAGContext, RAGRetriever


class SimpleKeywordRetriever(RAGRetriever):
    """关键词检索器（简单示例）。"""

    def __init__(self, documents: Dict[str, str]) -> None:
        self.documents = documents

    @property
    def name(self) -> str:
        return "simple_keyword"

    def retrieve(self, query: str, top_k: int = 3) -> RAGContext:
        query_lower = query.lower()
        results = []
        for title, content in self.documents.items():
            if any(kw in content.lower() for kw in query_lower.split()):
                results.append(content)
        return RAGContext(chunks=results[:top_k])


class CompanyKnowledge(KnowledgeCard):
    """公司知识库。"""

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        self._retriever: Optional[SimpleKeywordRetriever] = None

    def on_mount(self, host: HostContext) -> None:
        docs = self._load_documents()
        self._retriever = SimpleKeywordRetriever(docs)
        host.knowledge_hub.register_retriever(self.name, self._retriever)
        print(f"[MOUNT] Knowledge 卡已插入: {self.display_name}")

    def on_unmount(self, host: HostContext) -> None:
        host.knowledge_hub.unregister_retriever(self.name)
        self._retriever = None
        print(f"[UNMOUNT] Knowledge 卡已拔出: {self.display_name}")

    def get_retriever(self) -> RAGRetriever:
        if self._retriever is None:
            docs = self._load_documents()
            return SimpleKeywordRetriever(docs)
        return self._retriever

    def get_assets(self) -> Dict[str, Path]:
        return {}

    def _load_documents(self) -> Dict[str, str]:
        docs_dir = self.card_root / "docs"
        documents = {}
        if not docs_dir.exists():
            return documents
        for doc_path in sorted(docs_dir.glob("*.md")):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    documents[doc_path.stem] = f.read()
            except Exception:
                pass
        return documents
```

---

## 测试与验证

开发完卡片后，使用以下步骤验证自包含性：

```bash
# 1. 插卡
python scripts/mount_cards.py --skill templates/skill_card_template

# 2. 临时隐藏源代码（模拟卡片独立运行）
mv skills/hnu_freshman skills/hnu_freshman_backup

# 3. 运行主机
python -m zhixia

# 4. 验证通过后恢复
mv skills/hnu_freshman_backup skills/hnu_freshman
```

如果卡片正确实现了自包含导入，即使 `skills/` 目录被隐藏，卡片仍能从 `cards/slot_a/` 正常加载。

---

## 常见问题

**Q: 卡片可以依赖第三方 Python 包吗？**
A: 可以，但应在 `manifest.json` 的 `dependencies` 中声明，并在文档中说明安装方式。主机不负责自动安装依赖。

**Q: 卡片内部可以使用相对导入（`from .tools import xxx`）吗？**
A: 不建议。因为卡片是通过 `importlib.util.spec_from_file_location` 动态加载的，不是作为标准包导入的，相对导入可能失效。请使用本文档推荐的 `sys.path` 方式。

**Q: 多个卡片的工具模块名冲突怎么办？**
A: 只要卡片的 `tools/` 目录内模块名不冲突即可。不同卡片加载时会通过不同的 `sys.path` 前缀隔离，不会互相影响。

**Q: 如何调试卡片加载失败？**
A: 设置 `LOG_LEVEL=DEBUG`，`card_loader.py` 会输出详细的模块加载路径和 `sys.path` 状态。
