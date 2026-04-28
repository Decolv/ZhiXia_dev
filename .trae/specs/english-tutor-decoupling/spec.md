# 英语考试辅导助手卡片解耦优化 Spec

## Why
当前英语考试辅导助手的技能卡和知识卡存在紧耦合问题：
1. 技能卡中的工具直接硬编码引用特定知识卡的路径
2. 一个技能卡无法灵活搭配不同内容的知识卡（如CET4知识卡、雅思知识卡等）
3. 知识卡的内容结构变更会影响技能卡的正常运行

需要通过解耦设计，实现技能卡与知识卡的松耦合，支持一个技能卡搭配多种知识卡的灵活组合。

## What Changes
- **新增**: 知识卡接口标准 `KnowledgeProvider` - 定义知识卡对外暴露的统一接口
- **新增**: 技能卡知识发现机制 - 通过主机动态发现可用的知识卡
- **新增**: 知识内容适配器 - 统一不同知识卡的内容格式
- **修改**: 技能卡工具不再硬编码知识卡路径，改为通过接口获取内容
- **修改**: 知识卡 `manifest.json` 增加内容类型声明
- **BREAKING**: 技能卡工具构造函数改为接收 `KnowledgeProvider` 接口而非硬编码路径

## Impact
- Affected specs: 英语考试辅导助手技能卡、知识卡架构
- Affected code:
  - `skills/english_tutor_skill/tools/*.py` - 所有工具改为接口调用
  - `skills/english_tutor_knowledge/card.py` - 实现 `KnowledgeProvider` 接口
  - `zhixia/core/card_base.py` - 新增 `KnowledgeProvider` 基类接口
  - `skills/english_tutor_skill/card.py` - 动态发现和绑定知识卡

## ADDED Requirements

### Requirement: 知识提供者接口 (KnowledgeProvider)
系统 SHALL 定义 `KnowledgeProvider` 接口，统一知识卡对外提供内容的方式。

#### Scenario: 知识卡实现接口
- **WHEN** 知识卡挂载时
- **THEN** 知识卡实现 `KnowledgeProvider` 接口
- **AND** 提供统一的内容查询方法

#### Scenario: 技能卡获取内容
- **GIVEN** 技能卡需要使用知识内容
- **WHEN** 技能卡通过 `HostContext` 获取 `KnowledgeProvider`
- **THEN** 通过标准接口获取内容，无需关心具体知识卡

### Requirement: 知识内容类型声明
系统 SHALL 允许知识卡在 `manifest.json` 中声明提供的内容类型。

#### Scenario: 知识卡声明内容类型
- **GIVEN** 知识卡 `manifest.json`
- **WHEN** 添加 `content_types` 字段
- **THEN** 声明支持的内容类型（listening/sentences/writing/vocabulary）
- **AND** 主机可根据类型匹配合适的知识卡

```json
{
  "content_types": ["listening", "sentences", "writing", "vocabulary"],
  "supported_exams": ["cet4", "cet6", "ielts"]
}
```

### Requirement: 技能卡动态发现知识卡
系统 SHALL 支持技能卡动态发现和绑定知识卡。

#### Scenario: 技能卡挂载时发现知识卡
- **WHEN** 英语辅导技能卡挂载时
- **THEN** 通过 `HostContext` 查询可用的知识卡
- **AND** 根据内容类型匹配合适的知识卡
- **AND** 将知识卡的 `KnowledgeProvider` 注入到工具中

#### Scenario: 无知识卡时的降级处理
- **GIVEN** 技能卡挂载时无配套知识卡
- **THEN** 技能卡工具提供降级功能（如使用内置示例数据）
- **AND** 提示用户知识卡未挂载

### Requirement: 知识内容查询接口
系统 SHALL 提供标准化的知识内容查询接口。

```python
class KnowledgeProvider(Protocol):
    def get_listening_materials(
        self, 
        exam_type: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> List[ListeningMaterial]:
        ...
    
    def get_sentences(
        self,
        difficulty: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Sentence]:
        ...
    
    def get_writing_examples(
        self,
        exam_type: Optional[str] = None,
        essay_type: Optional[str] = None
    ) -> List[WritingExample]:
        ...
    
    def get_vocabulary(
        self,
        exam_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[VocabularyItem]:
        ...
```

## MODIFIED Requirements

### Requirement: 技能卡工具构造函数
**修改前**:
```python
class ListeningAssistantTool(Tool):
    KNOWLEDGE_BASE_PATH = "d:\\Code\\ZhiXia_dev\\skills\\english_tutor_knowledge\\docs\\listening"
    
    def __init__(self, llm_engine=None):
        ...
```

**修改后**:
```python
class ListeningAssistantTool(Tool):
    def __init__(self, llm_engine=None, knowledge_provider: Optional[KnowledgeProvider] = None):
        self._knowledge_provider = knowledge_provider
        ...
```

### Requirement: 知识卡 manifest.json
**修改后新增**:
```json
{
  "content_types": ["listening", "sentences", "writing", "vocabulary"],
  "supported_exams": ["cet4", "cet6", "ielts"],
  "content_version": "1.0.0"
}
```

## REMOVED Requirements

### Requirement: 技能卡硬编码知识卡路径
**Reason**: 违反解耦原则，技能卡不应依赖特定知识卡的路径
**Migration**: 通过 `KnowledgeProvider` 接口获取内容
