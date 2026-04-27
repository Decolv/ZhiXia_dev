# 替换FAQ为LLM智能调度生成与思考播报 Spec

## Why
当前项目中使用了硬编码的FAQ形式（如`CampusLifeGuideTool`中的`_faq`字典）来回答常见问题，这种方式缺乏灵活性，无法根据上下文智能生成答案。用户希望：
1. 完全移除FAQ这种硬编码形式
2. 使用LLM智能调度生成答案
3. 将模型的思考过程播报给用户，增加交互透明度

## What Changes
- **BREAKING**: 移除`CampusLifeGuideTool`中的硬编码FAQ数据库
- **BREAKING**: 将`CampusLifeGuideTool`改造为智能调度工具，使用LLM生成答案
- 新增思考过程播报功能，在`DisplayOutput`中支持实时显示思考状态
- 修改`StreamingDisplayHandler`以支持思考过程播报
- 更新`NullDisplay`以支持思考状态更新

## Impact
- Affected specs: Agent回调系统、Display显示系统、工具调用机制
- Affected code:
  - `skills/hnu_freshman/tools/life_guide.py` - 移除FAQ，改为LLM生成
  - `zhixia/display/base.py` - DisplayPayload支持思考内容
  - `zhixia/display/null_display.py` - 实现思考状态更新
  - `zhixia/agent/callbacks.py` - StreamingDisplayHandler支持思考播报
  - `zhixia/pipeline/orchestrator.py` - 集成思考播报到流水线
  - `zhixia/core/host_orchestrator.py` - 集成思考播报

## ADDED Requirements

### Requirement: 思考过程实时播报
系统 SHALL 支持将LLM的思考过程实时播报给用户。

#### Scenario: Agent思考时
- **WHEN** Agent进入thinking状态
- **THEN** 系统应通过Display显示思考状态
- **AND** 可通过语音播报思考内容（可选）

#### Scenario: 工具调用时
- **WHEN** Agent决定调用工具
- **THEN** 系统应显示"正在查询..."等状态
- **AND** 播报工具调用意图

### Requirement: LLM智能生成校园生活指南
系统 SHALL 使用LLM智能生成校园生活相关答案，而非硬编码FAQ。

#### Scenario: 用户询问食堂信息
- **GIVEN** 用户询问"食堂在哪里"
- **WHEN** `CampusLifeGuideTool`被调用
- **THEN** 工具应使用LLM生成个性化回答
- **AND** 回答应基于上下文和知识库

## MODIFIED Requirements

### Requirement: CampusLifeGuideTool重构
**修改前**:
- 使用硬编码的`_faq`字典存储常见问题答案
- 通过关键词匹配返回预设答案

**修改后**:
- 移除`_faq`字典
- 使用LLM动态生成答案
- 保留校园生活知识作为上下文注入LLM
- 支持流式生成和思考过程展示

### Requirement: DisplayPayload扩展
**修改前**:
```python
@dataclass
class DisplayPayload:
    text: str
    emotion: str = "neutral"
    is_thinking: bool = False
    metadata: Dict = field(default_factory=dict)
```

**修改后**:
```python
@dataclass
class DisplayPayload:
    text: str
    emotion: str = "neutral"
    is_thinking: bool = False
    thinking_text: str = ""  # 新增：思考内容
    metadata: Dict = field(default_factory=dict)
```

### Requirement: BaseCallbackHandler扩展
**修改前**:
```python
def on_agent_thought(self, run_id: str, thought: str, **kwargs: Any) -> None:
    """Agent 产生 Thought（可用于流式展示思考过程）。"""
    pass
```

**修改后**:
```python
def on_agent_thought(self, run_id: str, thought: str, **kwargs: Any) -> None:
    """Agent 产生 Thought（可用于流式展示思考过程）。"""
    pass

def on_thinking_start(self, run_id: str, **kwargs: Any) -> None:
    """开始思考时调用。"""
    pass

def on_thinking_end(self, run_id: str, **kwargs: Any) -> None:
    """结束思考时调用。"""
    pass
```

## REMOVED Requirements

### Requirement: 硬编码FAQ数据库
**Reason**: 缺乏灵活性，无法根据上下文智能生成答案
**Migration**: 使用LLM动态生成，保留知识作为上下文参考

### Requirement: 关键词匹配回答机制
**Reason**: 过于简单，无法处理复杂查询
**Migration**: 使用LLM理解用户意图并生成回答
