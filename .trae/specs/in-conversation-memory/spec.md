# 小匣对话内记忆增强方案

## Why
当前小匣在同一对话中缺乏有效的上下文记忆，无法记住前文提到的信息。参考MemPalace的层次化记忆思路，为小匣在单次对话内增强短期记忆能力。

## What Changes
- 实现对话内短期记忆模块
- 在消息构建时自动注入相关历史上下文
- 添加记忆配置选项（记忆长度、是否启用等）

## Impact
- Affected specs: LLM上下文管理、对话记忆
- Affected code: 
  - `zhixia/config/settings.py` (新增记忆配置)
  - `zhixia/pipeline/orchestrator.py` (消息构建时注入记忆)
  - 可能需要新增 `zhixia/memory/` 模块

## ADDED Requirements

### Requirement: 对话内短期记忆
系统 SHALL 在同一对话中记住前文提到的关键信息。

#### Scenario: 多轮对话记忆
- **WHEN** 用户进行多轮对话
- **THEN** 小匣能够记住前文提到的姓名、地点、偏好等信息

### Requirement: 可配置的记忆长度
系统 SHALL 允许用户配置记忆的最大轮数或token数量。

#### Scenario: 配置记忆长度
- **WHEN** 用户在配置中设置max_memory_rounds
- **THEN** 系统仅保留最近N轮对话历史

### Requirement: 智能上下文注入
系统 SHALL 根据当前问题智能提取并注入相关历史上下文。

#### Scenario: 引用前文内容
- **WHEN** 用户说"刚才提到的那个"或"他叫什么名字"
- **THEN** 系统能够从前文记忆中检索并注入相关信息

## MODIFIED Requirements

### Requirement: 消息构建逻辑
**修改前**: 仅包含system prompt和当前user输入
**修改后**: 包含system prompt + 相关历史记忆 + 当前user输入

### Requirement: LLM配置
**修改前**: 无记忆相关配置
**修改后**: 添加memory相关配置项

```python
# 新增配置项
memory_enabled: bool = True
max_memory_rounds: int = 5  # 最多记住5轮对话
max_memory_tokens: int = 512  # 记忆部分最大token数
```
