# 大模型调参与系统提示词人设

## Why
当前LLM配置过于保守（max_new_tokens=32, temperature=0.8），限制了回答的丰富性和想象力。同时系统提示词过于简单，缺乏明确的人设定义。

## What Changes
- 调整LLM参数以支持更丰富的回答（增加max_new_tokens、调整temperature等）
- 建立"智能助手小匣"的系统提示词人设
- 添加结构化输出支持的相关配置

## Impact
- Affected specs: LLM引擎配置、系统提示词
- Affected code: 
  - `zhixia/config/settings.py`
  - 可能影响 `zhixia/pipeline/orchestrator.py` 中的消息构建逻辑

## ADDED Requirements

### Requirement: 智能助手小匣人设
系统 SHALL 为LLM提供一个明确的"智能助手小匣"人设。

#### Scenario: 使用小匣人设进行对话
- **WHEN** 用户与系统进行对话
- **THEN** LLM应当以小匣的身份和风格进行回答

### Requirement: 更丰富的回答生成
系统 SHALL 支持生成长度更长、内容更丰富的回答。

#### Scenario: 生成详细回答
- **WHEN** 用户提出需要详细解释的问题
- **THEN** LLM应当能够生成足够长度的详细回答

## MODIFIED Requirements

### Requirement: LLM配置参数
**修改前**: 
- max_new_tokens: 32（过于受限）
- max_context_len: 512
- temperature: 0.8
- system_prompt: "你是AI助手，用一句话简短回答。"

**修改后**:
- max_new_tokens: 256（支持更丰富回答）
- max_context_len: 1024（支持更长上下文）
- temperature: 1.0（增加创造性和想象力）
- top_p: 0.95（保持不变）
- system_prompt: 完整的"智能助手小匣"人设提示词

## REMOVED Requirements

无移除需求。
