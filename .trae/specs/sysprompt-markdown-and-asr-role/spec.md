# 系统提示词Markdown格式与ASR结果角色规范

## Why
当前系统提示词格式不够清晰，且需要明确ASR识别结果始终作为user_prompt传入LLM。

## What Changes
- 将系统提示词改为类似Markdown的结构化格式
- 确保ASR识别结果始终作为user角色消息

## Impact
- Affected specs: LLM配置、消息构建
- Affected code: 
  - `zhixia/config/settings.py`
  - `zhixia/pipeline/orchestrator.py`

## MODIFIED Requirements

### Requirement: 系统提示词格式
**修改前**: 简单的多行字符串
**修改后**: 使用类似Markdown的结构化格式（标题、列表等）

### Requirement: ASR结果角色
**修改前**: ASR结果作为user角色消息
**修改后**: 保持不变，确保ASR结果始终作为user角色消息
