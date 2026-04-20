# MemPalace AI记忆系统集成评估

## Why
当前ZhiXia语音助手缺乏跨会话记忆能力，每次对话都是独立的。MemPalace是一个本地优先的AI记忆系统，在LongMemEval基准测试中达到96.6%的原始召回率，可能为小匣提供长期记忆能力。

## What Changes
- 研究MemPalace的核心架构和功能特性
- 评估与ZhiXia项目的集成可能性
- 分析技术可行性和资源消耗

## Impact
- Affected specs: LLM上下文管理、跨会话记忆
- Affected code: 
  - `zhixia/config/settings.py` (可能需要新增记忆配置)
  - `zhixia/pipeline/orchestrator.py` (消息构建时注入历史记忆)

## ADDED Requirements

### Requirement: MemPalace技术评估
系统 SHALL 提供MemPalace的技术调研报告，包括：
- 核心架构分析
- 与ZhiXia的集成方案
- 资源消耗评估

### Requirement: 跨会话记忆（可选）
系统 SHALL 为小匣提供跨会话记忆能力，记住用户偏好和历史交互。

## Impact Assessment

### MemPalace核心特性
1. **本地优先**: 完全离线运行，使用ChromaDB + SQLite
2. **层次化存储**: Wings（项目/人）→ Rooms（主题）→ Halls（记忆类型）→ Drawers（原始内容）
3. **无损压缩**: AAAK格式，30倍压缩率
4. **语义检索**: 96.6% R@5召回率（无需LLM）
5. **知识图谱**: 带时间窗口的实体关系图

### 与ZhiXia的契合度分析

**优势:**
- 完全离线，适合嵌入式设备
- MIT开源许可
- Python 3.9+支持
- 轻量级（~300MB磁盘空间）
- 可记住用户偏好和历史对话

**挑战:**
- RK3588设备内存有限（ChromaDB需要一定内存）
- 语音交互场景下记忆检索的延迟影响
- AAAK压缩可能增加处理开销

**推荐集成方案:**
1. 作为可选模块，默认关闭
2. 仅在用户启用时加载
3. 使用简化配置（关闭知识图谱，仅用向量存储）
