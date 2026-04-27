# 网络感知LLM切换（离线本地/在线云端）Spec

## Why
当前系统始终使用本地部署的RKLLM模型进行推理。当设备有网络连接时，用户希望使用更强大的云端大模型API（如OpenAI、Claude等）获得更好的回答质量；仅在设备无网络时回退到本地模型，确保离线可用性。

## What Changes
- 添加网络连通性检测模块
- 创建云端LLM引擎（支持OpenAI API格式）
- 实现LLM引擎自动切换逻辑：有网用云端，无网用本地
- 添加云端API配置支持（URL、API Key、模型名称等）
- 支持配置项控制此功能的启用/禁用

## Impact
- Affected specs: LLM引擎管理、配置系统
- Affected code:
  - `zhixia/config/settings.py` - 添加云端LLM配置
  - `zhixia/llm/` - 新增云端LLM引擎
  - `zhixia/utils/` - 新增网络检测工具
  - `zhixia/__main__.py` - 修改引擎创建逻辑

## ADDED Requirements

### Requirement: 网络连通性检测
系统 SHALL 提供可靠的网络连通性检测机制。

#### Scenario: 检测网络状态
- **WHEN** 系统需要判断网络状态
- **THEN** 系统能够准确判断设备是否可访问互联网

#### Scenario: 定期检测
- **WHEN** 系统运行期间
- **THEN** 定期检测网络状态变化（建议每30秒或按需检测）

### Requirement: 云端LLM引擎
系统 SHALL 支持通过标准OpenAI API格式调用云端大模型。

#### Scenario: 正常调用云端API
- **WHEN** 网络可用且配置有效
- **THEN** 系统能够成功调用云端API获取回答

#### Scenario: 流式输出支持
- **WHEN** 使用云端LLM引擎
- **THEN** 支持流式输出，与本地模型行为一致

### Requirement: 自动引擎切换
系统 SHALL 根据网络状态自动选择合适的LLM引擎。

#### Scenario: 有网络时使用云端
- **GIVEN** 设备有网络连接
- **WHEN** 用户发起对话请求
- **THEN** 系统使用云端大模型API进行推理

#### Scenario: 无网络时回退本地
- **GIVEN** 设备无网络连接
- **WHEN** 用户发起对话请求
- **THEN** 系统自动回退到本地RKLLM模型

#### Scenario: 网络变化时切换
- **GIVEN** 系统正在运行
- **WHEN** 网络状态发生变化（连接/断开）
- **THEN** 系统能够在下次请求时自动切换到合适的引擎

### Requirement: 云端API配置
系统 SHALL 支持通过配置文件设置云端API参数。

#### Scenario: 配置API密钥
- **WHEN** 用户在localconfig.json中配置云端API参数
- **THEN** 系统能够读取并使用这些配置

## MODIFIED Requirements

### Requirement: LLM引擎创建
**修改前**:
- 始终创建RKLLMEngine作为LLM引擎

**修改后**:
- 根据网络状态和配置决定创建云端引擎还是本地引擎
- 支持配置项`llm.enable_cloud_fallback`控制是否启用自动切换

## REMOVED Requirements

无移除需求。
