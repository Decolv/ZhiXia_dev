# 主机与卡片深度解耦架构 Spec

## Why
当前主机编排器（HostOrchestrator）包含太多卡片特定逻辑（如导航数据解析），导致无卡时主机仍依赖卡片内容。同时，Agent 类型由主机配置决定，不同技能卡无法灵活切换 Agent 行为。需要实现主机与卡片的深度解耦。

## What Changes
- **新增**: AgentConfigurator 接口 - 卡片可通过此接口定义 Agent 配置
- **新增**: ResponsePostProcessor 接口 - 卡片可注册响应后处理器（如导航数据解析）
- **修改**: HostOrchestrator 移除卡片特定逻辑，改为使用扩展点
- **修改**: HostContext 提供 Agent 配置和响应处理的注册接口
- **BREAKING**: 卡片特定逻辑（如导航界面）不再由主机直接处理

## Impact
- Affected specs: 插卡式架构、Agent 构建机制、响应处理机制
- Affected code:
  - `zhixia/core/host_orchestrator.py` - 移除卡片特定逻辑，使用扩展点
  - `zhixia/core/card_base.py` - 新增 AgentConfigurator 和 ResponsePostProcessor 接口
  - `skills/hnu_freshman/card.py` - 卡片自行注册导航后处理器

## ADDED Requirements

### Requirement: Agent 配置接口
系统 SHALL 提供 AgentConfigurator 接口，允许技能卡配置 Agent 类型和行为。

#### Scenario: 卡片配置 Agent
- **WHEN** 技能卡挂载时
- **THEN** 卡片可通过 HostContext.agent_configurator 注册 Agent 配置
- **AND** Agent 类型由卡片决定（ReAct / ToolCalling / 其他）

#### Scenario: 无卡时默认 Agent
- **GIVEN** 无技能卡挂载
- **THEN** 主机使用默认 Agent 配置
- **AND** 主机不泄露任何卡片特定内容

### Requirement: 响应后处理器接口
系统 SHALL 提供 ResponsePostProcessor 接口，允许卡片注册响应处理逻辑。

#### Scenario: 导航工具注册后处理器
- **WHEN** 校园导航卡片挂载时
- **THEN** 注册 NavResponseProcessor 到主机
- **AND** 主机在生成响应后自动调用后处理器
- **THEN** 5秒后自动隐藏导航界面

#### Scenario: 无卡时主机行为
- **GIVEN** 无技能卡挂载
- **THEN** 主机不执行任何卡片特定的响应处理
- **AND** 响应直接传递给 Display

### Requirement: 主机纯净模式
系统 SHALL 在无卡时保持纯净，不包含任何卡片特定逻辑。

#### Scenario: 无卡模式运行
- **WHEN** 无技能卡插入
- **THEN** 主机仅包含基础 LLM 对话功能
- **AND** 不解析 `__NAV_DATA__` 等卡片特定标记
- **AND** 不触发任何卡片特定的 UI 展示

## MODIFIED Requirements

### Requirement: HostOrchestrator重构
**修改前**:
- 直接解析 `__NAV_DATA__` 标记
- 直接操作 `show_navigation_ui()`
- Agent 类型由主机配置决定

**修改后**:
- 使用 ResponsePostProcessor 处理响应
- 主机仅负责调用注册的处理器
- Agent 类型可通过 HostContext 动态配置

### Requirement: HostContext扩展
**修改后新增**:
```python
@dataclass
class HostContext:
    # ... 现有字段
    agent_configurator: Optional[AgentConfigurator] = None
    response_processors: List[ResponsePostProcessor] = field(default_factory=list)
```

## REMOVED Requirements

### Requirement: 主机中硬编码导航逻辑
**Reason**: 违反主机与卡片解耦原则
**Migration**: 导航逻辑由卡片自行注册 ResponsePostProcessor 处理
