# 技能卡用户画像文件 (zhixia.md) Spec

## Why
当前系统在每次交互时都从零开始，缺乏对用户偏好的长期记忆。通过在技能卡中维护 `zhixia.md` 文件，可以实现用户画像的动态积累和更新，使AI能够根据用户特质提供个性化服务。

## What Changes
- **新增**: 技能卡目录下创建 `zhixia.md` 文件用于维护用户画像
- **新增**: `UserProfileLoader` 类加载和管理用户画像
- **新增**: `UserProfileUpdater` 在交互后更新用户画像
- **修改**: HostContext 增加用户画像上下文支持
- **修改**: Agent system prompt 注入用户画像信息

## Impact
- Affected specs: 技能卡架构、Agent system prompt 构建
- Affected code:
  - `zhixia/core/card_base.py` - 新增用户画像加载器
  - `zhixia/core/host_orchestrator.py` - 注入用户画像到 prompt
  - `skills/*/` - 每个技能卡增加 `zhixia.md` 文件

## ADDED Requirements

### Requirement: 技能卡用户画像文件
系统 SHALL 在每个技能卡目录下维护一个 `zhixia.md` 文件。

#### Scenario: 首次交互
- **WHEN** 用户首次与某个技能卡交互
- **THEN** `zhixia.md` 文件为空或仅有模板结构
- **AND** 系统正常提供服务

#### Scenario: 交互后更新
- **WHEN** 用户完成一次对话交互
- **THEN** 系统分析对话内容提取用户特征
- **AND** 将用户特征追加到 `zhixia.md` 文件中

### Requirement: 用户画像注入
系统 SHALL 在构建 Agent system prompt 时注入用户画像信息。

#### Scenario: 使用用户画像
- **GIVEN** `zhixia.md` 包含用户画像信息
- **WHEN** 构建 Agent 对话上下文
- **THEN** 用户画像信息被注入到 system prompt 中
- **AND** AI 基于用户画像提供个性化回答

### Requirement: 用户画像格式
`zhixia.md` SHALL 使用 Markdown 格式，包含以下结构：

```markdown
# 用户画像

## 基本信息
- 称呼: 
- 身份: 
- 偏好: 

## 对话历史摘要
- 

## 用户特质
- 

## 注意事项
- 
```

## MODIFIED Requirements

### Requirement: HostContext扩展
**修改前**:
```python
@dataclass
class HostContext:
    tool_registry: ToolRegistry
    persona_holder: PersonaHolder
    knowledge_hub: KnowledgeHub
    display: Optional[DisplayOutput] = None
    config: Optional[Any] = None
    card_root: Optional[Path] = None
```

**修改后**:
```python
@dataclass
class HostContext:
    tool_registry: ToolRegistry
    persona_holder: PersonaHolder
    knowledge_hub: KnowledgeHub
    user_profile: Optional[UserProfile] = None  # 新增：用户画像
    display: Optional[DisplayOutput] = None
    config: Optional[Any] = None
    card_root: Optional[Path] = None
```

## REMOVED Requirements

无移除需求。
