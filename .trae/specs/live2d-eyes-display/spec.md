# Live2D 眼睛动态展示 Spec

## Why
当前项目的显示输出较为基础（仅文本输出），缺乏视觉互动体验。添加 Live2D 眼睛可以在智能体说话时提供生动的视觉反馈，增强用户交互体验和情感连接。

## What Changes
- **新增**: Live2D 眼睛渲染器，支持眨眼、瞳孔移动等动画
- **新增**: 与 DisplayPayload 状态联动的眼睛互动逻辑
- **新增**: 独立的 GUI 窗口或 Web 展示面板
- **修改**: DisplayOutput 接口扩展支持 Live2D 状态控制
- **新增**: 默认 Live2D 眼睛模型资源

## Impact
- Affected specs: Display 显示系统、Agent 回调系统
- Affected code:
  - `zhixia/display/base.py` - 扩展 DisplayPayload 支持 Live2D 状态
  - `zhixia/display/` - 新增 Live2D 眼睛渲染器
  - `zhixia/core/host_orchestrator.py` - 初始化 Live2D 显示
  - `zhixia/agent/callbacks.py` - 回调触发眼睛动画
  - `assets/live2d/` - Live2D 眼睛模型资源

## ADDED Requirements

### Requirement: Live2D 眼睛渲染器
系统 SHALL 提供一个 Live2D 眼睛渲染器，在独立窗口中展示动态眼睛。

#### Scenario: 项目启动时
- **WHEN** 项目启动
- **THEN** Live2D 眼睛窗口自动创建并显示
- **AND** 眼睛处于自然待机状态（偶尔眨眼）

#### Scenario: 智能体思考时
- **WHEN** Agent 进入 thinking 状态
- **THEN** 眼睛显示思考表情（如微微眯眼、瞳孔转动）

#### Scenario: 智能体说话时
- **WHEN** Agent 输出回答文本
- **THEN** 眼睛根据情绪状态变化（开心时睁大、平静时自然等）

### Requirement: 眼睛互动模式
Live2D 眼睛 SHALL 支持以下互动状态：

| 状态 | 眼睛表现 |
|------|---------|
| neutral（平静）| 自然眨眼，瞳孔轻微移动 |
| thinking（思考）| 微微眯眼，瞳孔上下转动 |
| happy（开心）| 眼睛弯月形，眨眼频率降低 |
| working（工作中）| 瞳孔快速移动，偶尔眨眼 |
| sad（悲伤）| 半闭眼，眨眼频率低 |
| surprised（惊讶）| 睁大眼，眨眼频率高 |

### Requirement: Live2D 模型资源
系统 SHALL 内置至少一套默认的 Live2D 眼睛模型。

#### Scenario: 默认模型
- **WHEN** 未指定自定义模型
- **THEN** 使用内置的可爱风格眼睛模型
- **AND** 模型文件位于 `assets/live2d/eyes/` 目录

## MODIFIED Requirements

### Requirement: DisplayPayload扩展
**修改后**:
```python
@dataclass
class DisplayPayload:
    text: str
    emotion: str = "neutral"
    is_thinking: bool = False
    thinking_text: str = ""
    images: Optional[List[Union[str, Path]]] = None
    image_captions: Optional[List[str]] = None
    # Live2D 眼睛控制
    eye_state: str = ""  # 眼睛状态：auto/neutral/thinking/happy/working/sad/surprised
    blink_override: Optional[bool] = None  # 是否强制眨眼
    metadata: Dict = field(default_factory=dict)
```

### Requirement: DisplayOutput抽象基类
**新增方法**:
```python
def set_eye_state(self, state: str) -> None:
    """设置眼睛状态。默认空操作。"""
    pass

def set_eye_emotion(self, emotion: str) -> None:
    """设置眼睛情绪表情。默认空操作。"""
    pass
```

## REMOVED Requirements

无移除需求。
