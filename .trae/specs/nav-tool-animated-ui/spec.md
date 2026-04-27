# 导航工具动画式前端界面 Spec

## Why
当前导航工具返回的是纯文本路线指引，用户在获取导航信息时 Live2D 眼睛仍继续显示，缺乏沉浸式的导航体验。导航工具调用时应暂停眼睛动画，切换到专门的导航前端界面，提供路线动画和地图标注展示。

## What Changes
- **新增**: 导航前端界面渲染器（基于 Pygame 或 HTML）
- **新增**: 导航动画效果（路线图高亮、箭头指示、路径动画）
- **修改**: CampusNavigateTool 触发导航界面展示
- **修改**: DisplayCallbackHandler 在工具调用时暂停眼睛并展示导航界面
- **新增**: 导航界面支持地图图片和目的地实景照片展示

## Impact
- Affected specs: Live2D 眼睛显示系统、导航工具、回调系统
- Affected code:
  - `zhixia/display/` - 新增导航前端界面渲染器
  - `zhixia/display/base.py` - DisplayPayload 支持导航界面模式
  - `skills/hnu_freshman/tools/campus_navigate.py` - 触发导航界面
  - `zhixia/core/host_orchestrator.py` - 回调系统联动导航界面

## ADDED Requirements

### Requirement: 导航前端界面
系统 SHALL 在导航工具被调用时展示专门的导航界面。

#### Scenario: 工具调用时
- **WHEN** 用户请求校园导航
- **THEN** Live2D 眼睛窗口隐藏或暂停
- **AND** 导航界面窗口展示
- **AND** 显示路线动画、地图标注和目的地信息

#### Scenario: 导航完成后
- **WHEN** 导航信息展示完毕
- **THEN** 导航界面关闭
- **AND** Live2D 眼睛恢复显示

### Requirement: 导航界面内容
导航界面 SHALL 展示以下内容：
- 出发地和目的地名称
- 步行路线文字描述
- 路线图标注（箭头指示方向）
- 周边设施提示
- 预估步行时间

### Requirement: 导航界面动画效果
导航界面 SHALL 提供以下动画效果：
- 路线路径逐步高亮动画
- 箭头方向指示动画
- 目的地标记闪烁效果
- 信息逐行展示动画

## MODIFIED Requirements

### Requirement: DisplayPayload扩展
**修改后新增字段**:
```python
@dataclass
class DisplayPayload:
    # ... 现有字段
    # 导航界面控制
    show_nav_ui: bool = False  # 是否展示导航界面
    nav_data: Optional[Dict] = None  # 导航数据
    nav_completed: bool = False  # 导航是否完成（恢复眼睛）
```

### Requirement: DisplayOutput扩展
**新增方法**:
```python
def show_navigation_ui(self, nav_data: Dict) -> None:
    """展示导航界面。"""
    pass

def hide_navigation_ui(self) -> None:
    """隐藏导航界面。"""
    pass
```

## REMOVED Requirements

无移除需求。
