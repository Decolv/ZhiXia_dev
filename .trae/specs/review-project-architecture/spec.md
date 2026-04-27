# 项目整体架构审查与优化 Spec

## Why
项目经过多次功能迭代后，需要进行全面的架构审查，发现并修复潜在的BUG，优化代码结构，确保系统稳定性和可维护性。

## What Changes
- 修复CardLoader中卸载卡片时清除所有工具的核心Bug
- 修复CardLoader中detect_change在挂载失败后吞掉后续检测的问题
- 修复CardLoader中卸载后host.card_root未恢复的问题
- 修复PersonaHolder多层人设叠加问题（当前只支持单层）
- 修复Live2D眼睛显示模块的线程竞争、自动眨眼缺失、渲染线程崩溃处理
- 修复NavUI渲染器的锁内渲染、NavDisplay眼睛窗口未隐藏、pygame共享状态管理
- 优化HostContext线程安全性（response_processors列表保护）
- 优化SkillCard.get_tools()返回类型设计

## Impact
- Affected specs: host-card-decoupling, live2d-eyes-display, nav-tool-animated-ui
- Affected code: zhixia/core/card_loader.py, zhixia/core/card_base.py, zhixia/display/live2d_eyes.py, zhixia/display/live2d_display.py, zhixia/display/nav_ui.py, zhixia/display/nav_display.py

## MODIFIED Requirements

### Requirement: CardLoader工具生命周期管理
**现状问题**: `_unmount_card` 中遍历并注销了工具注册表中的**所有工具**，而非仅卸载当前卡片的工具。

**修改后**: 应在 `_mount_card` 时记录当前卡片注册的工具列表，`_unmount_card` 时仅注销这些工具。

### Requirement: CardLoader变化检测
**现状问题**: `detect_change` 在检测到变化后立即更新 `_last_signature`，无论挂载成功与否。如果挂载失败，后续无法重新检测。

**修改后**: 仅在挂载成功后更新 `_last_signature`，或在挂载失败时保留原签名以便重试。

### Requirement: CardLoader资源清理
**现状问题**: `_unmount_card` 中未恢复 `host.card_root` 到卸载前的值。

**修改后**: 卸载卡片后应恢复 `host.card_root`。

### Requirement: PersonaHolder多层人设叠加
**现状问题**: `PersonaHolder` 声称支持"多层人设叠加"，但实际只维护单层 `_overlay`。多张Skill卡同时挂载时，后挂载的会覆盖前一张的人设。

**修改后**: 使用列表存储多个 `(card_name, persona)` 对，支持真正的多层叠加。`get_current_persona()` 按顺序合并所有叠加人设。

### Requirement: Live2D眼睛线程安全
**现状问题**: 渲染线程读取 `current_state`、`pupil_x/y`、`current_open` 时未加锁，可能导致帧间数据不一致。

**修改后**: 渲染循环中读取共享状态时加锁保护，或使用线程安全的数据快照。

### Requirement: Live2D自动眨眼
**现状问题**: `PRESET_STATES` 定义了 `blink_interval_ms`，但渲染循环中没有实现自动眨眼逻辑。

**修改后**: 在渲染循环中检查自动眨眼间隔，到期时触发眨眼动画。

### Requirement: Live2D渲染线程异常处理
**现状问题**: `_render_loop` 没有try/except保护，pygame异常会导致线程静默退出且 `_running` 仍为True，`stop()` 死锁。

**修改后**: 渲染循环添加try/finally，确保异常时正确设置 `_running = False`。

### Requirement: NavUI渲染性能
**现状问题**: 整个渲染过程（包括耗时的draw操作）都在锁内执行，阻塞外部调用。

**修改后**: 仅在锁内复制数据，在锁外执行渲染。

### Requirement: NavDisplay眼睛窗口管理
**现状问题**: `show_navigation_ui()` 只设置标志位，未真正隐藏眼睛窗口，导致UI重叠。

**修改后**: 展示导航界面时隐藏/最小化眼睛窗口，隐藏导航时恢复。

### Requirement: Pygame共享状态管理
**现状问题**: 两个渲染器都调用 `pygame.init()` 和 `pygame.quit()`，共享状态冲突。

**修改后**: 使用引用计数或全局管理器管理pygame生命周期。

## ADDED Requirements

### Requirement: HostContext线程安全
HostContext的 `response_processors` 列表操作应使用锁保护，确保多线程环境下的原子性。

#### Scenario: 多线程注册/注销处理器
- **WHEN** 多个线程同时调用 `register_response_processor` 和 `unregister_response_processor`
- **THEN** 所有操作应原子执行，不会出现数据丢失或异常

### Requirement: SkillCard工具返回类型
`SkillCard.get_tools()` 应返回工具列表（`List[Tool]`），而非 `ToolRegistry` 实例，避免卡片访问全局注册表。

#### Scenario: 卡片返回工具列表
- **WHEN** 卡片实现 `get_tools()` 方法
- **THEN** 应返回 `List[Tool]`，由主机负责注册到全局注册表
