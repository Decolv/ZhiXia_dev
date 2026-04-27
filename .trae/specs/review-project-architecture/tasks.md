# 项目架构审查与优化任务清单

## 任务列表

- [x] 任务1：修复CardLoader核心Bug — 工具卸载逻辑、变化检测、资源清理
  - [x] 1.1 修复 `_unmount_card` 中注销所有工具的问题，改为仅注销当前卡片注册的工具
  - [x] 1.2 修复 `detect_change` 在挂载失败后吞掉检测的问题
  - [x] 1.3 修复 `_unmount_card` 中未恢复 `host.card_root` 的问题
  - [x] 1.4 清理 `set_current_card` 死代码，修复 `is_slot_empty` 语义

- [x] 任务2：修复PersonaHolder多层人设叠加问题
  - [x] 2.1 将 `_overlay` 从单层改为列表 `List[Tuple[str, str]]` 存储 `(card_name, persona)`
  - [x] 2.2 修改 `set_overlay` 支持追加/更新人设
  - [x] 2.3 修改 `clear_overlay` 支持按卡片名称清除指定人设
  - [x] 2.4 修改 `current_persona` 属性，合并所有叠加人设

- [x] 任务3：修复Live2D眼睛显示模块问题
  - [x] 3.1 修复渲染线程读取共享状态时的线程竞争问题
  - [x] 3.2 实现自动眨眼逻辑（基于 `blink_interval_ms`）
  - [x] 3.3 添加渲染循环异常处理（try/finally）防止stop()死锁
  - [x] 3.4 删除 `_is_thinking` 死代码

- [x] 任务4：修复NavUI/NavDisplay问题
  - [x] 4.1 优化NavUI渲染器：锁内仅复制数据，锁外渲染
  - [x] 4.2 修复NavDisplay展示导航时隐藏眼睛窗口
  - [x] 4.3 实现pygame生命周期全局管理器（引用计数）
  - [x] 4.4 添加Live2dEyeDisplay.hide()/show_eyes()方法

- [x] 任务5：优化HostContext线程安全性
  - [x] 5.1 为 `response_processors` 列表操作添加锁保护
  - [x] 5.2 为 `AgentConfigurator` 添加锁保护

- [x] 任务6：优化SkillCard.get_tools()返回类型
  - [x] 6.1 修改 `SkillCard.get_tools()` 返回类型为 `List[Any]`
  - [x] 6.2 更新 `card_loader.py` 中调用点适配新返回类型
  - [x] 6.3 更新技能卡实现 `get_tools()` 方法，添加 `registered_tool_names` 跟踪
  - [x] 6.4 添加 `llm_engine` 到 `HostContext` 和 `HostOrchestrator`

- [x] 任务7：验证所有修复并运行测试
  - [x] 7.1 运行编译检查确保无语法错误
  - [x] 7.2 验证所有模块导入成功
  - [x] 7.3 验证PersonaHolder多层人设叠加
  - [x] 7.4 验证AgentConfigurator线程安全

## 任务依赖关系

- 任务1 无依赖（可并行）
- 任务2 无依赖（可并行）
- 任务3 无依赖（可并行）
- 任务4 依赖于 任务3（共享pygame管理）
- 任务5 无依赖（可并行）
- 任务6 依赖于 任务1（CardLoader逻辑变更）
- 任务7 依赖于 所有其他任务
