# Tasks

- [x] Task 1: 扩展 DisplayPayload 和 DisplayOutput 支持导航界面
  - [x] SubTask 1.1: 在 DisplayPayload 中添加 show_nav_ui, nav_data, nav_completed 字段
  - [x] SubTask 1.2: 在 DisplayOutput 中添加 show_navigation_ui 和 hide_navigation_ui 方法
  - [x] SubTask 1.3: 更新 NullDisplay 提供空实现

- [x] Task 2: 实现导航前端界面渲染器
  - [x] SubTask 2.1: 创建 NavUIRenderer 类（基于 Pygame）
  - [x] SubTask 2.2: 实现路线动画效果
  - [x] SubTask 2.3: 实现地图图片标注展示
  - [x] SubTask 2.4: 实现信息逐行展示动画

- [x] Task 3: 创建 NavDisplay 类集成导航界面
  - [x] SubTask 3.1: 创建 NavDisplay 继承 DisplayOutput
  - [x] SubTask 3.2: 集成 NavUIRenderer
  - [x] SubTask 3.3: 实现显示/隐藏导航界面方法

- [x] Task 4: 回调系统联动导航界面
  - [x] SubTask 4.1: 更新 DisplayCallbackHandler 在导航工具调用时展示导航界面
  - [x] SubTask 4.2: 导航完成后隐藏界面并恢复眼睛

- [x] Task 5: 导航工具触发导航界面
  - [x] SubTask 5.1: 修改 CampusNavigateTool 返回结构化导航数据
  - [x] SubTask 5.2: 确保工具回调能正确传递导航数据

# Task Dependencies
- Task 2 independent
- Task 3 depends on Task 1, Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4
