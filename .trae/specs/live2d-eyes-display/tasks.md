# Tasks

- [x] Task 1: 创建 Live2D 眼睛资源目录和配置
  - [x] SubTask 1.1: 创建 `assets/live2d/eyes/` 目录
  - [x] SubTask 1.2: 创建默认眼睛模型配置文件 `model.json`
  - [x] SubTask 1.3: 使用 Pygame 绘制图形代替图片

- [x] Task 2: 实现 Live2D 眼睛渲染器
  - [x] SubTask 2.1: 选择 Pygame 渲染方案
  - [x] SubTask 2.2: 实现 `Live2dEyeRenderer` 类
  - [x] SubTask 2.3: 支持状态切换：neutral/thinking/happy/working/sad/surprised
  - [x] SubTask 2.4: 实现自动眨眼动画
  - [x] SubTask 2.5: 实现瞳孔跟随/转动动画

- [x] Task 3: 扩展 DisplayPayload 和 DisplayOutput
  - [x] SubTask 3.1: 在 `DisplayPayload` 中添加 `eye_state` 和 `blink_override` 字段
  - [x] SubTask 3.2: 在 `DisplayOutput` 中添加 `set_eye_state` 和 `set_eye_emotion` 方法
  - [x] SubTask 3.3: 更新 `NullDisplay` 提供空实现

- [x] Task 4: 创建 Live2D 眼睛 Display 实现
  - [x] SubTask 4.1: 创建 `Live2dEyeDisplay` 类继承 `DisplayOutput`
  - [x] SubTask 4.2: 集成渲染器到 Display 接口
  - [x] SubTask 4.3: 支持情绪到眼睛状态的自动映射

- [x] Task 5: 集成到 HostOrchestrator
  - [x] SubTask 5.1: 在 HostOrchestrator 初始化时创建 Live2D 眼睛显示
  - [x] SubTask 5.2: 配置眼睛显示为默认启用（enable_live2d_eyes 参数）

- [x] Task 6: 回调系统触发眼睛动画
  - [x] SubTask 6.1: 更新 `DisplayCallbackHandler` 在思考/说话时触发眼睛状态变化
  - [x] SubTask 6.2: 支持说完后眨眼一次

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 independent
- Task 4 depends on Task 2, Task 3
- Task 5 depends on Task 4
- Task 6 depends on Task 4, Task 5
