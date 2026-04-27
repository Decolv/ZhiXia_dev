# Tasks

- [x] Task 1: 新增 AgentConfigurator 和 ResponsePostProcessor 接口
  - [x] SubTask 1.1: 在 card_base.py 中定义 AgentConfigurator 接口
  - [x] SubTask 1.2: 在 card_base.py 中定义 ResponsePostProcessor 接口
  - [x] SubTask 1.3: 更新 HostContext 支持新接口

- [x] Task 2: 重构 HostOrchestrator 移除卡片特定逻辑
  - [x] SubTask 2.1: 移除 `__NAV_DATA__` 解析逻辑
  - [x] SubTask 2.2: 使用 ResponsePostProcessor 处理响应
  - [x] SubTask 2.3: 支持通过 AgentConfigurator 配置 Agent

- [x] Task 3: 导航卡片自行注册后处理器
  - [x] SubTask 3.1: 创建 NavResponseProcessor 类
  - [x] SubTask 3.2: 在 HNUFreshmanSkill.on_mount() 中注册
  - [x] SubTask 3.3: 在 on_unmount() 中注销

- [x] Task 4: 验证无卡时主机纯净
  - [x] SubTask 4.1: 确保无卡时不泄露卡片内容
  - [x] SubTask 4.2: 确保默认 Agent 配置正确

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1, Task 2
- Task 4 depends on Task 2, Task 3
