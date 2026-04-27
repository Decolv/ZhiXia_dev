# Tasks

- [ ] Task 1: 创建 zhixia.md 模板文件和目录结构
  - [ ] SubTask 1.1: 在 `skills/hnu_freshman/` 下创建 `zhixia.md` 空模板文件
  - [ ] SubTask 1.2: 在 `skills/` 目录下创建 `.zhixia_profiles/` 目录用于存储用户画像

- [ ] Task 2: 实现用户画像加载器
  - [ ] SubTask 2.1: 在 `zhixia/core/` 下创建 `user_profile.py` 模块
  - [ ] SubTask 2.2: 实现 `UserProfile` 类加载和解析 zhixia.md
  - [ ] SubTask 2.3: 实现用户画像的保存和更新方法

- [ ] Task 3: 扩展 HostContext 支持用户画像
  - [ ] SubTask 3.1: 在 HostContext 中添加 user_profile 字段
  - [ ] SubTask 3.2: 在卡片挂载时加载用户画像
  - [ ] SubTask 3.3: 在卡片卸载时保存更新的用户画像

- [ ] Task 4: 注入用户画像到 Agent system prompt
  - [ ] SubTask 4.1: 在 HostOrchestrator 的 `_run_agent` 中读取用户画像
  - [ ] SubTask 4.2: 将用户画像注入到 system prompt 中

- [ ] Task 5: 创建用户画像维护工具
  - [ ] SubTask 5.1: 创建工具类 `UserProfileUpdater` 用于动态更新用户画像
  - [ ] SubTask 5.2: 在对话结束后自动调用更新方法

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 3
