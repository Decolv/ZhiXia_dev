# Tasks
- [x] Task 1: 添加记忆配置项
  - [x] SubTask 1.1: 在LLMConfig中添加memory_enabled、max_memory_rounds、max_memory_tokens配置

- [x] Task 2: 实现对话内短期记忆模块
  - [x] SubTask 2.1: 创建ConversationMemory类用于存储和检索对话历史
  - [x] SubTask 2.2: 实现消息历史记录和截取逻辑

- [x] Task 3: 集成到消息构建流程
  - [x] SubTask 3.1: 修改_build_messages方法，在system prompt后注入记忆上下文
  - [x] SubTask 3.2: 测试多轮对话记忆功能

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
