# Tasks

## 阶段一：定义接口和基础架构

- [x] Task 1: 定义 KnowledgeProvider 接口
  - [x] SubTask 1.1: 在 `zhixia/core/card_base.py` 中定义 `KnowledgeProvider` 协议类
  - [x] SubTask 1.2: 定义数据模型（ListeningMaterial, Sentence, WritingExample, VocabularyItem）
  - [x] SubTask 1.3: 定义接口方法签名

- [x] Task 2: 更新知识卡 manifest.json 格式
  - [x] SubTask 2.1: 在 `CardManifest` 中增加 `content_types` 和 `supported_exams` 字段
  - [x] SubTask 2.2: 更新英语考试知识卡的 `manifest.json`

## 阶段二：知识卡实现接口

- [x] Task 3: 英语考试知识卡实现 KnowledgeProvider 接口
  - [x] SubTask 3.1: 修改 `card.py` 实现 `KnowledgeProvider` 接口
  - [x] SubTask 3.2: 实现 `get_listening_materials()` 方法
  - [x] SubTask 3.3: 实现 `get_sentences()` 方法
  - [x] SubTask 3.4: 实现 `get_writing_examples()` 方法
  - [x] SubTask 3.5: 实现 `get_vocabulary()` 方法

## 阶段三：技能卡工具解耦改造

- [x] Task 4: 重构考试准备计划器工具
  - [x] SubTask 4.1: 修改构造函数接收 `knowledge_provider` 参数
  - [x] SubTask 4.2: 移除硬编码路径依赖
  - [x] SubTask 4.3: 添加无知识卡时的降级处理

- [x] Task 5: 重构听力辅助器工具
  - [x] SubTask 5.1: 修改构造函数接收 `knowledge_provider` 参数
  - [x] SubTask 5.2: 使用 `get_listening_materials()` 接口获取内容
  - [x] SubTask 5.3: 添加缓存机制避免重复查询

- [x] Task 6: 重构长难句助力器工具
  - [x] SubTask 6.1: 修改构造函数接收 `knowledge_provider` 参数
  - [x] SubTask 6.2: 使用 `get_sentences()` 接口获取内容
  - [x] SubTask 6.3: 移除文件路径解析逻辑

- [x] Task 7: 重构词汇复习器工具
  - [x] SubTask 7.1: 修改构造函数接收 `knowledge_provider` 参数
  - [x] SubTask 7.2: 使用 `get_vocabulary()` 接口获取内容
  - [x] SubTask 7.3: 支持动态切换不同考试类型的词汇库

- [x] Task 8: 重构作文辅导器工具
  - [x] SubTask 8.1: 修改构造函数接收 `knowledge_provider` 参数
  - [x] SubTask 8.2: 使用 `get_writing_examples()` 接口获取内容
  - [x] SubTask 8.3: 支持获取不同考试类型的范文

## 阶段四：技能卡主文件改造

- [x] Task 9: 技能卡动态发现和绑定知识卡
  - [x] SubTask 9.1: 在 `on_mount()` 中查询可用的知识卡
  - [x] SubTask 9.2: 根据内容类型匹配合适的知识卡
  - [x] SubTask 9.3: 将知识卡注入到工具构造函数中
  - [x] SubTask 9.4: 处理无知识卡时的降级逻辑

## 阶段五：测试和验证

- [x] Task 10: 测试解耦后的功能
  - [x] SubTask 10.1: 测试技能卡搭配原知识卡
  - [x] SubTask 10.2: 创建测试用知识卡验证可替换性
  - [x] SubTask 10.3: 测试无知识卡时的降级行为

# Task Dependencies
- Task 2 依赖于 Task 1
- Task 3 依赖于 Task 1 和 Task 2
- Task 4-8 依赖于 Task 1 和 Task 3
- Task 9 依赖于 Task 4-8
- Task 10 依赖于 Task 9
