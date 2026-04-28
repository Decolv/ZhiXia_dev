# Checklist

## 接口定义检查项

- [x] KnowledgeProvider 接口已定义在 `zhixia/core/card_base.py`
- [x] 数据模型类已定义（ListeningMaterial, Sentence, WritingExample, VocabularyItem）
- [x] CardManifest 已增加 `content_types` 和 `supported_exams` 字段
- [x] 英语考试知识卡 `manifest.json` 已更新

## 知识卡实现检查项

- [x] 英语考试知识卡实现 `KnowledgeProvider` 接口
- [x] `get_listening_materials()` 方法正确实现
- [x] `get_sentences()` 方法正确实现
- [x] `get_writing_examples()` 方法正确实现
- [x] `get_vocabulary()` 方法正确实现

## 技能卡工具解耦检查项

- [x] 考试准备计划器工具
  - [x] 构造函数接收 `knowledge_provider` 参数
  - [x] 移除硬编码路径依赖
  - [x] 无知识卡时有降级处理
- [x] 听力辅助器工具
  - [x] 构造函数接收 `knowledge_provider` 参数
  - [x] 使用接口获取内容
  - [x] 有缓存机制
- [x] 长难句助力器工具
  - [x] 构造函数接收 `knowledge_provider` 参数
  - [x] 使用接口获取内容
  - [x] 移除文件路径解析
- [x] 词汇复习器工具
  - [x] 构造函数接收 `knowledge_provider` 参数
  - [x] 使用接口获取内容
  - [x] 支持动态切换词汇库
- [x] 作文辅导器工具
  - [x] 构造函数接收 `knowledge_provider` 参数
  - [x] 使用接口获取内容
  - [x] 支持获取不同考试类型范文

## 技能卡主文件检查项

- [x] `on_mount()` 中查询可用知识卡
- [x] 根据内容类型匹配合适的知识卡
- [x] 知识卡正确注入到工具中
- [x] 无知识卡时有降级逻辑

## 测试验证检查项

- [x] 技能卡搭配原知识卡功能正常
- [x] 技能卡可搭配其他知识卡
- [x] 无知识卡时降级行为正常
