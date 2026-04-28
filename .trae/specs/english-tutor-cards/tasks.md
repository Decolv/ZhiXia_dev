# Tasks

## 阶段一：知识卡 (Knowledge Card) 开发

- [x] Task 1: 创建英语考试知识卡目录结构
  - [x] SubTask 1.1: 创建 `skills/english_tutor_knowledge/` 目录
  - [x] SubTask 1.2: 创建 `manifest.json` 文件
  - [x] SubTask 1.3: 创建 `card.py` 主文件

- [x] Task 2: 创建听力内容知识库
  - [x] SubTask 2.1: 创建 `docs/listening/` 目录结构（按考试类型分类）
  - [x] SubTask 2.2: 创建示例听力材料文件（包含原文、翻译、难度）
  - [x] SubTask 2.3: 实现听力内容检索器

- [x] Task 3: 创建长难句知识库
  - [x] SubTask 3.1: 创建 `docs/sentences/` 目录结构
  - [x] SubTask 3.2: 创建示例长难句文件（包含原句、翻译、语法分析）
  - [x] SubTask 3.3: 实现长难句检索器

- [x] Task 4: 创建作文案例知识库
  - [x] SubTask 4.1: 创建 `docs/writing/` 目录结构
  - [x] SubTask 4.2: 创建作文模板和范文文件
  - [x] SubTask 4.3: 创建万能句库文件
  - [x] SubTask 4.4: 实现作文资源检索器

- [x] Task 5: 创建词汇库
  - [x] SubTask 5.1: 创建 `docs/vocabulary/` 目录结构
  - [x] SubTask 5.2: 创建核心词汇表文件
  - [x] SubTask 5.3: 创建词汇搭配文件

## 阶段二：技能卡 (Skill Card) 开发

- [x] Task 6: 创建英语考试技能卡基础结构
  - [x] SubTask 6.1: 创建 `skills/english_tutor_skill/` 目录
  - [x] SubTask 6.2: 创建 `manifest.json` 文件
  - [x] SubTask 6.3: 创建 `card.py` 主文件
  - [x] SubTask 6.4: 创建 `persona.json` 人设文件

- [x] Task 7: 实现考试准备计划器工具 (ExamPlannerTool)
  - [x] SubTask 7.1: 创建 `tools/exam_planner.py`
  - [x] SubTask 7.2: 实现用户画像记录功能
  - [x] SubTask 7.3: 实现薄弱点分析功能
  - [x] SubTask 7.4: 实现排期规划算法
  - [x] SubTask 7.5: 实现计划存储和读取

- [x] Task 8: 实现听力辅助器工具 (ListeningAssistantTool)
  - [x] SubTask 8.1: 创建 `tools/listening_assistant.py`
  - [x] SubTask 8.2: 实现从知识卡获取听力内容
  - [x] SubTask 8.3: 实现听力播放交互逻辑
  - [x] SubTask 8.4: 实现用户理解测试和反馈

- [x] Task 9: 实现长难句助力器工具 (LongSentenceTool)
  - [x] SubTask 9.1: 创建 `tools/long_sentence.py`
  - [x] SubTask 9.2: 实现长难句获取和展示
  - [x] SubTask 9.3: 实现语法知识点讲解
  - [x] SubTask 9.4: 实现交互式学习流程

- [x] Task 10: 实现词汇复习器工具 (VocabularyReviewerTool)
  - [x] SubTask 10.1: 创建 `tools/vocabulary_reviewer.py`
  - [x] SubTask 10.2: 实现词汇记忆计划制定
  - [x] SubTask 10.3: 实现滚动复习机制
  - [x] SubTask 10.4: 实现定期检测功能
  - [x] SubTask 10.5: 实现学习进度汇总和建议

- [x] Task 11: 实现作文辅导器工具 (WritingAssistantTool)
  - [x] SubTask 11.1: 创建 `tools/writing_assistant.py`
  - [x] SubTask 11.2: 实现作文案例获取
  - [x] SubTask 11.3: 实现写作思路建议
  - [x] SubTask 11.4: 实现万能句推荐
  - [x] SubTask 11.5: 实现作文润色和词汇升级建议

## 阶段三：集成与测试

- [x] Task 12: 技能卡与知识卡集成
  - [x] SubTask 12.1: 在技能卡中引用知识卡资源
  - [x] SubTask 12.2: 实现工具与检索器的联动

- [x] Task 13: 测试与验证
  - [x] SubTask 13.1: 测试知识卡文档加载
  - [x] SubTask 13.2: 测试各工具功能
  - [x] SubTask 13.3: 测试完整交互流程

# Task Dependencies
- Task 2-5 依赖于 Task 1（知识卡基础结构）
- Task 7-11 依赖于 Task 6（技能卡基础结构）
- Task 8 依赖于 Task 2（听力内容库）
- Task 9 依赖于 Task 3（长难句库）
- Task 10 依赖于 Task 5（词汇库）
- Task 11 依赖于 Task 4（作文案例库）
- Task 12 依赖于 Task 7-11 和 Task 2-5
- Task 13 依赖于 Task 12
