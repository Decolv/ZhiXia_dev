# 英语考试辅导助手技能卡与知识卡 Spec

## Why
用户需要一个专业的英语考试辅导助手，能够针对英语考试（如四六级、雅思、托福等）提供全方位的备考支持。通过技能卡和知识卡的结合，实现智能化的考试规划、听力训练、长难句解析、词汇记忆和作文辅导功能。

## What Changes
- **新增**: 英语考试辅导技能卡 `english_tutor_skill`，包含5个核心工具
- **新增**: 英语考试知识卡 `english_tutor_knowledge`，存储听力材料、长难句、作文案例等
- **新增**: 技能卡人设文件 `persona.json`，定义英语考试辅导助手角色
- **新增**: 知识卡文档结构，分类存储各类英语学习资料

## Impact
- Affected specs: 技能卡架构、知识卡架构、Agent system prompt 构建
- Affected code:
  - `skills/english_tutor_skill/` - 英语辅导技能卡目录
  - `skills/english_tutor_knowledge/` - 英语辅导知识卡目录
  - `zhixia/core/host_orchestrator.py` - 可能需要的协调器更新

## ADDED Requirements

### Requirement: 考试准备计划器 (ExamPlannerTool)
系统 SHALL 提供考试准备计划器工具，帮助用户智能安排备考计划。

#### Scenario: 首次创建计划
- **GIVEN** 用户告知考试类型和截止日期
- **WHEN** 用户请求创建备考计划
- **THEN** 系统记录用户画像（当前水平、目标分数、可用时间）
- **AND** 构建多维度薄弱点分析（单词、长难句、作文、听力）
- **AND** 根据考试截止日期和当前日期生成排期规划

#### Scenario: 查看和调整计划
- **GIVEN** 用户已有备考计划
- **WHEN** 用户请求查看或调整计划
- **THEN** 系统展示当前计划进度
- **AND** 允许用户调整优先级或时间安排

### Requirement: 听力辅助器 (ListeningAssistantTool)
系统 SHALL 提供听力辅助工具，帮助用户进行听力训练。

#### Scenario: 播放听力材料
- **GIVEN** 知识卡中存储了多种听力内容及中文翻译
- **WHEN** 用户请求进行听力练习
- **THEN** 系统从知识卡获取听力内容
- **AND** 智能交互播放听力音频（文本模拟或TTS）

#### Scenario: 听力理解测试
- **GIVEN** 用户已听完一段材料
- **WHEN** 用户提交对听力内容的中文含义推测
- **THEN** 系统对照正确翻译告知用户理解准确度
- **AND** 提供针对性改进建议

### Requirement: 长难句助力器 (LongSentenceTool)
系统 SHALL 提供长难句解析工具，帮助用户理解复杂句子结构。

#### Scenario: 长难句学习
- **GIVEN** 知识卡中存储了外刊长难句及中文翻译和语法结构知识点
- **WHEN** 用户请求学习长难句
- **THEN** 系统从知识卡获取长难句内容
- **AND** 展示原句、中文翻译和语法结构分析
- **AND** 与用户进行交互式学习

#### Scenario: 语法知识点讲解
- **GIVEN** 用户在学习长难句过程中
- **WHEN** 用户对某个语法点有疑问
- **THEN** 系统详细解释相关语法知识
- **AND** 提供类似例句帮助理解

### Requirement: 词汇复习器 (VocabularyReviewerTool)
系统 SHALL 提供词汇复习工具，参照先进记忆软件设计逻辑。

#### Scenario: 制定词汇计划
- **WHEN** 用户开始词汇学习
- **THEN** 系统根据考试要求制定词汇记忆计划
- **AND** 采用滚动复习机制安排学习节奏

#### Scenario: 词汇检测
- **GIVEN** 用户已学习一段时间词汇
- **WHEN** 系统进行定期检测
- **THEN** 汇总用户掌握情况
- **AND** 给予鼓励或改进建议

#### Scenario: 智能复习提醒
- **GIVEN** 用户有即将遗忘的词汇
- **WHEN** 系统根据记忆曲线判断
- **THEN** 主动提醒用户复习相关词汇

### Requirement: 作文辅导器 (WritingAssistantTool)
系统 SHALL 提供作文辅导工具，帮助用户提升写作能力。

#### Scenario: 获取作文案例
- **GIVEN** 知识卡内存储了优秀作文案例和往期作文题材
- **WHEN** 用户请求查看作文案例
- **THEN** 系统展示相关题材的优秀范文

#### Scenario: 写作思路建议
- **GIVEN** 用户面对特定作文题材
- **WHEN** 用户请求写作思路
- **THEN** 系统针对不同题材提供思路建议
- **AND** 从知识卡获取万能句进行推荐

#### Scenario: 作文润色升级
- **GIVEN** 用户完成作文初稿
- **WHEN** 用户请求作文批改
- **THEN** 系统智能研判作文中可升级的词汇
- **AND** 提供加分词汇替换建议

### Requirement: 英语考试知识卡数据结构
系统 SHALL 在知识卡中维护以下数据结构：

#### 听力内容 (listening/)
```
listening/
├── cet4/
│   ├── passage_01.md
│   └── passage_02.md
├── cet6/
│   ├── passage_01.md
│   └── passage_02.md
└── ielts/
    └── section_01.md
```
每个听力文件包含：原文、中文翻译、难度等级、话题标签

#### 长难句库 (sentences/)
```
sentences/
├── by_source/
│   ├── economist.md
│   ├── nytimes.md
│   └── scientific_american.md
└── by_difficulty/
    ├── beginner.md
    ├── intermediate.md
    └── advanced.md
```
每个长难句包含：原句、中文翻译、语法结构分析、重点词汇

#### 作文案例 (writing/)
```
writing/
├── templates/
│   ├── argumentation.md
│   ├── narration.md
│   └── exposition.md
├── examples/
│   ├── cet4/
│   ├── cet6/
│   └── ielts/
└── universal_sentences.md
```

#### 词汇库 (vocabulary/)
```
vocabulary/
├── cet4_core.md
├── cet6_core.md
├── ielts_academic.md
└── collocations.md
```

## MODIFIED Requirements
无修改需求。

## REMOVED Requirements
无移除需求。
