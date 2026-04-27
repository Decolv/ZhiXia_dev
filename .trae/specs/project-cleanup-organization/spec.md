# 项目代码仓库清理与归纳 Spec

## Why

当前项目存在大量临时测试文件、遗留脚本、重复文档和不规范的目录结构，影响项目可维护性和专业形象。需要系统性地清理、归类和规范化，使项目成为成熟、优秀的开源项目。

## What Changes

- **清理根目录散落文件**：将测试脚本移入 tests/，归档或移除一次性脚本
- **规范化目录结构**：统一 scripts/、tools/、docs/ 等目录职责
- **完善 pyproject.toml**：补充完整的构建配置、可选依赖、入口点
- **更新 .gitignore**：清理无效规则，补充遗漏规则
- **重构主 README**：合并分散文档，提供完整的项目指南
- **清理旧 specs**：归档已完成的 spec，保留活跃 spec
- **移除冗余文档**：整合 docs/ 目录下重复或过时内容
- **完善代码规范**：统一模块导入、类型注解、文档字符串风格

## Impact

- **Affected specs**: 所有已完成 spec 可归档
- **Affected code**: 根目录所有 .py 脚本、docs/ 目录、pyproject.toml、.gitignore、README.md
- **Breaking changes**: 部分脚本路径变更，需更新使用文档

## ADDED Requirements

### Requirement: 目录结构规范化
项目应采用清晰的目录分层，根目录只保留入口脚本和配置文件。

#### Scenario: 根目录整洁
- **WHEN** 查看项目根目录
- **THEN** 只看到 README, pyproject.toml, 核心入口脚本, 配置目录

### Requirement: 测试文件统一管理
所有测试文件应位于 tests/ 目录下。

#### Scenario: 测试文件归位
- **WHEN** 查找测试文件
- **THEN** 所有 test_*.py 文件都在 tests/ 目录中

### Requirement: 构建配置完善
pyproject.toml 应包含完整的构建、依赖、入口点配置。

#### Scenario: 标准安装流程
- **WHEN** 用户执行 pip install -e .
- **THEN** 所有依赖正确安装，zhixia 命令可用

## MODIFIED Requirements

### Requirement: README 文档
README 应合并关键信息，提供完整的安装、使用、开发指南，并引用详细文档而非重复内容。

### Requirement: .gitignore
.gitignore 应只保留有效规则，移除对不存在文件的引用。

## REMOVED Requirements

### Requirement: 根目录临时脚本
**Reason**: test_*.py、validate_refactoring.py 等临时脚本应归入 tests/ 或移除
**Migration**: 有保留价值的脚本移入 tests/ 或 scripts/，无价值的直接删除

### Requirement: 分散的文档文件
**Reason**: REFACTORING_SUMMARY.md、docs/ 下多个报告文件内容重复或过时
**Migration**: 关键信息合并到 README 或 docs/ 下的单一文档
