# 项目清理与归纳检查清单

- [x] 根目录无散落的 test_*.py 文件
- [x] 根目录无 validate_refactoring.py、rkllm_inference.py 等应归位的文件
- [x] 所有脚本文件位于 scripts/ 或对应包内
- [x] main/ 目录已处理（移除或整合）
- [x] .kilo/ 和 .claude/ 目录已清理
- [x] docs/ 目录无重复或过时文档
- [x] pyproject.toml 包含完整 metadata 和入口点
- [x] .gitignore 规则有效且无引用不存在文件的条目
- [x] README.md 包含完整的安装、使用、开发指南
- [x] README.md 无冗余代码示例
- [x] 所有 tests/ 文件可正常导入
- [x] tests/conftest.py 存在并提供共享 fixture
- [x] zhixia/ 包所有 __init__.py 正确导出公共 API
- [x] python -m zhixia 可正常启动
- [x] ruff check 通过无错误
- [x] 所有模块导入路径正确无循环依赖
- [x] 已完成的 specs 已归档
- [x] CLAUDE.md 与项目结构一致
