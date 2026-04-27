# 项目清理与归纳任务清单

## 任务列表

- [x] 任务1：清理根目录散落文件
  - [x] 1.1 将根目录 test_*.py 移至 tests/ 目录
  - [x] 1.2 将 validate_refactoring.py 移至 tests/ 目录
  - [x] 1.3 将 rkllm_inference.py 移入 zhixia/llm/ 包内
  - [x] 1.4 将 asr_llm_tts_piper.py 移入 scripts/ 或标记为废弃
  - [x] 1.5 将 mount_cards.py 移入 scripts/ 目录
  - [x] 1.6 将 install_fast_tts.sh 移入 scripts/ 目录
  - [x] 1.7 创建 scripts/ 目录并编写 README 说明用途

- [x] 任务2：清理 main/ 目录
  - [x] 2.1 检查 main/ 目录是否仍在项目中被引用
  - [x] 2.2 若未被引用，移除 main/ 目录（属于旧架构遗留）
  - [x] 2.3 若仍被引用，移入 zhixia/ 包内或 scripts/

- [x] 任务3：整合与清理文档
  - [x] 3.1 将 docs/foragent.md 内容整合到 CLAUDE.md
  - [x] 3.2 将 docs/USAGE.md 关键内容合并到主 README
  - [x] 3.3 将 docs/REVIEW.md 和 docs/CODE_REVIEW_REPORT.md 归档为开发历史
  - [x] 3.4 将 docs/VIBE_CODING_ERRORS.md 归档或移除
  - [x] 3.5 将 docs/PRIORITY_CLASSIFICATION.md 归档或移除
  - [x] 3.6 将 REFACTORING_SUMMARY.md 内容精简后归档
  - [x] 3.7 清理 .kilo/ 目录（AI 工具内部计划文件）

- [x] 任务4：完善 pyproject.toml
  - [x] 4.1 添加完整的 metadata（authors, license, readme, keywords）
  - [x] 4.2 添加 [project.scripts] 入口点（zhixia 命令）
  - [x] 4.3 添加 [project.urls]（Homepage, Documentation, Repository）
  - [x] 4.4 添加 [tool.ruff] 配置（从项目实践中提取）
  - [x] 4.5 添加 [tool.pytest] 配置
  - [x] 4.6 添加 [build-system] 完整配置

- [x] 任务5：更新 .gitignore
  - [x] 5.1 移除对不存在文件的规则（install_dependencies.sh, setup_rknpu.sh, run_fast_tts.sh, run_npu_only.sh）
  - [x] 5.2 添加 IDE 特定忽略（.vscode/, .idea/, *.swp, .DS_Store）
  - [x] 5.3 添加 Python 标准忽略（.mypy_cache/, *.egg-info/, dist/, build/）
  - [x] 5.4 添加 AI 工具忽略（.claude/, .kilo/）
  - [x] 5.5 规范化格式和分组注释

- [x] 任务6：重构主 README.md
  - [x] 6.1 合并安装指南（整合 docs/README_PIPER.md 内容）
  - [x] 6.2 添加完整的配置说明章节
  - [x] 6.3 添加开发指南（代码规范、测试、构建）
  - [x] 6.4 添加贡献指南
  - [x] 6.5 添加目录索引到 docs/ 下保留的文档
  - [x] 6.6 移除冗余代码示例，保持精简专业

- [x] 任务7：归档已完成的 specs
  - [x] 7.1 检查所有 specs 的完成状态
  - [x] 7.2 将已完成的 specs 标记为 archived
  - [x] 7.3 保持活跃 specs 可继续推进

- [x] 任务8：规范化 zhixia/ 包
  - [x] 8.1 检查并补全所有 __init__.py 的公共 API 导出
  - [x] 8.2 检查 display/ 目录是否有缺失的 __init__.py 导出
  - [x] 8.3 统一模块文档字符串格式
  - [x] 8.4 检查是否有未使用的导入

- [x] 任务9：完善 tests/ 目录
  - [x] 9.1 合并所有测试文件到 tests/
  - [x] 9.2 创建 tests/conftest.py（共享 fixture）
  - [x] 9.3 更新测试文件导入路径
  - [x] 9.4 添加 pytest.ini 或配置 pyproject.toml

- [x] 任务10：验证与测试
  - [x] 10.1 运行 ruff check 确保代码规范
  - [x] 10.2 验证 python -m zhixia 仍可正常运行
  - [x] 10.3 验证所有导入路径正确
  - [x] 10.4 运行现有测试确认功能正常

## 任务依赖关系

- 任务1 无依赖（可并行）
- 任务2 无依赖（可并行）
- 任务3 无依赖（可并行）
- 任务4 无依赖（可并行）
- 任务5 无依赖（可并行）
- 任务6 依赖于 任务3（需要先完成文档整合）
- 任务7 无依赖（可并行）
- 任务8 依赖于 任务1（需要先完成文件移动）
- 任务9 依赖于 任务1（测试文件需要先移动）
- 任务10 依赖于 所有其他任务
