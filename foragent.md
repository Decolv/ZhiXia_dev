# 目录放置位置声明

## 模型文件放置位置

### 1. models/ 目录
- **用途**: 存放 RKLLM 和 Piper 模型文件
- **位置**: 项目根目录下的 `models/` 文件夹
- **RKLLM 模型**: `models/Qwen3-1.7B-w8a8-rk3588.rkllm`（约2.2GB）
  - 从 RKLLM 官方渠道下载或通过 `convert_to_rkllm.py` 转换
- **Piper 模型**: `models/piper/zh_CN-huayan-medium.onnx`（约42MB）
  - 从 HuggingFace 下载：https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/
- **模型下载脚本**: 运行 `scripts/install_fast_tts.sh` 自动下载 Piper 模型

### 2. rknn_libs/ 目录
- **用途**: 存放 RKNN 运行时库和工具
- **位置**: 项目根目录下的 `rknn_libs/` 文件夹
- **包含文件**:
  - librkllmrt.so (RKLLM 运行时库)
  - librknnrt.so (RKNN 运行时库)
  - rkllm.h (头文件)
- **获取方式**: 从瑞芯微官方网站下载 RKNN SDK

## 配置说明

### 配置文件位置
- **主配置**: `localconfig/localconfig.json`
  - 重构后使用分层加载（代码默认值 + JSON 用户覆盖）
  - 新增配置：RAG（检索增强）、结构化输出、Display 接口
- **向后兼容**: 原有配置文件无需修改即可继续使用

### 环境变量
- **MODELSCOPE_CACHE**: ModelScope 模型缓存目录（`.cache/modelscope`）
- **PYTHONPATH**: Python 模块路径（包含 `.local/lib/python3.9/site-packages`）
- **LD_LIBRARY_PATH**: RKNN 库路径（`rknn_libs`）

## 重构后目录结构
```
ZhiXia_dev/
├── zhixia/                          # 主包（新增）
│   ├── config/                     # 配置管理
│   ├── pipeline/                    # 管线编排
│   ├── asr/                         # ASR 引擎（FunASR/Whisper）
│   ├── llm/                         # LLM 引擎 + RAG + 输出解析
│   ├── tts/                         # TTS 引擎（Piper）
│   ├── audio/                       # 音频 I/O（播放/录音）
│   ├── display/                     # 显示接口（预留）
│   └── utils/                       # 工具（日志/内存）
├── tests/                           # 测试目录（新增）
│   └── test_pipeline_stages.ipynb   # 分阶段测试
├── asr_llm_tts_piper.py             # 改为薄 shim（向后兼容）
├── run.sh                           # 更新调用新入口
└── pyproject.toml                   # 项目元数据（新增）
```

## 运行脚本

### 启动方式
1. **重构后（推荐）**: `python -m zhixia` 或 `./run.sh`
2. **向后兼容**: `python asr_llm_tts_piper.py`（薄 shim 调用新版本）

### 安装脚本
- **安装_fast_tts.sh**: 自动下载 Piper TTS 和模型

## 新功能特性

### RAG 检索增强（预留接口）
- 通过 `localconfig.json` 中 `rag.enabled` 启用
- 抽象接口 `RAGRetriever`，支持多种检索后端
- 默认 `NullRAGRetriever`（无检索）

### 结构化输出（情绪/元数据）
- 通过 `llm.enable_structured_output` 启用
- 解析器支持 JSON 和 `[emotion:xxx]` 前缀约定
- 输出包含：文本（TTS）、情绪（Display）、元数据（扩展）

### 模块化设计
- 每个组件有抽象基类（ABC）和具体实现
- 工厂模式创建引擎，支持切换
- 模型懒加载，启动零开销，峰值占用 ~3.1GB（8GB 系统）

## 注意事项

- 重构后保持完全向后兼容
- 模型文件下载路径不变（在 `.gitignore` 中配置为忽略）
- 新增测试文件在 `tests/` 目录，分阶段验证各模块
- 首次运行仍需下载模型文件
- 确保有足够磁盘空间（至少 3GB）
- RK3588 上需要 RKNN 驱动