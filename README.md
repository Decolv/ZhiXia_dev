# ZhiXia - 离线智能语音助手

> 基于RK3588平台的插卡式智能语音助手，支持LLM智能调度与思考过程播报

## 项目简介

ZhiXia（知匣）是一个为RK3588平台设计的离线智能语音助手项目，采用创新的**插卡式Agent架构**，通过技能卡（Skill Card）和知识卡（Knowledge Card）灵活扩展功能。

### 核心特性

- 🧠 **LLM智能调度**：所有工具回答均由大模型临时生成，拒绝FAQ硬编码
- 💭 **思考过程播报**：实时展示AI的思考过程，增加交互透明度
- 🃏 **插卡式架构**：通过Skill Card和Knowledge Card动态扩展能力
- 🔄 **流式流水线**：ASR → LLM → TTS → Play 并发流水线，首句延迟极低
- 📡 **完全离线**：所有模型本地运行，保护隐私无需网络
- ⚡ **NPU加速**：使用RKLLM在NPU上运行大模型，高效低功耗

### 架构优势

| 特性 | 传统方式 | ZhiXia |
|------|---------|--------|
| 工具回答 | 硬编码FAQ，死板 | LLM动态生成，灵活 |
| 交互体验 | 黑盒等待 | 思考过程实时播报 |
| 功能扩展 | 修改代码重新部署 | 插卡即用，动态加载 |
| 响应速度 | 串行处理，慢 | 并发流水线，快 |

## 技术栈

| 组件 | 技术方案 | 特点 |
|------|---------|------|
| ASR | FunASR / Whisper | 中文识别，支持离线 |
| LLM | RKLLM (Qwen3-1.7B) / Cloud LLM | NPU加速或云端调用 |
| TTS | Piper | 超高速，模型小(42MB) |
| Agent | ReAct / ToolCalling | 结构化工具调用 |
| RAG | ChromaDB | 可选的检索增强生成 |
| WakeWord | Snowboy | 低功耗语音唤醒 |

## 快速开始

### 环境要求

- Python 3.9+
- RK3588开发板（或兼容设备）
- 音频输入输出设备

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/zhixia.git
cd zhixia

# 安装基础依赖
pip install -e .

# 安装Piper TTS
bash install_fast_tts.sh
```

### 运行

```bash
# 方式1: 使用启动脚本
bash run.sh

# 方式2: 直接运行
python -m zhixia

# 方式3: 使用插卡式Agent
python mount_cards.py
```

## 项目架构

### 核心模块

```
zhixia/
├── core/                  # 核心编排
│   ├── host_orchestrator.py   # 主机编排器（插卡式Agent核心）
│   ├── card_base.py           # 卡片基类
│   └── card_loader.py         # 卡片加载器
├── agent/                 # Agent系统
│   ├── react_agent.py         # ReAct Agent
│   ├── tool_agent.py          # ToolCalling Agent
│   ├── callbacks.py           # 回调系统（支持思考播报）
│   └── state.py               # Agent状态管理
├── llm/                   # 大语言模型
│   ├── rkllm_engine.py        # RK3588 NPU引擎
│   ├── cloud_engine.py        # 云端LLM引擎
│   └── rag/                   # RAG检索增强
├── asr/                   # 语音识别
│   ├── funasr_engine.py       # FunASR引擎
│   └── whisper_engine.py      # Whisper引擎
├── tts/                   # 语音合成
│   └── piper_engine.py        # Piper TTS引擎
├── display/               # 显示输出
│   ├── base.py                # 显示抽象基类（支持图片）
│   └── null_display.py        # 空显示实现
├── audio/                 # 音频处理
│   ├── player.py              # 音频播放
│   └── recorder.py            # 音频录制
└── config/                # 配置管理
    └── settings.py            # 应用配置
```

### 插卡式架构

```
skills/                    # 技能卡
└── hnu_freshman/          # 湖南大学新生助手
    ├── card.py                # 技能卡入口
    ├── tools/                 # 工具集
    │   ├── campus_navigate.py     # 校园导航（LLM生成）
    │   ├── life_guide.py          # 生活指南（LLM生成）
    │   └── major_query.py         # 专业查询（LLM生成）
    ├── assets/              # 图片资源
    │   ├── maps/                # 地图图片
    │   └── photos/              # 实景照片
    ├── persona.json         # 人设配置
    └── manifest.json        # 卡片清单

knowledge/                 # 知识卡
└── hnu_campus/            # 湖南大学校园知识
    ├── card.py                  # 知识卡入口
    └── docs/                    # 知识文档
```

## 核心功能

### 1. LLM智能工具调度

所有工具回答均由大模型基于知识上下文动态生成，告别死板的FAQ：

```python
# 工具使用LLM生成回答
class CampusLifeGuideTool(Tool):
    CAMPUS_KNOWLEDGE = "..."  # 知识作为上下文
    
    def _guide(self, query: str) -> str:
        # 使用LLM基于知识生成个性化回答
        return self._generate_with_llm(query)
```

### 2. 思考过程播报

通过回调系统实时展示AI的思考过程：

```python
class DisplayCallbackHandler(BaseCallbackHandler):
    def on_thinking_start(self, run_id: str, **kwargs):
        self.display.update_thinking(True, "正在思考...")
    
    def on_agent_thought(self, run_id: str, thought: str, **kwargs):
        self.display.show(DisplayPayload(
            text="", emotion="thinking",
            is_thinking=True, thinking_text=thought
        ))
```

### 3. 流式并发流水线

```
Thread-A (LLM)  → 分句 → tts_queue
Thread-B (TTS)  → 合成 → play_queue  
Thread-C (Play) → 播放
```

首句延迟 = ASR + LLM首token + TTS首句合成

### 4. 图片资源支持

导航工具支持地图图片和实景照片展示：

```python
class CampusNavigateTool(Tool):
    def get_location_images(self, location_name: str):
        return map_image, photos  # 返回图片路径
```

## 配置说明

### 本地配置 (`localconfig/localconfig.json`)

```json
{
  "llm": {
    "type": "rkllm",
    "model_path": "models/Qwen3-1.7B.rkllm",
    "max_new_tokens": 256,
    "system_prompt": "你是智能助手小匣..."
  },
  "asr": {
    "type": "funasr",
    "model_path": "models/funasr"
  },
  "tts": {
    "type": "piper",
    "model_path": "models/piper"
  }
}
```

## 开发指南

### 添加新工具

```python
from zhixia.agent.tool import Tool

class MyTool(Tool):
    def __init__(self, llm_engine=None):
        super().__init__(
            name="my_tool",
            description="工具描述",
            func=self._execute,
        )
        self._llm_engine = llm_engine
    
    def _execute(self, query: str) -> str:
        # 使用LLM生成回答
        return self._generate_with_llm(query)
```

### 添加新技能卡

```python
from zhixia.core.card_base import SkillCard, HostContext

class MySkill(SkillCard):
    def on_mount(self, host: HostContext) -> None:
        host.tool_registry.register(MyTool())
        host.persona_holder.set_overlay("人设内容", self.name)
    
    def on_unmount(self, host: HostContext) -> None:
        host.tool_registry.unregister("my_tool")
        host.persona_holder.clear_overlay()
```

## 性能表现

在RK3588平台上的典型性能：

| 指标 | 数值 |
|------|------|
| ASR识别 | 1-2秒 |
| LLM首token | 0.5-1秒 |
| TTS首句合成 | 0.3-0.5秒 |
| 首句播放延迟 | 2-4秒 |
| 并发效率 | 70-85% |

## 故障排除

### 常见问题

1. **TTS合成失败**：确保Piper模型已正确安装
2. **LLM响应慢**：检查NPU是否正常工作
3. **卡片未加载**：检查卡片目录结构和manifest.json

### 日志调试

```bash
# 开启详细日志
export LOG_LEVEL=DEBUG
python -m zhixia
```

## 许可证

本项目采用MIT许可证。

### 依赖项目许可

- Piper TTS: MIT License
- FunASR: Apache-2.0
- RKLLM: 参考瑞芯微官方许可
- ChromaDB: Apache-2.0

## 致谢

- Rhasspy团队 - Piper TTS
- 阿里巴巴达摩院 - FunASR
- 瑞芯微 - RKLLM SDK
- ModelScope社区

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交Issue或查看详细文档。
