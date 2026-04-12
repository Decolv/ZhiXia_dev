# RKLLM NPU加速语音助手

## 概述

本项目实现了基于RK3588 NPU的ASR + LLM + TTS完整语音助手流程。

## 模型文件

已提供的模型文件：
- **Qwen3-1.7B-w8a8-rk3588.rkllm** (2.3GB) - 位于 `/home/quark/code/models/`

## 当前状态

### ⚠️ 重要提示

当前系统**缺少RKNPU驱动**，无法使用NPU加速。

**错误信息：**
```
E RKNN: failed to open rknpu module, need to insmod rknpu driver!
E RKNN: failed to open rknn device!
E RKNN: Device is not available
```

## 解决方案

### 方案1：安装RKNPU驱动（推荐）

需要系统管理员安装RKNPU内核驱动：

```bash
# 1. 检查是否有驱动模块
find /lib/modules/$(uname -r) -name "*rknpu*"

# 2. 如果没有，需要编译安装驱动
# 参考 Rockchip 官方文档:
# https://github.com/airockchip/rknn-llm

# 3. 加载驱动
sudo modprobe rknpu

# 4. 验证
ls /dev/rknpu*
cat /sys/kernel/debug/rknpu/version
```

### 方案2：使用CPU推理（当前可用）

当NPU不可用时，程序会自动回退到CPU推理：

```bash
# 运行CPU版本
./run_voice_assistant.sh
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `rkllm_inference.py` | RKLLM Python绑定，支持Qwen2/Qwen3 |
| `asr_llm_tts_rknn.py` | NPU版本主程序 |
| `asr_llm_tts.py` | CPU版本主程序 |
| `run_voice_assistant_rknn.sh` | NPU版本启动脚本 |
| `run_voice_assistant.sh` | CPU版本启动脚本 |

## 性能对比

| 推理方式 | 速度 | 状态 |
|---------|------|------|
| RKNN NPU | ~15 token/s | ❌ 需要驱动 |
| CPU | ~1-2 token/s | ✅ 可用 |

## 使用说明

### 测试RKLLM（需要驱动）

```bash
export LD_LIBRARY_PATH="/home/quark/code/rknn_libs:${LD_LIBRARY_PATH}"
python3 rkllm_inference.py
```

### 运行完整流程（CPU回退）

```bash
./run_voice_assistant.sh
```

## 模型支持

当前代码支持：
- ✅ Qwen3-1.7B (RKLLM格式)
- ✅ Qwen2.5-1.5B (RKLLM格式)
- ✅ Qwen2.5-1.5B (HuggingFace格式，CPU)

## 待完成工作

1. [ ] 安装RKNPU内核驱动
2. [ ] 验证NPU推理功能
3. [ ] 测试完整ASR+LLM+TTS流程
4. [ ] 性能优化和调参

## 参考链接

- [RKNN-LLM GitHub](https://github.com/airockchip/rknn-llm)
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [Rockchip NPU文档](https://docs.radxa.com/rock5/rock5b/app-development/ai/rkllm-usage)
