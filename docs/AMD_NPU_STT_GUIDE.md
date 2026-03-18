# AMD NPU (RyzenAI) STT 部署说明

本文档介绍如何在具有 AMD NPU (XDNA) 的设备上运行 Whisper STT 加速。

## 🛠️ 环境准备

1. **硬件要求**：
   - AMD Ryzen™ 7040 / 8040 系列处理器（带 NPU）或更高。
   - 确保 BIOS 中已启用 NPU。

2. **驱动安装**：
   - 安装最新的 IPU (NPU) 驱动。
   - 安装 [Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/inst.html)。

3. **依赖安装**：
   ```bash
   pip install -r requirements-amd-npu.txt
   ```
   核心依赖是 `onnxruntime-vitisai`，它提供了对接 AMD NPU 的 Execution Provider。

## 📦 模型准备

AMD NPU 不直接运行普通的 Hugging Face 模型，需要进行量化和编译（针对 IPU 架构）。
推荐使用 [ryzen-ai-sw](https://github.com/amd/ryzen-ai-sw) 社区提供的预编译模型，或使用 `vaip` 编译器转换模型。

1. 下载适配 NPU 的 Whisper ONNX 模型。
2. 将模型路径配置在 `config-amd-npu-stt.yaml` 的 `model_path` 中。

## 🏃 运行识别

使用专用配置文件启动：

```bash
python main/cli_stt_ryzenai.py --config config-amd-npu-stt.yaml
```

## ⚠️ 注意事项

- **Vitis AI EP**：确保 `VitisAIExecutionProvider` 在 `onnxruntime.get_available_providers()` 中可见。
- **内存限制**：NPU 推理通常需要特定的对齐和预分配内存。
