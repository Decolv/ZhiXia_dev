# RDK STT 极简测试指南 (Human 版)

> 目标：3 步验证录音与识别逻辑，跳过硬件报错。

### 1. 准备环境 (RDK/虚拟机通用)
```bash
source venv/bin/activate
# 修复代理报错 (如果是 socks 代理请执行)
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
unset all_proxy ALL_PROXY socks_proxy SOCKS_PROXY
```

### 2. 模式 A：无麦克风/虚拟机测试 (验证模型)
```bash
# 生成一个 1 秒的测试音频并识别
python3 main/cli_stt_rdk.py --wav tests/audio/test_800hz.wav
```

### 3. 模式 B：RDK 实机录音识别 (验证硬件)
```bash
# 单次录音识别 (默认 4 秒)
python3 main/cli_stt_rdk.py

# 持续录音交互 (按 Enter 开始每一轮)
python3 main/cli_stt_rdk.py --loop
```

---
**常见报错：**
- `PortAudio library not found`: 执行 `sudo apt install libportaudio2`。
- `Unknown scheme for proxy`: 确保环境变量中没有 `socks://` 协议，改用 `http://`。
