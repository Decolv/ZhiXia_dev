# ZhiXia 延迟优化 TODO

## 已完成

- [x] **加强模型预热** — 启动时预热 ASR（加载模型）、LLM（加载+短推理）、TTS（加载+短合成）
  - 文件：`zhixia/__main__.py`
  - 预计收益：500ms~1.5s

- [x] **TTS 不走临时文件** — `PiperEngine.synthesize_to_bytes()` 已用 `io.BytesIO` 直接写内存
  - 文件：`zhixia/tts/piper_engine.py`
  - 无需改动

## 待做（按优先级排序）

### P0: VAD 流式 ASR

- **目标**：把录音+ASR 从串行改成边录边识别
- **方案**：用 VAD（webrtcvad / Silero）检测语音活动，200ms 滑动窗口实时送入 FunASR
- **预计收益**：1~2s（最大收益项）
- **涉及文件**：`zhixia/audio/recorder.py`、`zhixia/asr/funasr_engine.py`、`zhixia/pipeline/orchestrator.py`
- **难点**：当前 FunASR 基于文件输入，需改为流式 API；Snowboy 和 sounddevice 麦克风独占冲突

### P1: 预合成唤醒提示音

- **目标**：把 `我在` 等 TTS 提示音提前合成好保存为文件
- **方案**：启动时预合成 `我在.wav` 到 `assets/`，唤醒后直接播放文件
- **预计收益**：200~500ms
- **涉及文件**：`zhixia/__main__.py`（预热段）、`assets/`、`zhixia/wakeword/wakeword_loop.py`

### P1: 高频回复音频缓存

- **目标**：常见短回复直接播放缓存音频，跳过 LLM+TTS
- **方案**：启动时预合成 `你好`/`在的`/`再见` 等，维护 text→wav bytes 字典，在 TTS 层拦截
- **预计收益**：特定场景 1~3s
- **涉及文件**：`zhixia/tts/piper_engine.py` 或新增 `zhixia/tts/cached_engine.py`

### P2: 播放器进程常驻

- **目标**：避免每次播放创建子进程的开销
- **方案**：尝试让 aplay/paplay 进程常驻
- **预计收益**：50~100ms
- **状态**：已验证 aplay 读到 EOF 会退出，不支持 stdin 复用。收益较小，暂缓。

## 内存占用参考（RK3588, 8GB RAM）

| 模块 | 加载后 RAM |
|------|-----------|
| LLM (RKLLM Qwen3-1.7B w8a8) | ~2.5~3.5 GB |
| ASR (FunASR INT8) | ~400~800 MB |
| TTS (Piper ONNX) | ~150~250 MB |
| 唤醒词 (Snowboy) | ~20~50 MB |
| Python + 依赖 | ~200~400 MB |
| **总计** | **~3.5~5.5 GB** |

---

*最后更新：2026/04/19*
