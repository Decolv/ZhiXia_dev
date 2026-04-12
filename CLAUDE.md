# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZhiXia (智侠) — a Chinese voice assistant running on RK3588 (Rockchip) NPU hardware. Three-stage pipeline: ASR (speech-to-text) → LLM (AI response) → TTS (text-to-speech).

**Target platform:** RK3588 SBC, user `quark` on `/home/quark/`. All scripts auto-configure `LD_LIBRARY_PATH`, `PYTHONPATH`, and `MODELSCOPE_CACHE` relative to their own directory. Many hardcoded paths reference `/home/quark/` — these are the board paths, not development machine paths.

## Architecture

### Pipeline scripts (root level)

Two variants of the full ASR→LLM→TTS pipeline:
- `asr_llm_tts_npu_fast.py` — speed-optimized: MeloTTS (offline). Targets <2s TTS synthesis.
- `asr_llm_tts_npu_only.py` — offline-optimized: PaddleSpeech (offline). No network required.

Both share the same structure: `asr_recognition_int8()` → `llm_inference_npu_stream()` → TTS synthesis → `play_audio()`. Each stage explicitly frees models with `del` + `gc.collect()` to fit in ~3GB RAM.

### Core NPU module: `rkllm_inference.py`

Python ctypes wrapper around `librkllmrt.so` (Rockchip's NPU inference library). Key components:
- **`RKLLM` class** — handles model init, `generate()`, `chat()`, and Qwen2/Qwen3 chat templates
- **`RKLLMConfig` dataclass** — all inference params (temperature, top_p, max_tokens, etc.)
- **`create_rkllm_from_hf()`** — factory function that auto-detects model type from filename
- Supports Qwen3 "thinking" mode via `enable_thinking` flag

Model type is detected from filename: files containing "qwen3" use Qwen3 chat template (`<|im_start|>` format), others default to Qwen2 format (`<|system|>` / `<|user|>` format).

### Model conversion: `convert_to_rkllm.py`

Generates quantization calibration data and an export script. Designed to run on x86 Linux (needs `rkllm-toolkit`), then transfer `.rkllm` files to the RK3588 board. Targets W8A8 quantization for rk3588 platform.

### Standalone STT module: `main/`

Separate CLI for speech-to-text only, using faster-whisper (not FunASR):
- `main/cli_stt_rdk.py` — CLI entry point, reads `config-edge-rdk-stt.yaml` (not in repo)
- `main/services/stt.py` — `FasterWhisperSTT` class wrapping `faster_whisper.WhisperModel`
- `main/services/audio_recorder.py` — `sounddevice`-based mic recording

### ASR models

The pipeline scripts use FunASR with Paraformer (`speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1`), INT8 quantized. The `main/` module uses faster-whisper instead — two independent STT approaches.

## Running

No formal build system. Scripts are run directly on the RK3588 board:

```bash
# Full pipeline (fast TTS variant)
python3 asr_llm_tts_npu_fast.py

# Full pipeline (offline TTS variant)
python3 asr_llm_tts_npu_only.py

# Test RKLLM inference standalone
python3 test_rkllm.py

# Convert HuggingFace model to RKLLM format (run on x86)
python3 convert_to_rkllm.py --model-path /path/to/Qwen2.5-1.5B-Instruct

# Standalone STT CLI
cd main && python3 cli_stt_rdk.py --config config-edge-rdk-stt.yaml
```

## Dependencies (no requirements.txt)

Runtime deps: Python 3.9, PyTorch 2.8.0 (CPU), FunASR 1.3.1, ChatTTS, MeloTTS, PaddleSpeech, pyttsx3, faster-whisper, sounddevice, PyYAML. Model downloads via ModelScope.

Native libs in `rknn_libs/` (gitignored): `librkllmrt.so`, `librknnrt.so`, `rkllm.h`.

## Directory layout (gitignored, must be set up on board)

- `models/` — `.rkllm` model files (~2.3GB each)
- `asset/` — ChatTTS model files (Decoder, DVAE, Embed, Vocos safetensors)
- `rknn_libs/` — RKNN runtime libraries and drivers
- `output/` — generated audio files

## Key constraints

- **RAM:** Minimum 3GB available. Scripts call `force_gc()` between pipeline stages to reclaim memory.
- **NPU driver:** Requires `rknpu` kernel module loaded. Without it, RKLLM init fails with "failed to open rknpu module".
- **Audio:** Uses ALSA (`aplay`/`paplay`/`ffplay` for playback). Input must be 16kHz WAV for the FunASR models.
- **No tests beyond `test_rkllm.py`:** There is no test suite. The only test is a manual RKLLM inference smoke test.
