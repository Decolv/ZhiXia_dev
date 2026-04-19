# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZhiXia (知匣) is an offline voice assistant for RK3588 embedded devices. It runs a complete speech interaction pipeline: **ASR → LLM → TTS → Audio Output**, fully on-device with NPU acceleration.

- **Primary language**: Python 3.9+
- **Package name**: `zhixia`
- **Entry point**: `python -m zhixia`
- **Target hardware**: QuarkPi (RK3588) with NPU; PC dev supported via fallback mocks

## Common Commands

### Run the application

```bash
# RK3588 (production)
python -m zhixia
# or
bash run.sh

# PC development (uses fake LLM fallback, no NPU needed)
$env:ZHIXIA_CONFIG="localconfig/localconfig.pc.json"        # PowerShell
$env:ZHIXIA_ALLOW_FAKE_LLM="1"
python -m zhixia
```

### Install dependencies

```bash
# Base + ASR (FunASR)
pip install -e ".[asr-funasr,dev]"

# Or ASR (Whisper)
pip install -e ".[asr-whisper,dev]"

# TTS (Piper)
pip install -e ".[tts-piper,dev]"
```

### Test individual modules

```bash
# Run the Jupyter notebook for staged testing
jupyter notebook tests/test_pipeline_stages.ipynb

# Validate refactoring integrity
python validate_refactoring.py
```

There is no traditional pytest test suite; testing is done via the Jupyter notebook or manual module-level scripts.

### Code quality

```bash
# The project uses ruff (evidenced by .ruff_cache)
ruff check zhixia/
ruff format zhixia/
```

## Architecture

### Module Structure

```
zhixia/
├── __main__.py          # Entry point: factory functions, model warmup, config load
├── config/settings.py   # Layered config: dataclass defaults + JSON file override
├── pipeline/orchestrator.py  # VoicePipeline: 3-thread streaming pipeline
├── asr/                 # ASREngine ABC; FunASR (default) & Whisper implementations
├── llm/                 # LLMEngine ABC; RKLLM wrapper + fallback LLM for PC dev
│   ├── output_parser.py # Parses structured output (JSON/emotion prefix)
│   └── rag/             # RAGRetriever ABC; default NullRAGRetriever
├── tts/                 # TTSEngine ABC; Piper implementation
├── audio/               # AudioPlayer ABC (ALSA/pipe); AudioRecorder (sounddevice)
├── display/             # DisplayOutput ABC; default NullDisplay
└── utils/               # Logging setup, memory helpers
```

### Pipeline Flow (3-Thread Concurrent)

`VoicePipeline.process_audio()` in `pipeline/orchestrator.py` orchestrates:

1. **ASR** (main thread): Transcribes input WAV to text
2. **LLM Worker** (thread): Streams tokens via `llm_engine.stream_chat()`. For structured output, incrementally parses JSON to extract `emotion` for display and `text` chunks for TTS.
3. **TTS Worker** (thread): Consumes text chunks from a queue, synthesizes to WAV bytes in memory via `tts_engine.synthesize_to_bytes()`, pushes to play queue.
4. **Play Worker** (thread): Consumes WAV bytes from queue, plays via `audio_player.play_bytes()` piped to `aplay`/`ffplay` stdin.

This concurrent design minimizes time-to-first-audio: LLM, TTS, and playback overlap rather than running sequentially.

### Key Design Patterns

- **Abstract Base Classes**: Every engine (ASR, LLM, TTS, AudioPlayer, DisplayOutput, RAGRetriever) defines an ABC in `*/base.py`. New implementations must subclass these.
- **Lazy Loading**: Models are loaded on first use (`_ensure_model()` / `_ensure_voice()`), not at import time.
- **Factory Functions**: `__main__.py` contains `create_*_engine()` functions that select implementations based on config.
- **Layered Configuration**: `AppSettings.load()` merges dataclass defaults with a JSON file (`localconfig/localconfig.json`). Override via `ZHIXIA_CONFIG` env var.
- **PC Fallback**: `RKLLMEngine` falls back to `_FallbackLLM` (fixed text, simulated streaming) when `ZHIXIA_ALLOW_FAKE_LLM=1` is set, enabling development without the RK3588 NPU runtime.

### External Dependencies

- **RKLLM**: `rkllm_inference.py` (root-level, ctypes wrapper around `librkllmrt.so`). `RKLLMEngine` imports this at runtime and adds it to `sys.path`.
- **FunASR/Whisper**: ASR backends, chosen via config `asr.engine`.
- **Piper**: TTS backend. Models auto-download from HuggingFace if missing.
- **sounddevice**: Audio recording (lazy-imported).

### Model Files (not in repo)

- `models/Qwen3-1.7B-w8a8-rk3588.rkllm` (~2.2GB) — LLM model for RK3588 NPU
- `models/piper/zh_CN-huayan-medium.onnx` (~42MB) — TTS voice model
- `.cache/modelscope/` — FunASR model cache (set via `MODELSCOPE_CACHE` env)

### Config Files

- `localconfig/localconfig.json` — Production (RK3588) defaults
- `localconfig/localconfig.pc.json` — PC development config (points to `sample.wav`, disables NPU)
- `localconfig/localconfig.linux.json` — Linux variant

### Important Code Conventions

- **Type hints**: Used throughout (`list[str]`, `Optional[...]`, `Generator[str, None, None]`).
- **Logging**: Use `logging.getLogger(__name__)` in all modules. Log level controlled by config `log_level`.
- **Streaming**: LLM engines must implement `stream_chat()` to yield tokens. The base class provides a fallback that yields the full string at once.
- **Memory synthesis**: TTS engines should override `synthesize_to_bytes()` to avoid disk I/O. The base class provides a temp-file fallback.
- **Structured output parsing**: `output_parser.py` handles Qwen3 `<think>...</think>` stripping, JSON extraction (including partial/streaming JSON), and `[emotion:xxx]` prefix fallback.

## Platform Notes

- **RK3588**: Requires `librkllmrt.so` in `rknn_libs/` or system path. Audio playback uses ALSA (`aplay`).
- **PC/Windows**: No NPU library; use `ZHIXIA_ALLOW_FAKE_LLM=1` to mock LLM. Audio playback falls back to whatever player is available (`aplay`, `paplay`, `ffplay`).
- **Python environments**: The repo contains `.venv` and `.venv310`. `run_pc.ps1` auto-detects a venv with `funasr` installed.

## Agent Usage Rules

When processing user requests, prefer the `Agent` tool over serial tool calls in these situations:

1. **Codebase exploration / search**: If more than 3 sequential `Glob`/`Grep`/`Read` calls are needed, **must** use `Agent` with `subagent_type=Explore` for parallel search.
2. **Multi-module research**: When investigating multiple independent modules, launch multiple `Agent` calls in parallel rather than sequentially.
3. **Complex refactoring / architecture analysis**: Use `Agent` with `subagent_type=Plan` to design the approach first, then implement.
4. **After 3 failed lookups**: If a target is not found after 3 tool attempts, immediately delegate to an `Explore` agent rather than continuing to guess paths.
