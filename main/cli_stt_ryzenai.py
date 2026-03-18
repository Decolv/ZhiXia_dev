from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
from typing import Any
import yaml

from services.stt_npu import STTFactory
from services.audio_recorder import AudioRecorder

def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMD NPU (RyzenAI) STT 命令行示例")
    parser.add_argument("--config", default="config-amd-npu-stt.yaml", help="配置文件路径")
    parser.add_argument("--seconds", type=float, default=None, help="单次录音时长（秒）")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ 配置文件不存在：{config_path}")
        return 1

    config = load_config(config_path)
    stt_cfg = config.get("stt", {})
    audio_cfg = config.get("audio", {})

    seconds = args.seconds if args.seconds is not None else float(audio_cfg.get("seconds", 4))
    output_dir = Path(audio_cfg.get("output_dir", "recordings"))
    output_dir.mkdir(exist_ok=True)

    print(f"🚀 启动 AMD NPU STT CLI")
    print(f"   引擎={stt_cfg.get('engine')} | 提供商={stt_cfg.get('provider')}")

    try:
        # 使用工厂模式获取 STT 引擎
        stt = STTFactory.get_engine(stt_cfg)
        
        recorder = AudioRecorder(
            sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            channels=int(audio_cfg.get("channels", 1)),
        )
        recorder.ensure_input_device()

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = output_dir / f"npu_record_{timestamp}.wav"
        
        print(f"\n🎙️ 开始录音：{seconds:.1f} 秒...")
        recorder.record_to_wav(seconds=seconds, output_path=wav_path)
        print(f"✅ 录音完成：{wav_path}")

        print("🧠 正在使用 AMD NPU 识别...")
        text = stt.transcribe_wav(wav_path)

        if text:
            print(f"📝 识别结果：{text}")
        else:
            print("⚠️ 未识别到有效语音。")

    except Exception as e:
        print(f"❌ 运行出错：{e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
