from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    return {
        "stt": {
            "model_name": data.get("stt", {}).get("model_name", "tiny"),
            "language": data.get("stt", {}).get("language", "zh"),
            "device": data.get("stt", {}).get("device", "cpu"),
            "compute_type": data.get("stt", {}).get("compute_type", "int8"),
        },
        "audio": {
            "sample_rate": data.get("audio", {}).get("sample_rate", 16000),
            "channels": data.get("audio", {}).get("channels", 1),
            "seconds": data.get("audio", {}).get("seconds", 4),
            "output_dir": data.get("audio", {}).get("output_dir", "recordings"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDK 最简 STT 命令行示例")
    parser.add_argument("--config", default="config-edge-rdk-stt.yaml", help="配置文件路径")
    parser.add_argument("--seconds", type=float, default=None, help="单次录音时长（秒）")
    parser.add_argument("--loop", action="store_true", help="循环模式（每轮录音后继续）")
    parser.add_argument("--wav", type=str, default=None, help="直接转写指定的 WAV 文件（跳过录音）")
    return parser.parse_args()


def run_once(recorder: Any, stt: Any, output_dir: Path, seconds: float, wav_in: str | None = None) -> None:
    if wav_in:
        wav_path = Path(wav_in)
        if not wav_path.exists():
            print(f"❌ 文件不存在：{wav_path}")
            return
        print(f"\n📂 正在读取文件：{wav_path}")
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = output_dir / f"record_{timestamp}.wav"
        print(f"\n🎙️ 开始录音：{seconds:.1f} 秒...")
        recorder.record_to_wav(seconds=seconds, output_path=wav_path)
        print(f"✅ 录音完成：{wav_path}")

    print("🧠 正在识别...")
    text = stt.transcribe_wav(wav_path)

    if text:
        print(f"📝 识别结果：{text}")
    else:
        print("⚠️ 未识别到有效语音，请重试并靠近麦克风。")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ 配置文件不存在：{config_path}")
        return 1

    config = load_config(config_path)
    stt_cfg = config["stt"]
    audio_cfg = config["audio"]

    seconds = args.seconds if args.seconds is not None else float(audio_cfg["seconds"])
    output_dir = Path(audio_cfg["output_dir"])

    print("🚀 启动 RDK 最简 STT CLI")
    print(
        f"   模型={stt_cfg['model_name']} | device={stt_cfg['device']} | "
        f"compute_type={stt_cfg['compute_type']} | lang={stt_cfg['language']}"
    )

    try:
        from services.stt import FasterWhisperSTT

        if not args.wav:
            from services.audio_recorder import AudioRecorder
            recorder = AudioRecorder(
                sample_rate=int(audio_cfg["sample_rate"]),
                channels=int(audio_cfg["channels"]),
            )
            recorder.ensure_input_device()
        else:
            recorder = None

        stt = FasterWhisperSTT(
            model_name=str(stt_cfg["model_name"]),
            device=str(stt_cfg["device"]),
            compute_type=str(stt_cfg["compute_type"]),
            language=str(stt_cfg["language"]),
        )
    except Exception as exc:
        print(f"❌ 初始化失败：{exc}")
        print("\n排查建议：")
        print("1) 安装依赖：pip install -r requirements-edge-stt.txt")
        print("2) 预热模型：python -c \"from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')\"")
        print("3) 确认麦克风可用（ALSA 设备正常）")
        return 1

    try:
        if args.wav:
            run_once(None, stt, output_dir, seconds, wav_in=args.wav)
        elif args.loop:
            print("\n按 Enter 开始每一轮录音，Ctrl+C 退出。")
            while True:
                input("\n按 Enter 开始录音...")
                run_once(recorder, stt, output_dir, seconds)
        else:
            input("\n按 Enter 开始录音...")
            run_once(recorder, stt, output_dir, seconds)
    except KeyboardInterrupt:
        print("\n👋 已退出。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
