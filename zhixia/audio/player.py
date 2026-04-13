"""ALSA 音频播放"""

import logging
import shutil
import subprocess
from pathlib import Path

from zhixia.audio.base import AudioPlayer

logger = logging.getLogger(__name__)

_PLAYER_COMMANDS = [
    ["aplay"],
    ["paplay"],
    ["ffplay", "-nodisp", "-autoexit"],
]

# 支持从 stdin 读取的播放器（aplay/ffplay 均支持 -）
_STDIN_CAPABLE = {"aplay", "ffplay"}


class ALSAAudioPlayer(AudioPlayer):

    def play(self, audio_path: Path, blocking: bool = True) -> bool:
        if not audio_path.exists():
            logger.error("音频文件不存在: %s", audio_path)
            return False

        for cmd in _PLAYER_COMMANDS:
            if shutil.which(cmd[0]) is None:
                continue

            logger.info("播放音频 (%s): %s", cmd[0], audio_path)
            proc = subprocess.Popen(
                cmd + [str(audio_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if blocking:
                proc.wait()
            return True

        logger.warning("未找到音频播放器")
        return False

    def play_bytes(self, wav_bytes: bytes, blocking: bool = True) -> bool:
        """直接将 WAV bytes pipe 给播放器，无需写临时文件。"""
        for cmd in _PLAYER_COMMANDS:
            player = cmd[0]
            if shutil.which(player) is None:
                continue

            if player in _STDIN_CAPABLE:
                # aplay 和 ffplay 都支持从 stdin 读取（用 - 表示）
                stdin_cmd = cmd + ["-"]
                logger.debug("pipe 播放 (%s): %d bytes", player, len(wav_bytes))
                proc = subprocess.Popen(
                    stdin_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.stdin.write(wav_bytes)
                proc.stdin.close()
                if blocking:
                    proc.wait()
                return True
            else:
                # paplay 不支持 stdin，回退到临时文件
                return super().play_bytes(wav_bytes, blocking=blocking)

        logger.warning("未找到音频播放器")
        return False

    def is_available(self) -> bool:
        return any(shutil.which(cmd[0]) for cmd in _PLAYER_COMMANDS)

