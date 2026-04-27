"""Pytest 共享配置和 fixture"""

import os
import sys
from pathlib import Path

import pytest

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 设置默认环境变量
os.environ.setdefault("MODELSCOPE_CACHE", str(_PROJECT_ROOT / ".cache" / "modelscope"))


@pytest.fixture
def project_root():
    """返回项目根目录路径"""
    return _PROJECT_ROOT


@pytest.fixture
def sample_wav_path(project_root):
    """返回示例音频文件路径"""
    return project_root / "assets" / "sample.wav"
