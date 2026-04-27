#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[已废弃] ZhiXia 语音助手 — 重构后兼容入口

⚠️ 此脚本已废弃，不再推荐使用。

推荐使用方式:
    python -m zhixia

此文件保留仅为向后兼容，功能完全由 zhixia.__main__ 提供。
"""

import os
import sys
import subprocess
from pathlib import Path

# 获取当前脚本所在目录（项目根目录）
script_dir = Path(__file__).resolve().parent.parent

# 添加项目根目录到 Python 路径，确保可以导入 zhixia 包
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 设置环境变量（与原实现保持一致）
os.environ['MODELSCOPE_CACHE'] = str(script_dir / '.cache' / 'modelscope')
os.environ['HOME'] = str(script_dir)
os.environ['PYTHONPATH'] = str(script_dir / '.local' / 'lib' / 'python3.9' / 'site-packages') + ':' + os.environ.get('PYTHONPATH', '')
os.environ['LD_LIBRARY_PATH'] = str(script_dir / 'rknn_libs') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# 确保目录存在（保持与原实现一致）
os.makedirs(os.environ['MODELSCOPE_CACHE'], exist_ok=True)
os.makedirs(script_dir / 'output', exist_ok=True)
os.makedirs(script_dir / 'models' / 'piper', exist_ok=True)


def main():
    """调用新实现的 zhixia 包"""
    print("=" * 60)
    print("🎙️ ZhiXia 语音助手（重构版）")
    print("   使用模块化架构，支持 RAG 和结构化输出")
    print("=" * 60 + "\n")

    try:
        # 调用新的实现
        subprocess.run([sys.executable, "-m", "zhixia"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行失败，错误码: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
