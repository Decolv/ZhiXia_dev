#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RKLLM NPU推理
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['LD_LIBRARY_PATH'] = os.path.join(script_dir, 'rknn_libs') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTHONPATH'] = os.path.join(script_dir, '.local', 'lib', 'python3.9', 'site-packages') + ':' + os.environ.get('PYTHONPATH', '')

from rkllm_inference import create_rkllm_from_hf

def main():
    print("="*60)
    print("RKLLM NPU推理测试")
    print("="*60)
    
    model_path = '/home/quark/code/models/Qwen3-1.7B-w8a8-rk3588.rkllm'
    
    print(f"\n模型路径: {model_path}")
    print(f"模型存在: {os.path.exists(model_path)}")
    
    try:
        print("\n正在加载模型...")
        llm = create_rkllm_from_hf(model_path, max_new_tokens=64)
        print("✅ 模型加载成功!")
        
        print("\n测试1: 简单生成")
        print("-"*60)
        response = llm.generate("你好，请介绍一下自己。")
        print(f"回复: {response}")
        
        print("\n测试2: 对话模式")
        print("-"*60)
        messages = [
            {"role": "system", "content": "你是一个 helpful 的AI助手。"},
            {"role": "user", "content": "什么是人工智能？"}
        ]
        response = llm.chat(messages)
        print(f"回复: {response}")
        
        print("\n" + "="*60)
        print("✅ 所有测试通过!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
