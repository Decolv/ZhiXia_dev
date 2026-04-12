#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将HuggingFace模型转换为RKLLM格式
在RK3588上直接转换（需要原始模型已下载）
"""

import os
import sys
import json
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = os.path.join(script_dir, '.local', 'lib', 'python3.9', 'site-packages') + ':' + os.environ.get('PYTHONPATH', '')

def check_model_files(model_path: str) -> bool:
    """检查模型文件是否完整"""
    required_files = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'model.safetensors.index.json',  # 或 pytorch_model.bin.index.json
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            # 检查替代文件
            if f == 'model.safetensors.index.json':
                if os.path.exists(os.path.join(model_path, 'pytorch_model.bin.index.json')):
                    continue
                # 检查是否有单个模型文件
                if any(os.path.exists(os.path.join(model_path, f'model-{i:05d}-of-{j:05d}.safetensors')) 
                       for i in range(1, 10) for j in range(1, 10)):
                    continue
            missing.append(f)
    
    if missing:
        print(f"❌ 缺少必要文件: {missing}")
        return False
    
    return True


def generate_quant_data(model_path: str, output_path: str = "data_quant.json"):
    """
    生成量化校准数据
    
    Args:
        model_path: 模型路径
        output_path: 输出文件路径
    """
    print("="*60)
    print("生成量化校准数据")
    print("="*60)
    
    # 量化校准数据示例
    quant_data = {
        "version": "1.0",
        "model_path": model_path,
        "quantization": {
            "method": "w8a8",
            "calibration_data": [
                "你好，请介绍一下自己。",
                "什么是人工智能？",
                "请解释一下机器学习。",
                "深度学习是什么？",
                "神经网络如何工作？",
                "自然语言处理的应用有哪些？",
                "计算机视觉是什么？",
                "强化学习的基本原理是什么？",
                "生成式AI有哪些应用？",
                "大语言模型是如何训练的？"
            ]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(quant_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 量化校准数据已保存: {output_path}")
    return output_path


def create_export_script(model_path: str, output_path: str, max_context: int = 4096):
    """
    创建模型导出脚本
    
    Args:
        model_path: 模型路径
        output_path: 输出RKLLM文件路径
        max_context: 最大上下文长度
    """
    print("="*60)
    print("创建导出脚本")
    print("="*60)
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKLLM模型导出脚本
自动生成的导出配置
"""

import os
import sys

# 模型路径
model_path = "{model_path}"
output_path = "{output_path}"
max_context = {max_context}

def export_rkllm():
    """导出RKLLM模型"""
    
    # 尝试导入rkllm_toolkit
    try:
        from rkllm.api import RKLLM
        print("✅ 成功导入RKLLM Toolkit")
    except ImportError as e:
        print(f"❌ 无法导入RKLLM Toolkit: {{e}}")
        print("请确保已安装rkllm-toolkit")
        return False
    
    # 创建RKLLM实例
    llm = RKLLM()
    
    # 加载模型
    print(f"正在加载模型: {{model_path}}")
    ret = llm.load_huggingface(model=model_path)
    if ret != 0:
        print(f"❌ 模型加载失败，错误码: {{ret}}")
        return False
    print("✅ 模型加载成功")
    
    # 构建模型
    print("正在构建RKLLM模型...")
    print(f"配置: max_context={{max_context}}, quantization=w8a8")
    
    ret = llm.build(
        rkllm_param=None,  # 使用默认参数
        max_context_len=max_context,
        quantization='w8a8',  # W8A8量化
        target_platform='rk3588'
    )
    if ret != 0:
        print(f"❌ 模型构建失败，错误码: {{ret}}")
        return False
    print("✅ 模型构建成功")
    
    # 导出模型
    print(f"正在导出模型到: {{output_path}}")
    ret = llm.export_rkllm(output_path)
    if ret != 0:
        print(f"❌ 模型导出失败，错误码: {{ret}}")
        return False
    print("✅ 模型导出成功")
    
    return True


if __name__ == "__main__":
    success = export_rkllm()
    sys.exit(0 if success else 1)
'''
    
    script_path = os.path.join(os.path.dirname(output_path), 'export_rkllm_script.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✅ 导出脚本已创建: {script_path}")
    return script_path


def convert_model(model_path: str, output_dir: str = None, max_context: int = 4096):
    """
    转换模型到RKLLM格式
    
    Args:
        model_path: HuggingFace模型路径
        output_dir: 输出目录
        max_context: 最大上下文长度
    """
    print("\n" + "="*60)
    print("RKLLM模型转换")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"最大上下文: {max_context}")
    print("="*60 + "\n")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("\n请确保模型已下载。可以使用以下命令下载:")
        print(f"  export PYTHONPATH=\"{os.path.join(script_dir, '.local', 'lib', 'python3.9', 'site-packages')}:${{PYTHONPATH}}\"")
        print(f"  python3 -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('{os.path.basename(model_path)}', cache_dir='{os.path.join(script_dir, 'models')}')\"")
        return False
    
    if not check_model_files(model_path):
        print("❌ 模型文件不完整")
        return False
    
    # 确定输出路径
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'models')
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(os.path.normpath(model_path))
    output_path = os.path.join(output_dir, f"{model_name}_w8a8_rk3588.rkllm")
    
    # 生成量化数据
    quant_data_path = os.path.join(output_dir, "data_quant.json")
    generate_quant_data(model_path, quant_data_path)
    
    # 创建导出脚本
    script_path = create_export_script(model_path, output_path, max_context)
    
    print("\n" + "="*60)
    print("转换准备完成")
    print("="*60)
    print(f"\n由于网络限制，需要手动完成以下步骤:")
    print(f"\n1. 在可以访问外网的x86 Linux机器上:")
    print(f"   - 安装RKLLM Toolkit: pip install rkllm-toolkit")
    print(f"   - 下载模型: git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct")
    print(f"\n2. 运行导出脚本:")
    print(f"   python3 {script_path}")
    print(f"\n3. 将生成的RKLLM模型复制到本机:")
    print(f"   scp {output_path} user@rk3588:{output_path}")
    print(f"\n或者使用预转换模型:")
    print(f"   从 https://github.com/airockchip/rknn-llm/releases 下载预转换模型")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='将HuggingFace模型转换为RKLLM格式')
    parser.add_argument('--model-path', type=str, 
                        default=os.path.join(script_dir, 'models', 'Qwen2.5-1.5B-Instruct'),
                        help='HuggingFace模型路径')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(script_dir, 'models'),
                        help='输出目录')
    parser.add_argument('--max-context', type=int, default=4096,
                        help='最大上下文长度')
    
    args = parser.parse_args()
    
    success = convert_model(args.model_path, args.output_dir, args.max_context)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
