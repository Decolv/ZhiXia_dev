#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR + LLM + TTS 完整流程 (Piper TTS超高速版)
使用Piper TTS，速度极快，专为嵌入式设备优化
"""

import os
import sys
import gc
import time

# 自动配置环境变量
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['MODELSCOPE_CACHE'] = os.path.join(script_dir, '.cache', 'modelscope')
os.environ['HOME'] = script_dir
os.environ['PYTHONPATH'] = os.path.join(script_dir, '.local', 'lib', 'python3.9', 'site-packages') + ':' + os.environ.get('PYTHONPATH', '')
os.environ['LD_LIBRARY_PATH'] = os.path.join(script_dir, 'rknn_libs') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# 确保目录存在
os.makedirs(os.path.join(script_dir, '.cache', 'modelscope'), exist_ok=True)
os.makedirs(os.path.join(script_dir, 'output'), exist_ok=True)
os.makedirs(os.path.join(script_dir, 'models', 'piper'), exist_ok=True)


def force_gc():
    """强制垃圾回收"""
    gc.collect()
    gc.collect()


def llm_inference_npu_stream(user_input, max_new_tokens=32):
    """使用NPU进行LLM推理（快速响应模式）"""
    from rkllm_inference import create_rkllm_from_hf
    
    model_path = os.path.join(script_dir, 'models', 'Qwen3-1.7B-w8a8-rk3588.rkllm')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RKLLM模型不存在: {model_path}")
    
    print("="*60)
    print("加载RKLLM模型 (NPU加速 - 快速模式)")
    print("="*60)
    
    llm = create_rkllm_from_hf(
        model_path,
        max_context_len=512,  # 减少上下文长度
        max_new_tokens=max_new_tokens,
        temperature=0.8,  # 提高温度，减少计算
        top_p=0.95
    )
    
    llm.set_chat_template(
        system_prompt="你是AI助手，用一句话简短回答。",
        prompt_prefix="",
        prompt_postfix=""
    )
    
    print("✅ RKLLM模型加载成功")
    
    messages = [{"role": "user", "content": user_input}]
    
    print(f"用户输入: {user_input}")
    print("正在使用NPU生成回复...")
    
    response = llm.chat(messages, max_new_tokens=max_new_tokens)
    response = response.strip()
    
    print(f"✅ LLM回复: {response}")
    
    del llm
    force_gc()
    time.sleep(0.3)
    
    return response


def asr_recognition_int8(audio_path):
    """使用INT8量化ASR模型进行中文语音识别"""
    from funasr import AutoModel
    
    print("="*60)
    print("步骤1: 语音识别 (ASR - INT8量化版)")
    print("="*60)
    
    print("正在加载INT8量化ASR模型...")
    
    try:
        model = AutoModel(
            model="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",
            vad_model=None,
            punc_model=None,
            disable_update=True,
            hub="ms",
            quantize=True,
            device="cpu",
        )
        print("✅ INT8量化ASR模型加载完成")
    except Exception as e:
        print(f"⚠️ INT8模型加载失败: {e}")
        model = AutoModel(
            model="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",
            vad_model=None,
            punc_model=None,
            disable_update=True,
            hub="ms"
        )
        print("✅ ASR模型加载完成(标准版)")
    
    print(f"正在识别音频: {audio_path}")
    result = model.generate(input=audio_path)
    
    del model
    force_gc()
    time.sleep(0.5)
    
    if result and len(result) > 0:
        text = result[0].get('text', '')
        print(f"✅ 识别结果: {text}")
        return text
    else:
        print("❌ 识别失败")
        return None


def download_piper_model():
    """下载Piper中文模型"""
    import urllib.request
    
    model_dir = os.path.join(script_dir, 'models', 'piper')
    model_file = os.path.join(model_dir, 'zh_CN-huayan-medium.onnx')
    config_file = os.path.join(model_dir, 'zh_CN-huayan-medium.onnx.json')
    
    if os.path.exists(model_file) and os.path.exists(config_file):
        return model_file, config_file
    
    print("正在下载Piper中文模型...")
    
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/"
    
    try:
        print("下载模型文件...")
        urllib.request.urlretrieve(
            base_url + "zh_CN-huayan-medium.onnx",
            model_file
        )
        
        print("下载配置文件...")
        urllib.request.urlretrieve(
            base_url + "zh_CN-huayan-medium.onnx.json",
            config_file
        )
        
        print("✅ 模型下载完成")
        return model_file, config_file
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("\n手动下载方法:")
        print(f"1. 访问: {base_url}")
        print(f"2. 下载 zh_CN-huayan-medium.onnx 到 {model_dir}")
        print(f"3. 下载 zh_CN-huayan-medium.onnx.json 到 {model_dir}")
        return None, None


def tts_piper(text, output_path):
    """
    使用Piper TTS进行超高速语音合成
    速度比ChatTTS快10-20倍，专为嵌入式设备优化
    """
    print("="*60)
    print("步骤3: 语音合成 (TTS - Piper)")
    print("="*60)
    
    # 方法1: 使用piper-tts Python包
    try:
        from piper import PiperVoice
        import wave
        
        print("正在加载Piper TTS模型...")
        
        # 下载或获取模型路径
        model_file, config_file = download_piper_model()
        
        if not model_file or not config_file:
            raise FileNotFoundError("Piper模型文件不存在")
        
        # 加载模型
        voice = PiperVoice.load(model_file, config_file)
        
        print(f"正在合成语音: {text}")
        
        # 开始计时
        start_time = time.time()
        
        # 合成语音
        with wave.open(output_path, 'wb') as wav_file:
            voice.synthesize(text, wav_file)
        
        # 计算耗时
        elapsed = time.time() - start_time
        print(f"✅ 语音合成完成，耗时: {elapsed:.2f}秒")
        print(f"✅ 语音已保存到: {output_path}")
        
        # 释放模型
        del voice
        force_gc()
        
        return True
        
    except ImportError:
        # 方法2: 使用命令行版本的piper
        print("⚠️ piper-tts Python包未安装，尝试使用命令行版本...")
        return tts_piper_cli(text, output_path)
        
    except Exception as e:
        print(f"❌ Piper TTS合成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def tts_piper_cli(text, output_path):
    """使用Piper命令行版本"""
    import subprocess
    
    try:
        # 检查piper命令是否存在
        result = subprocess.run(['which', 'piper'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️ 未找到piper命令")
            return False
        
        model_file, config_file = download_piper_model()
        if not model_file:
            return False
        
        print(f"正在使用Piper CLI合成: {text}")
        
        # 开始计时
        start_time = time.time()
        
        # 使用piper命令行
        cmd = f'echo "{text}" | piper --model {model_file} --output_file {output_path}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"✅ 语音合成完成，耗时: {elapsed:.2f}秒")
            print(f"✅ 语音已保存到: {output_path}")
            return True
        else:
            print(f"❌ Piper CLI执行失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Piper CLI失败: {e}")
        return False





def play_audio(audio_path):
    """自动播放音频文件"""
    import subprocess
    
    print("="*60)
    print("步骤4: 播放音频")
    print("="*60)
    
    if not os.path.exists(audio_path):
        print(f"❌ 音频文件不存在: {audio_path}")
        return False
    
    print(f"正在播放: {audio_path}")
    
    players = [
        ['aplay', audio_path],
        ['paplay', audio_path],
        ['ffplay', '-nodisp', '-autoexit', audio_path],
    ]
    
    for player in players:
        try:
            result = subprocess.run(['which', player[0]], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                print(f"使用播放器: {player[0]}")
                subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✅ 音频播放已启动")
                return True
        except:
            continue
    
    print("⚠️ 未找到音频播放器，请手动播放")
    print(f"音频文件位置: {audio_path}")
    return False


def check_memory():
    """检查可用内存"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            mem_available = None
            for line in lines:
                if line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) / 1024 / 1024
                    break
            return mem_available
    except:
        return None


def main():
    print("\n" + "="*60)
    print("🎙️ ASR + LLM + TTS 智能语音助手 (Piper版)")
    print("   快速响应模式")
    print("="*60 + "\n")
    
    mem_available = check_memory()
    if mem_available:
        print(f"当前可用内存: {mem_available:.2f} GB")
        if mem_available < 2.0:
            print("⚠️ 警告: 可用内存不足2GB")
        print()
    
    input_audio = "/home/quark/音乐/test.wav"
    
    if not os.path.exists(input_audio):
        print(f"❌ 输入音频文件不存在: {input_audio}")
        sys.exit(1)
    
    output_audio = os.path.join(script_dir, "output", "llm_response_piper.wav")
    
    # 步骤1: ASR识别
    recognized_text = asr_recognition_int8(input_audio)
    
    if not recognized_text:
        print("\n❌ ASR识别失败，流程终止")
        sys.exit(1)
    
    # 步骤2: LLM推理（快速模式：max_new_tokens=32）
    try:
        llm_response = llm_inference_npu_stream(recognized_text, max_new_tokens=32)
    except Exception as e:
        print(f"\n❌ LLM推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not llm_response:
        print("\n❌ LLM返回空回复，流程终止")
        sys.exit(1)
    
    # 步骤3: TTS合成（仅使用Piper）
    success = tts_piper(llm_response, output_audio)
    
    if not success:
        print("\n❌ TTS合成失败，流程终止")
        print("\n请安装Piper TTS:")
        print("  pip install piper-tts")
        print("\n或运行安装脚本:")
        print("  bash install_fast_tts.sh")
        sys.exit(1)
    
    # 步骤4: 自动播放
    play_audio(output_audio)
    
    # 完成
    print("\n" + "="*60)
    print("🎉 完整流程成功！")
    print("="*60)
    print(f"📥 输入音频: {input_audio}")
    print(f"📝 识别文本: {recognized_text}")
    print(f"🤖 AI回复: {llm_response}")
    print(f"⚡ 推理后端: RKNN NPU (快速模式)")
    print(f"🔧 ASR量化: INT8")
    print(f"🚀 TTS引擎: Piper")
    print(f"📤 输出音频: {output_audio}")
    print("="*60)


if __name__ == "__main__":
    main()

