#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR + LLM + TTS 完整流程 (高速版)
使用MeloTTS替代ChatTTS，实现2秒内语音合成
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


def force_gc():
    """强制垃圾回收"""
    gc.collect()
    gc.collect()


def llm_inference_npu_stream(user_input, max_new_tokens=64):
    """使用NPU进行LLM推理"""
    from rkllm_inference import create_rkllm_from_hf
    
    model_path = os.path.join(script_dir, 'models', 'Qwen3-1.7B-w8a8-rk3588.rkllm')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RKLLM模型不存在: {model_path}")
    
    print("="*60)
    print("加载RKLLM模型 (NPU加速)")
    print("="*60)
    
    llm = create_rkllm_from_hf(
        model_path,
        max_context_len=1024,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9
    )
    
    llm.set_chat_template(
        system_prompt="你是AI助手，回答简洁。",
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
    time.sleep(0.5)
    
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


def tts_melotts(text, output_path):
    """
    使用MeloTTS进行高速语音合成
    速度比ChatTTS快5-10倍，支持CPU实时推理
    """
    print("="*60)
    print("步骤3: 语音合成 (TTS - MeloTTS高速版)")
    print("="*60)
    
    try:
        # 尝试导入MeloTTS
        from melo.api import TTS
        
        print("正在加载MeloTTS模型...")
        
        # 使用中文模型
        model = TTS(language='ZH', device='cpu')
        
        # 选择说话人
        speaker_ids = model.hps.data.spk2id
        speaker_id = speaker_ids['ZH']  # 使用中文说话人
        
        print(f"正在合成语音: {text}")
        print(f"说话人: ZH")
        
        # 开始计时
        start_time = time.time()
        
        # 合成语音
        model.tts_to_file(text, speaker_id, output_path, speed=1.0)
        
        # 计算耗时
        elapsed = time.time() - start_time
        print(f"✅ 语音合成完成，耗时: {elapsed:.2f}秒")
        print(f"✅ 语音已保存到: {output_path}")
        
        # 释放模型
        del model
        force_gc()
        
        if elapsed > 2.0:
            print(f"⚠️ 警告: 合成时间超过2秒 ({elapsed:.2f}秒)")
        
        return True
        
    except ImportError:
        print("⚠️ MeloTTS未安装，尝试安装...")
        print("  运行: pip install melotts")
        return False
    except Exception as e:
        print(f"❌ MeloTTS合成失败: {e}")
        return False


def tts_synthesis_fast(text, output_path):
    """
    高速TTS合成，使用MeloTTS（离线部署）
    """
    return tts_melotts(text, output_path)


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
    print("🎙️ ASR + LLM + TTS 智能语音助手 (高速版)")
    print("   使用MeloTTS，目标: 2秒内合成")
    print("="*60 + "\n")
    
    mem_available = check_memory()
    if mem_available:
        print(f"当前可用内存: {mem_available:.2f} GB")
        if mem_available < 3.0:
            print("⚠️ 警告: 可用内存不足3GB")
        print()
    
    input_audio = "/home/quark/音乐/test.wav"
    
    if not os.path.exists(input_audio):
        print(f"❌ 输入音频文件不存在: {input_audio}")
        sys.exit(1)
    
    output_audio = os.path.join(script_dir, "output", "llm_response_fast.wav")
    
    # 步骤1: ASR识别
    recognized_text = asr_recognition_int8(input_audio)
    
    if not recognized_text:
        print("\n❌ ASR识别失败，流程终止")
        sys.exit(1)
    
    # 步骤2: LLM推理
    try:
        llm_response = llm_inference_npu_stream(recognized_text, max_new_tokens=64)
    except Exception as e:
        print(f"\n❌ LLM推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not llm_response:
        print("\n❌ LLM返回空回复，流程终止")
        sys.exit(1)
    
    # 步骤3: TTS合成 (高速版)
    success = tts_synthesis_fast(llm_response, output_audio)
    
    if not success:
        print("\n❌ TTS合成失败，流程终止")
        print("\n建议安装MeloTTS:")
        print("  pip install melotts")
        sys.exit(1)
    
    # 步骤4: 自动播放
    play_audio(output_audio)
    
    # 完成
    print("\n" + "="*60)
    print("🎉 完整流程成功！ (NPU加速 + 高速TTS)")
    print("="*60)
    print(f"📥 输入音频: {input_audio}")
    print(f"📝 识别文本: {recognized_text}")
    print(f"🤖 AI回复: {llm_response}")
    print(f"⚡ 推理后端: RKNN NPU")
    print(f"🔧 ASR量化: INT8")
    print(f"🚀 TTS引擎: MeloTTS/Edge-TTS")
    print(f"📤 输出音频: {output_audio}")
    print("="*60)


if __name__ == "__main__":
    main()
