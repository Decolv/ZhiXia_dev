"""
ZhiXia Notebook 辅助模块 - 封装所有UI和逻辑
"""

import os
import gc
import time
import psutil
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, Audio, clear_output
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ASR 辅助函数
# ============================================================================

def create_asr_ui(project_root, asr_model):
    """创建ASR识别界面"""

    input_mode = widgets.RadioButtons(
        options=['指定WAV文件路径', '上传WAV文件'],
        description='输入方式:',
    )

    file_path_input = widgets.Text(
        value=os.path.join(project_root, 'test.wav'),
        placeholder='输入WAV文件路径',
        description='文件路径:',
        style={'description_width': '100px'}
    )

    file_upload = widgets.FileUpload(accept='.wav', multiple=False, description='上传WAV')

    asr_run_button = widgets.Button(description='🚀 开始识别', button_style='success')
    asr_output = widgets.Output()

    def on_asr_run_clicked(b):
        with asr_output:
            clear_output()

            if asr_model is None:
                print("❌ ASR模型未加载")
                return

            # 确定输入文件
            if input_mode.value == '上传WAV文件':
                if not file_upload.value:
                    print("❌ 请先上传WAV文件")
                    return
                uploaded_filename = list(file_upload.value.keys())[0]
                audio_path = os.path.join(project_root, 'output', uploaded_filename)
                with open(audio_path, 'wb') as f:
                    f.write(file_upload.value[uploaded_filename]['content'])
            else:
                audio_path = file_path_input.value

            if not os.path.exists(audio_path):
                print(f"❌ 文件不存在: {audio_path}")
                return

            print(f"正在识别音频: {audio_path}")

            try:
                start = time.time()
                result = asr_model.generate(audio_path, batch_size_s=300,
                                          frontend_conf={"n_mels": 80, "frame_shift": 10,
                                                        "frame_length": 25, "sample_rate": 16000, "fft": 512})
                elapsed = time.time() - start

                text = result[0].get('text', '') if isinstance(result, list) and result else str(result)

                print(f"✅ 识别完成 (耗时: {elapsed:.2f}s)\n📝 结果: {text}")

                global asr_result
                asr_result = text

            except Exception as e:
                print(f"❌ 失败: {e}")

    asr_run_button.on_click(on_asr_run_clicked)

    display(HTML("<h4>选择输入方式</h4>"))
    display(input_mode)
    display(file_path_input)
    display(file_upload)
    display(asr_run_button)
    display(asr_output)


# ============================================================================
# LLM 辅助函数
# ============================================================================

def create_llm_ui(project_root, llm_model):
    """创建LLM推理界面"""

    from rkllm_inference import create_rkllm_from_hf

    models_dir = os.path.join(project_root, 'models')
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.rkllm')] if os.path.exists(models_dir) else []

    if not available_models:
        print("❌ 未找到 .rkllm 模型文件")
        return

    print(f"找到 {len(available_models)} 个模型")

    model_selector = widgets.Dropdown(
        options=available_models,
        value=available_models[0],
        description='选择模型:',
        style={'description_width': '100px'}
    )

    load_model_button = widgets.Button(description='📥 加载模型', button_style='info')
    model_output = widgets.Output()

    def on_load_model_clicked(b):
        with model_output:
            clear_output()
            global llm_model

            model_path = os.path.join(models_dir, model_selector.value)
            print(f"正在加载: {model_selector.value}...")

            try:
                llm_model = create_rkllm_from_hf(
                    model_path,
                    max_context_len=1024,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9
                )
                print(f"✅ 加载成功! (模型类型: {llm_model.config.model_type})")
            except Exception as e:
                print(f"❌ 加载失败: {e}")
                llm_model = None

    load_model_button.on_click(on_load_model_clicked)

    display(model_selector)
    display(load_model_button)
    display(model_output)


def create_llm_inference_ui(llm_model):
    """创建LLM推理测试界面"""

    temperature_slider = widgets.FloatSlider(value=0.7, min=0.1, max=1.0, step=0.1, description='Temperature:')
    top_p_slider = widgets.FloatSlider(value=0.9, min=0.1, max=1.0, step=0.1, description='Top-p:')
    max_tokens_slider = widgets.IntSlider(value=128, min=64, max=512, step=64, description='Max Tokens:')

    mode_tabs = widgets.ToggleButtons(options=['Generate', 'Chat'], description='模式:')

    generate_prompt = widgets.Textarea(
        value='你好，请介绍一下你自己。',
        placeholder='输入prompt',
        description='Prompt:',
        rows=3
    )

    system_prompt = widgets.Text(value='你是一个有用的AI助手。', description='System:')
    user_message = widgets.Textarea(value='什么是人工智能？', description='User:', rows=2)

    llm_run_button = widgets.Button(description='🚀 推理', button_style='success')
    llm_output = widgets.Output()

    def on_llm_run_clicked(b):
        with llm_output:
            clear_output()

            if llm_model is None:
                print("❌ LLM模型未加载")
                return

            try:
                start = time.time()
                llm_model.config.temperature = temperature_slider.value
                llm_model.config.top_p = top_p_slider.value
                llm_model.config.max_new_tokens = max_tokens_slider.value

                if mode_tabs.value == 'Generate':
                    response = llm_model.generate(generate_prompt.value)
                else:
                    messages = [
                        {"role": "system", "content": system_prompt.value},
                        {"role": "user", "content": user_message.value}
                    ]
                    response = llm_model.chat(messages)

                elapsed = time.time() - start
                print(f"✅ 推理完成 (耗时: {elapsed:.2f}s)\n💬 回复:\n{response}")

            except Exception as e:
                print(f"❌ 推理失败: {e}")

    llm_run_button.on_click(on_llm_run_clicked)

    display(HTML("<h4>推理参数</h4>"))
    display(temperature_slider)
    display(top_p_slider)
    display(max_tokens_slider)

    display(HTML("<h4>选择推理模式</h4>"))
    display(mode_tabs)

    mode_output = widgets.Output()

    def update_mode(change):
        with mode_output:
            clear_output()
            display(generate_prompt if mode_tabs.value == 'Generate' else widgets.VBox([system_prompt, user_message]))

    mode_tabs.observe(update_mode, names='value')
    display(mode_output)
    update_mode(None)

    display(llm_run_button)
    display(llm_output)


# ============================================================================
# TTS 辅助函数
# ============================================================================

def create_tts_synthesis_ui(project_root, tts_synthesis_fast, tts_synthesis_offline):
    """创建单个TTS合成界面"""

    tts_version = widgets.RadioButtons(
        options=['快速版 (MeloTTS + Edge-TTS)', '离线版 (PaddleSpeech + pyttsx3)'],
        description='TTS版本:',
    )

    tts_text = widgets.Textarea(
        value='你好，我是ZhiXia语音助手。',
        placeholder='输入要合成的文本',
        description='文本:',
        rows=3
    )

    tts_run_button = widgets.Button(description='🚀 合成', button_style='success')
    tts_output = widgets.Output()

    def on_tts_run_clicked(b):
        with tts_output:
            clear_output()

            if not tts_text.value.strip():
                print("❌ 请输入要合成的文本")
                return

            try:
                start = time.time()
                mem_before = psutil.virtual_memory().used / (1024**3)

                output_path = os.path.join(project_root, 'output', 'tts_output.wav')

                if '快速版' in tts_version.value:
                    tts_synthesis_fast(tts_text.value, output_path)
                else:
                    tts_synthesis_offline(tts_text.value, output_path)

                elapsed = time.time() - start
                mem_after = psutil.virtual_memory().used / (1024**3)
                mem_used = mem_after - mem_before

                print(f"✅ 合成完成")
                print(f"⏱️ 耗时: {elapsed:.2f}s | 💾 内存: {mem_used:.2f}GB")

                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / 1024
                    print(f"📁 文件: {file_size:.1f}KB\n🔉 试听:")
                    display(Audio(output_path, autoplay=False))

            except Exception as e:
                print(f"❌ 合成失败: {e}")

    tts_run_button.on_click(on_tts_run_clicked)

    display(HTML("<h4>选择TTS版本</h4>"))
    display(tts_version)
    display(HTML("<h4>输入合成文本</h4>"))
    display(tts_text)
    display(tts_run_button)
    display(tts_output)


def create_tts_comparison_ui(project_root, tts_synthesis_fast, tts_synthesis_offline):
    """创建TTS性能对比界面"""

    compare_text = widgets.Textarea(
        value='这是一个语音合成性能测试文本。我们将分别使用快速版和离线版进行合成，对比它们的合成时间和资源占用。',
        placeholder='输入对比文本',
        description='对比文本:',
        rows=3
    )

    compare_button = widgets.Button(description='📈 开始对比', button_style='warning')
    compare_output = widgets.Output()

    def on_compare_clicked(b):
        with compare_output:
            clear_output()

            if not compare_text.value.strip():
                print("❌ 请输入对比文本")
                return

            results = []

            # 快速版
            print("正在测试快速版...")
            try:
                start = time.time()
                mem_before = psutil.virtual_memory().used / (1024**3)

                output_path = os.path.join(project_root, 'output', 'tts_fast.wav')
                tts_synthesis_fast(compare_text.value, output_path)

                fast_elapsed = time.time() - start
                mem_after = psutil.virtual_memory().used / (1024**3)

                results.append({
                    'Version': '快速版 (MeloTTS)',
                    '合成时间(s)': f"{fast_elapsed:.2f}",
                    '内存占用(GB)': f"{mem_after - mem_before:.2f}",
                    'Status': '✅'
                })
                print(f"✅ 完成 ({fast_elapsed:.2f}s)")

            except Exception as e:
                print(f"⚠️ 失败: {e}")
                results.append({
                    'Version': '快速版 (MeloTTS)',
                    '合成时间(s)': '-',
                    '内存占用(GB)': '-',
                    'Status': '❌'
                })

            # 清理内存
            print("\n清理内存中...")
            gc.collect()
            time.sleep(2)

            # 离线版
            print("正在测试离线版...")
            try:
                start = time.time()
                mem_before = psutil.virtual_memory().used / (1024**3)

                output_path = os.path.join(project_root, 'output', 'tts_offline.wav')
                tts_synthesis_offline(compare_text.value, output_path)

                offline_elapsed = time.time() - start
                mem_after = psutil.virtual_memory().used / (1024**3)

                results.append({
                    'Version': '离线版 (PaddleSpeech)',
                    '合成时间(s)': f"{offline_elapsed:.2f}",
                    '内存占用(GB)': f"{mem_after - mem_before:.2f}",
                    'Status': '✅'
                })
                print(f"✅ 完成 ({offline_elapsed:.2f}s)")

            except Exception as e:
                print(f"⚠️ 失败: {e}")
                results.append({
                    'Version': '离线版 (PaddleSpeech)',
                    '合成时间(s)': '-',
                    '内存占用(GB)': '-',
                    'Status': '❌'
                })

            # 展示结果
            print("\n" + "="*60)
            print("📊 性能对比结果")
            print("="*60)
            df = pd.DataFrame(results)
            print("\n" + df.to_string(index=False))

    compare_button.on_click(on_compare_clicked)

    display(HTML("<h4>输入对比文本</h4>"))
    display(compare_text)
    display(compare_button)
    display(compare_output)


# ============================================================================
# 完整流程辅助函数
# ============================================================================

def create_pipeline_ui(project_root, asr_model, llm_model, tts_synthesis_fast, tts_synthesis_offline):
    """创建完整流程界面"""

    pipeline_input = widgets.Text(
        value=os.path.join(project_root, 'test.wav'),
        placeholder='输入音频文件路径',
        description='输入音频:',
        style={'description_width': '100px'}
    )

    pipeline_tts = widgets.RadioButtons(
        options=['快速版', '离线版'],
        description='TTS版本:',
        value='快速版'
    )

    pipeline_run_button = widgets.Button(description='▶️ 运行流程', button_style='success')
    pipeline_output = widgets.Output()

    def on_pipeline_run_clicked(b):
        with pipeline_output:
            clear_output()

            if asr_model is None or llm_model is None:
                print("❌ 模型未加载完整")
                return

            if not os.path.exists(pipeline_input.value):
                print(f"❌ 文件不存在: {pipeline_input.value}")
                return

            print("="*60)
            print("🎬 完整流程执行")
            print("="*60)

            total_start = time.time()

            try:
                # ASR
                print("\n[1/3] 语音识别...")
                asr_start = time.time()
                result = asr_model.generate(pipeline_input.value)
                text = result[0].get('text', '') if isinstance(result, list) and result else str(result)
                asr_time = time.time() - asr_start
                print(f"✅ {asr_time:.2f}s | {text}")

                # LLM
                print("\n[2/3] AI推理...")
                llm_start = time.time()
                response = llm_model.chat([{"role": "user", "content": text}])
                llm_time = time.time() - llm_start
                print(f"✅ {llm_time:.2f}s | {response}")

                # TTS
                print(f"\n[3/3] 语音合成 ({pipeline_tts.value})...")
                tts_start = time.time()
                output_path = os.path.join(project_root, 'output', 'pipeline.wav')
                if '快速版' in pipeline_tts.value:
                    tts_synthesis_fast(response, output_path)
                else:
                    tts_synthesis_offline(response, output_path)
                tts_time = time.time() - tts_start

                total_time = time.time() - total_start
                print(f"✅ {tts_time:.2f}s")

                print("\n" + "="*60)
                print(f"✅ 完成 (总耗时: {total_time:.2f}s)")
                print("="*60)
                print(f"\n⏱️ ASR:{asr_time:.2f}s | LLM:{llm_time:.2f}s | TTS:{tts_time:.2f}s")
                print(f"\n🔉 最终音频:")
                display(Audio(output_path, autoplay=False))

            except Exception as e:
                print(f"❌ 失败: {e}")

    pipeline_run_button.on_click(on_pipeline_run_clicked)

    display(HTML("<h4>配置输入</h4>"))
    display(pipeline_input)
    display(HTML("<h4>选择TTS版本</h4>"))
    display(pipeline_tts)
    display(pipeline_run_button)
    display(pipeline_output)
