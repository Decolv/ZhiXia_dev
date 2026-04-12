#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKLLM NPU推理模块
使用ctypes调用librkllmrt.so进行NPU加速推理
支持Qwen2.5和Qwen3模型
"""

import os
import sys
import ctypes
import json
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass
from enum import IntEnum

# 模型类型检测
MODEL_TYPE_QWEN2 = "qwen2"
MODEL_TYPE_QWEN3 = "qwen3"

# 加载RKLLM Runtime库
script_dir = os.path.dirname(os.path.abspath(__file__))
rkllm_lib_path = os.path.join(script_dir, 'rknn_libs', 'librkllmrt.so')

if not os.path.exists(rkllm_lib_path):
    # 尝试其他路径
    rkllm_lib_path = '/usr/lib/librkllmrt.so'
    if not os.path.exists(rkllm_lib_path):
        rkllm_lib_path = os.path.join(script_dir, 'librkllmrt.so')

try:
    _rkllm_lib = ctypes.CDLL(rkllm_lib_path)
    print(f"✅ 成功加载RKLLM库: {rkllm_lib_path}")
except OSError as e:
    print(f"❌ 无法加载RKLLM库: {e}")
    print(f"查找路径: {rkllm_lib_path}")
    _rkllm_lib = None


class LLMCallState(IntEnum):
    """LLM调用状态"""
    RUN_NORMAL = 0
    RUN_WAITING = 1
    RUN_FINISH = 2
    RUN_ERROR = 3


class RKLLMInputType(IntEnum):
    """输入类型"""
    PROMPT = 0
    TOKEN = 1
    EMBED = 2
    MULTIMODAL = 3


class RKLLMInferMode(IntEnum):
    """推理模式"""
    GENERATE = 0
    GET_LAST_HIDDEN_LAYER = 1
    GET_LOGITS = 2


# 定义C结构体
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", ctypes.c_void_p),  # RKLLMEmbedInput
        ("token_input", ctypes.c_void_p),  # RKLLMTokenInput
        ("multimodal_input", ctypes.c_void_p),  # RKLLMMultiModalInput
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input", RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", ctypes.c_void_p),
        ("logits", ctypes.c_void_p),
        ("perf", RKLLMPerfStat),
    ]


# 回调函数类型
LLMResultCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int
)


@dataclass
class RKLLMConfig:
    """RKLLM配置"""
    model_path: str
    max_context_len: int = 4096
    max_new_tokens: int = 256
    top_k: int = 40
    top_p: float = 0.9
    temperature: float = 0.7
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    skip_special_token: bool = True
    model_type: str = "auto"  # auto, qwen2, qwen3
    enable_thinking: bool = False  # Qwen3思考模式


class RKLLM:
    """RKLLM推理类"""
    
    def __init__(self, config: RKLLMConfig):
        if _rkllm_lib is None:
            raise RuntimeError("RKLLM库未加载，无法初始化")
        
        self.config = config
        self.handle = ctypes.c_void_p()
        self._callback = None
        self._result_buffer = []
        
        # 自动检测模型类型
        if self.config.model_type == "auto":
            self.config.model_type = self._detect_model_type()
        
        # 创建默认参数
        self._create_default_param = _rkllm_lib.rkllm_createDefaultParam
        self._create_default_param.restype = RKLLMParam
        
        # 初始化函数
        self._init = _rkllm_lib.rkllm_init
        self._init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), LLMResultCallback]
        self._init.restype = ctypes.c_int
        
        # 运行函数
        self._run = _rkllm_lib.rkllm_run
        self._run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self._run.restype = ctypes.c_int
        
        # 销毁函数
        self._destroy = _rkllm_lib.rkllm_destroy
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = ctypes.c_int
        
        # 设置chat template
        self._set_chat_template = _rkllm_lib.rkllm_set_chat_template
        self._set_chat_template.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self._set_chat_template.restype = ctypes.c_int
        
        self._init_model()
    
    def _detect_model_type(self) -> str:
        """从模型文件名检测模型类型"""
        model_name = os.path.basename(self.config.model_path).lower()
        if "qwen3" in model_name:
            return MODEL_TYPE_QWEN3
        elif "qwen2" in model_name or "qwen-2" in model_name:
            return MODEL_TYPE_QWEN2
        else:
            # 默认使用Qwen2格式
            return MODEL_TYPE_QWEN2
    
    def _result_callback(self, result_ptr, userdata, state):
        """结果回调函数"""
        if result_ptr:
            result = result_ptr.contents
            if result.text:
                text = result.text.decode('utf-8')
                self._result_buffer.append(text)
                
                # 打印性能统计
                if state == LLMCallState.RUN_FINISH:
                    perf = result.perf
                    print(f"\n[性能统计] Prefill: {perf.prefill_time_ms:.2f}ms ({perf.prefill_tokens} tokens), "
                          f"Generate: {perf.generate_time_ms:.2f}ms ({perf.generate_tokens} tokens), "
                          f"Memory: {perf.memory_usage_mb:.2f}MB")
        
        return 0  # 继续推理
    
    def _init_model(self):
        """初始化模型"""
        # 获取默认参数
        param = self._create_default_param()
        
        # 设置参数
        param.model_path = self.config.model_path.encode('utf-8')
        param.max_context_len = self.config.max_context_len
        param.max_new_tokens = self.config.max_new_tokens
        param.top_k = self.config.top_k
        param.top_p = self.config.top_p
        param.temperature = self.config.temperature
        param.repeat_penalty = self.config.repeat_penalty
        param.frequency_penalty = self.config.frequency_penalty
        param.presence_penalty = self.config.presence_penalty
        param.skip_special_token = self.config.skip_special_token
        param.is_async = False
        
        # 设置扩展参数
        param.extend_param.enabled_cpus_num = 4  # 启用4个CPU核心
        param.extend_param.enabled_cpus_mask = 0x0F  # CPU 0-3
        
        # 创建回调
        self._callback = LLMResultCallback(self._result_callback)
        
        # 初始化
        ret = self._init(ctypes.byref(self.handle), ctypes.byref(param), self._callback)
        if ret != 0:
            raise RuntimeError(f"RKLLM初始化失败，错误码: {ret}")
        
        print(f"✅ RKLLM模型初始化成功: {self.config.model_path}")
    
    def set_chat_template(self, system_prompt: str = "", prompt_prefix: str = "", prompt_postfix: str = ""):
        """设置对话模板"""
        ret = self._set_chat_template(
            self.handle,
            system_prompt.encode('utf-8'),
            prompt_prefix.encode('utf-8'),
            prompt_postfix.encode('utf-8')
        )
        if ret != 0:
            print(f"⚠️ 设置chat template失败，错误码: {ret}")
    
    def generate(self, prompt: str, role: str = "user") -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            role: 角色 (user/tool)
            
        Returns:
            生成的文本
        """
        self._result_buffer = []
        
        # 构建输入
        input_data = RKLLMInput()
        input_data.role = role.encode('utf-8')
        input_data.enable_thinking = False
        input_data.input_type = RKLLMInputType.PROMPT
        input_data.input.prompt_input = prompt.encode('utf-8')
        
        # 构建推理参数
        infer_param = RKLLMInferParam()
        infer_param.mode = RKLLMInferMode.GENERATE
        infer_param.lora_params = None
        infer_param.prompt_cache_params = None
        infer_param.keep_history = 1
        
        # 运行推理
        ret = self._run(self.handle, ctypes.byref(input_data), ctypes.byref(infer_param), None)
        if ret != 0:
            raise RuntimeError(f"RKLLM推理失败，错误码: {ret}")
        
        return ''.join(self._result_buffer)
    
    def chat(self, messages: list, max_new_tokens: Optional[int] = None) -> str:
        """
        对话模式
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            max_new_tokens: 最大生成token数
            
        Returns:
            生成的回复
        """
        # 根据模型类型选择不同的对话模板
        if self.config.model_type == MODEL_TYPE_QWEN3:
            return self._chat_qwen3(messages, max_new_tokens)
        else:
            return self._chat_qwen2(messages, max_new_tokens)
    
    def _chat_qwen2(self, messages: list, max_new_tokens: Optional[int] = None) -> str:
        """Qwen2格式对话"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")
        
        prompt_parts.append("<|assistant|>\n")
        prompt = "\n".join(prompt_parts)
        
        return self.generate(prompt)
    
    def _chat_qwen3(self, messages: list, max_new_tokens: Optional[int] = None) -> str:
        """Qwen3格式对话 - 使用思考模式"""
        prompt_parts = ['<|im_start|>']
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f'<|im_start|>system\n{content}<|im_end|>')
            elif role == "user":
                prompt_parts.append(f'<|im_start|>user\n{content}<|im_end|>')
            elif role == "assistant":
                prompt_parts.append(f'<|im_start|>assistant\n{content}<|im_end|>')
            elif role == "tool":
                prompt_parts.append(f'<|im_start|>tool\n{content}<|im_end|>')
        
        # 添加助手开始标记，启用思考模式
        thinking_content = "" if not self.config.enable_thinking else "<think>\n"
        prompt_parts.append(f'<|im_start|>assistant\n{thinking_content}')
        prompt = "".join(prompt_parts)
        
        # 设置enable_thinking
        return self._generate_with_thinking(prompt, self.config.enable_thinking)
    
    def _generate_with_thinking(self, prompt: str, enable_thinking: bool = False) -> str:
        """生成文本（支持思考模式）"""
        self._result_buffer = []
        
        # 构建输入
        input_data = RKLLMInput()
        input_data.role = b"user"
        input_data.enable_thinking = enable_thinking
        input_data.input_type = RKLLMInputType.PROMPT
        input_data.input.prompt_input = prompt.encode('utf-8')
        
        # 构建推理参数
        infer_param = RKLLMInferParam()
        infer_param.mode = RKLLMInferMode.GENERATE
        infer_param.lora_params = None
        infer_param.prompt_cache_params = None
        infer_param.keep_history = 1
        
        # 运行推理
        ret = self._run(self.handle, ctypes.byref(input_data), ctypes.byref(infer_param), None)
        if ret != 0:
            raise RuntimeError(f"RKLLM推理失败，错误码: {ret}")
        
        return ''.join(self._result_buffer)
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'handle') and self.handle:
            self._destroy(self.handle)
            print("✅ RKLLM资源已释放")


def create_rkllm_from_hf(model_path: str, **kwargs) -> RKLLM:
    """
    从HuggingFace模型路径创建RKLLM
    
    Args:
        model_path: 模型路径（.rkllm文件或目录）
        **kwargs: 其他配置参数
        
    Returns:
        RKLLM实例
    """
    # 查找.rkllm文件
    if os.path.isdir(model_path):
        rkllm_files = [f for f in os.listdir(model_path) if f.endswith('.rkllm')]
        if rkllm_files:
            model_path = os.path.join(model_path, rkllm_files[0])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 自动检测模型类型
    if 'model_type' not in kwargs:
        model_name = os.path.basename(model_path).lower()
        if 'qwen3' in model_name:
            kwargs['model_type'] = MODEL_TYPE_QWEN3
            print(f"检测到Qwen3模型: {model_path}")
        elif 'qwen2' in model_name or 'qwen-2' in model_name:
            kwargs['model_type'] = MODEL_TYPE_QWEN2
            print(f"检测到Qwen2模型: {model_path}")
    
    config = RKLLMConfig(model_path=model_path, **kwargs)
    return RKLLM(config)


# 测试代码
if __name__ == "__main__":
    # 自动查找模型文件
    model_paths = [
        "/home/quark/code/models/Qwen3-1.7B-w8a8-rk3588.rkllm",
        "/home/quark/code/models/qwen2.5-1.5b-instruct_w8a8_rk3588.rkllm",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"❌ 未找到模型文件，请确保以下路径之一存在:")
        for path in model_paths:
            print(f"   - {path}")
        sys.exit(1)
    
    try:
        print(f"\n加载模型: {model_path}")
        llm = create_rkllm_from_hf(model_path, max_new_tokens=128)
        
        # 测试生成
        print("\n测试生成:")
        response = llm.generate("你好，请介绍一下自己。")
        print(f"回复: {response}")
        
        # 测试对话
        print("\n测试对话:")
        messages = [
            {"role": "system", "content": "你是一个 helpful 的AI助手。"},
            {"role": "user", "content": "什么是人工智能？"}
        ]
        response = llm.chat(messages)
        print(f"回复: {response}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
