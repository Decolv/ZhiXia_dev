"""端到端测试：验证 KIMI / Moonshot API 在项目 LLM 链路中的可用性（不依赖音频硬件）

用法:
    export KIMI_API_KEY="sk-xxxxxxxx"
    python test_kimi_pipeline.py
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zhixia.config.settings import LLMConfig
from zhixia.llm.cloud_engine import CloudLLMEngine
from zhixia.llm.base import LLMMessage
from zhixia.llm.output_parser import parse_llm_output, get_format_instruction


def test_kimi_llm_pipeline():
    api_key = os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        print("错误: 请设置 KIMI_API_KEY 环境变量")
        sys.exit(1)

    config = LLMConfig(
        engine="cloud",
        cloud_api_url="https://api.moonshot.cn/v1/chat/completions",
        cloud_api_key=api_key,
        cloud_model_name="moonshot-v1-8k",
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
        system_prompt="你是「小匣」，一个温暖有趣的智能助手。用一句话简短回答用户的问题。",
    )

    engine = CloudLLMEngine(config)

    messages = [
        LLMMessage(role="system", content=config.system_prompt),
        LLMMessage(role="user", content="你好，请自我介绍一下"),
    ]

    print("=" * 60)
    print("测试 1: 非流式 chat()")
    print("-" * 60)
    try:
        response = engine.chat(messages, max_new_tokens=128)
        print(f"原始输出: {response[:200]}...")
        parsed = parse_llm_output(response)
        print(f"解析文本: {parsed.text[:200]}...")
        print(f"情感标签: {parsed.emotion}")
        print("[PASS] 非流式调用成功\n")
    except Exception as e:
        print(f"[FAIL] 非流式调用失败: {e}\n")
        return False

    print("=" * 60)
    print("测试 2: 流式 stream_chat()")
    print("-" * 60)
    try:
        tokens = []
        for token in engine.stream_chat(messages, max_new_tokens=128):
            tokens.append(token)
            print(token, end="", flush=True)
        full_response = "".join(tokens)
        print("\n")
        parsed = parse_llm_output(full_response)
        print(f"解析文本: {parsed.text[:200]}...")
        print(f"情感标签: {parsed.emotion}")
        print(f"Token 数量: {len(tokens)}")
        print("[PASS] 流式调用成功\n")
    except Exception as e:
        print(f"\n[FAIL] 流式调用失败: {e}\n")
        return False

    print("=" * 60)
    print("测试 3: 结构化输出（JSON 模式）")
    print("-" * 60)
    config.enable_structured_output = True
    config.system_prompt = (
        "你是「小匣」，一个温暖有趣的智能助手。"
        + get_format_instruction()
    )
    engine = CloudLLMEngine(config)

    messages_json = [
        LLMMessage(role="system", content=config.system_prompt),
        LLMMessage(role="user", content="今天天气怎么样？"),
    ]
    try:
        tokens = []
        for token in engine.stream_chat(messages_json, max_new_tokens=128):
            tokens.append(token)
        full_response = "".join(tokens)
        print(f"原始输出: {full_response[:300]}...")
        parsed = parse_llm_output(full_response)
        print(f"解析文本: {parsed.text[:200]}...")
        print(f"情感标签: {parsed.emotion}")
        print(f"元数据: {parsed.metadata}")
        print("[PASS] 结构化输出测试成功\n")
    except Exception as e:
        print(f"[FAIL] 结构化输出测试失败: {e}\n")
        return False

    print("=" * 60)
    print("全部测试通过！KIMI API 可完美接入项目。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_kimi_llm_pipeline()
    sys.exit(0 if success else 1)
