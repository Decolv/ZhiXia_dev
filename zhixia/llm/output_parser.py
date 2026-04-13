"""LLM 输出解析器 — 从原始输出中提取结构化信息（文本 + 情绪 + 元数据）

对 1.7B 小模型做了容错设计：
1. 剥离 Qwen3 <think...</think 思考标签
2. 尝试整体 JSON 解析
3. 尝试正则提取 JSON
4. 检查 [emotion:xxx] 前缀约定
5. 兜底：整段文本作为 text，emotion="neutral"
"""

import json
import logging
import re
from typing import Optional

from zhixia.llm.base import StructuredOutput

logger = logging.getLogger(__name__)

EMOTION_LABELS = {"neutral", "happy", "sad", "thinking", "surprised", "worried", "angry"}

_JSON_KEY_PATTERN = re.compile(r"\{[^{}]*\}")
_FORMAT_INSTRUCTION = (
    "\n\n输出格式要求：请以JSON格式回复，格式如下：\n"
    '{"emotion": "情绪标签", "text": "你的回答文本"}\n'
    "情绪标签可选值：neutral, happy, sad, thinking, surprised, worried, angry\n"
    "如果无法判断情绪，使用neutral。"
)


def get_format_instruction() -> str:
    """当 enable_structured_output 开启时追加到 system prompt 的格式指令"""
    return _FORMAT_INSTRUCTION


def parse_llm_output(raw: str) -> StructuredOutput:
    """解析 LLM 原始输出为 StructuredOutput"""
    # 1. 剥离思考标签
    cleaned = _strip_thinking_tokens(raw)

    # 2. 尝试整体 JSON
    result = _try_parse_json(cleaned)
    if result is not None:
        return result

    # 3. 尝试从文本中提取 JSON
    match = _JSON_KEY_PATTERN.search(cleaned)
    if match:
        result = _try_parse_json(match.group())
        if result is not None:
            return result

    # 4. 检查 [emotion:xxx] 前缀
    result = _try_parse_emotion_prefix(cleaned)
    if result is not None:
        return result

    # 5. 兜底
    return StructuredOutput(text=cleaned.strip(), emotion="neutral")


def _strip_thinking_tokens(text: str) -> str:
    return re.sub(r"<think.*?</think\s*>", "", text, flags=re.DOTALL).strip()


def _try_parse_json(text: str) -> Optional[StructuredOutput]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    text_content = str(data.get("text", ""))
    if not text_content:
        text_content = text.strip()

    emotion = str(data.get("emotion", "neutral")).lower().strip()
    if emotion not in EMOTION_LABELS:
        emotion = "neutral"

    metadata = {k: v for k, v in data.items() if k not in ("text", "emotion")}
    return StructuredOutput(text=text_content, emotion=emotion, metadata=metadata)


def _try_parse_emotion_prefix(text: str) -> Optional[StructuredOutput]:
    lines = text.strip().split("\n")
    if not lines:
        return None
    match = re.match(r"^\[emotion:(\w+)\]", lines[0].strip())
    if match:
        emotion = match.group(1).lower()
        if emotion not in EMOTION_LABELS:
            emotion = "neutral"
        remaining = "\n".join(lines[1:]).strip()
        return StructuredOutput(text=remaining, emotion=emotion)
    return None
