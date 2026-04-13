"""RKLLM NPU 推理引擎（包装根目录 rkllm_inference.py）"""

import logging
import os
import sys
from pathlib import Path
from typing import Generator, List, Optional

from zhixia.config.settings import LLMConfig
from zhixia.llm.base import LLMEngine, LLMMessage

logger = logging.getLogger(__name__)

# rkllm_inference.py 在项目根目录，需要将根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class RKLLMEngine(LLMEngine):

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._llm = None

    @property
    def name(self) -> str:
        return "rkllm"

    def _ensure_model(self) -> None:
        if self._llm is not None:
            return
        from rkllm_inference import create_rkllm_from_hf

        model_path = self._config.model_path
        if not os.path.isabs(model_path):
            model_path = str(_PROJECT_ROOT / model_path)

        logger.info("加载 RKLLM 模型: %s", model_path)
        self._llm = create_rkllm_from_hf(
            model_path,
            max_context_len=self._config.max_context_len,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
        )
        self._llm.set_chat_template(
            system_prompt=self._config.system_prompt,
            prompt_prefix="",
            prompt_postfix="",
        )
        logger.info("RKLLM 模型加载完成")

    def chat(self, messages: List[LLMMessage], max_new_tokens: int = 32) -> str:
        self._ensure_model()
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        raw = self._llm.chat(msg_dicts, max_new_tokens=max_new_tokens)
        logger.info("LLM 原始输出: %s", raw[:200])
        return raw.strip()

    def stream_chat(self, messages: List[LLMMessage], max_new_tokens: int = 32) -> Generator[str, None, None]:
        """流式输出 token，逐 token yield。"""
        self._ensure_model()
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        for token in self._llm.stream_chat(msg_dicts, max_new_tokens=max_new_tokens):
            yield token

    def set_system_prompt(self, prompt: str) -> None:
        self._config.system_prompt = prompt
        if self._llm is not None:
            self._llm.set_chat_template(system_prompt=prompt)

    def shutdown(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            from zhixia.utils.memory import force_gc
            force_gc()
