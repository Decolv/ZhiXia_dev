"""云端LLM引擎（OpenAI API格式）"""

import json
import logging
from typing import Generator, List

from zhixia.config.settings import LLMConfig
from zhixia.llm.base import LLMEngine, LLMMessage

logger = logging.getLogger(__name__)


class CloudLLMEngine(LLMEngine):
    """支持OpenAI API格式的云端LLM引擎"""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._system_prompt = config.system_prompt

    @property
    def name(self) -> str:
        return "cloud"

    def _get_headers(self) -> dict:
        """获取API请求头"""
        headers = {
            "Content-Type": "application/json",
        }
        if self._config.cloud_api_key:
            headers["Authorization"] = f"Bearer {self._config.cloud_api_key}"
        return headers

    def _build_request_body(
        self,
        messages: List[LLMMessage],
        max_new_tokens: int,
        stream: bool = False,
    ) -> dict:
        """构建API请求体"""
        # 转换消息格式
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        body = {
            "model": self._config.cloud_model_name,
            "messages": api_messages,
            "max_tokens": max_new_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "stream": stream,
        }
        return body

    def chat(self, messages: List[LLMMessage], max_new_tokens: int = 32) -> str:
        """非流式调用云端API"""
        try:
            import urllib.request
            import urllib.error

            body = self._build_request_body(messages, max_new_tokens, stream=False)
            data = json.dumps(body).encode("utf-8")

            req = urllib.request.Request(
                self._config.cloud_api_url,
                data=data,
                headers=self._get_headers(),
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                logger.info("云端LLM输出: %s", content[:200])
                return content.strip()

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error("云端API HTTP错误: %s - %s", e.code, error_body)
            raise RuntimeError(f"云端API调用失败: HTTP {e.code}")
        except Exception as exc:
            logger.exception("云端API调用异常")
            raise RuntimeError(f"云端API调用失败: {exc}")

    def stream_chat(
        self, messages: List[LLMMessage], max_new_tokens: int = 32
    ) -> Generator[str, None, None]:
        """流式调用云端API"""
        try:
            import urllib.request

            body = self._build_request_body(messages, max_new_tokens, stream=True)
            data = json.dumps(body).encode("utf-8")

            req = urllib.request.Request(
                self._config.cloud_api_url,
                data=data,
                headers=self._get_headers(),
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except Exception as exc:
            logger.exception("云端API流式调用异常")
            raise RuntimeError(f"云端API流式调用失败: {exc}")

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self._system_prompt = prompt

    def shutdown(self) -> None:
        """释放资源"""
        pass
