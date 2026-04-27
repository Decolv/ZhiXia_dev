"""ToolCallingAgent — 结构化工具调用 Agent

这是现代 LLM（OpenAI、KIMI、Qwen3 等）推荐的方式：
1. 将工具描述转为 JSON Schema，通过系统消息或专用字段传给 LLM。
2. LLM 输出结构化的 tool_call 对象（而非自由文本 ReAct）。
3. 解析器直接提取 tool_call，无需复杂的正则匹配。

相比传统 ReAct：
- ✅ 更可靠：结构化输出避免格式解析错误。
- ✅ 更快：LLM 不需要生成冗长的 Thought/Action 文本。
- ✅ 更省 token：工具调用是紧凑的 JSON。
- ⚠️ 要求 LLM 原生支持 function calling / tool_call。

实现策略：
- CloudLLM (KIMI)：原生支持 tool_choice + tools 参数。
- RKLLM (Qwen3)：Qwen3-1.7B 支持 function calling，可通过 chat_template 注入工具描述。
- 兜底：如果 LLM 不支持 tool_call，自动回退到 ReAct 文本模式（由上层决定）。

本模块实现：
    ToolCallingAgent     — 结构化工具调用决策
    ToolCallParser       — 从 LLM 输出解析 tool_call
    BoundLLM             — 绑定了工具的 LLM 包装器
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from zhixia.agent.base import (
    AgentAction,
    AgentDecision,
    AgentFinish,
    BaseAgent,
    BaseTool,
)
from zhixia.agent.tool import ToolRegistry
from zhixia.llm.base import LLMEngine, LLMMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolSchemaBuilder — 将 BaseTool 转为 JSON Schema
# ---------------------------------------------------------------------------

class ToolSchemaBuilder:
    """构建符合 OpenAI function calling 规范的 JSON Schema。"""

    @staticmethod
    def build(tool: BaseTool) -> Dict[str, Any]:
        """将单个工具转为 function schema。"""
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        # 如果工具有 args_schema，尝试提取参数
        if tool.args_schema is not None:
            schema["function"]["parameters"] = ToolSchemaBuilder._schema_from_pydantic(
                tool.args_schema
            )
        return schema

    @staticmethod
    def build_all(tools: ToolRegistry) -> List[Dict[str, Any]]:
        """将注册表中的所有工具转为 schema 列表。"""
        return [ToolSchemaBuilder.build(t) for t in tools.list_tools()]

    @staticmethod
    def _schema_from_pydantic(model_cls: type) -> Dict[str, Any]:
        """从 Pydantic 模型或 dataclass 提取 JSON Schema。

        简单实现：遍历 __annotations__ 提取字段名和类型。
        如果安装了 pydantic，优先使用 model_json_schema()。
        """
        try:
            # 尝试 Pydantic v2
            if hasattr(model_cls, "model_json_schema"):
                return model_cls.model_json_schema()
            # 尝试 Pydantic v1
            if hasattr(model_cls, "schema"):
                return model_cls.schema()
        except Exception:
            pass

        # 兜底：从 type hints 推导
        import typing

        properties = {}
        required = []
        hints = getattr(model_cls, "__annotations__", {})
        for name, hint in hints.items():
            if name.startswith("_"):
                continue
            field_info = {"type": "string"}
            # 简单类型映射
            origin = typing.get_origin(hint)
            if origin is typing.Union:
                args = typing.get_args(hint)
                if type(None) in args:
                    # Optional[T]
                    real_type = [a for a in args if a is not type(None)][0]
                    field_info["type"] = ToolSchemaBuilder._python_type_to_json(real_type)
                else:
                    field_info["type"] = "string"
            else:
                field_info["type"] = ToolSchemaBuilder._python_type_to_json(hint)
                required.append(name)
            properties[name] = field_info

        return {"type": "object", "properties": properties, "required": required}

    @staticmethod
    def _python_type_to_json(t: type) -> str:
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return mapping.get(t, "string")


# ---------------------------------------------------------------------------
# ToolCallParser — 从 LLM 输出解析 tool_call
# ---------------------------------------------------------------------------

class ToolCallParser:
    """解析 LLM 的 tool_call 输出。

    支持多种格式：
    1. OpenAI 原生格式：choices[0].message.tool_calls
    2. Qwen / KIMI 兼容格式：assistant 消息中包含 function call
    3. 文本 JSON 格式：assistant 消息内容是一个 JSON 对象
    """

    @staticmethod
    def parse_from_text(text: str) -> Optional[AgentAction]:
        """尝试从纯文本中提取 tool_call JSON。"""
        text = text.strip()
        if not text:
            return None

        # 尝试解析整个文本为 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取文本中的 JSON 块
            data = ToolCallParser._extract_json_block(text)

        if not isinstance(data, dict):
            return None

        # 格式 1: {"name": "tool_name", "arguments": {...}}
        if "name" in data or "function" in data:
            return ToolCallParser._parse_openai_format(data)

        # 格式 2: {"tool": "name", "params": {...}}
        if "tool" in data:
            return AgentAction(
                tool=data.get("tool", ""),
                tool_input=json.dumps(data.get("params", data.get("arguments", ""))),
                thought=data.get("thought", ""),
                log=text,
            )

        return None

    @staticmethod
    def _extract_json_block(text: str) -> Optional[Dict]:
        """从文本中提取 ```json ... ``` 代码块。"""
        import re

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试直接匹配 { ... }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _parse_openai_format(data: Dict) -> Optional[AgentAction]:
        """解析 OpenAI function call 格式。"""
        if "function" in data:
            func = data["function"]
            name = func.get("name", "")
            arguments = func.get("arguments", "")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            return AgentAction(tool=name, tool_input=arguments, thought="", log="")

        if "name" in data:
            name = data.get("name", "")
            arguments = data.get("arguments", data.get("params", ""))
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            return AgentAction(tool=name, tool_input=arguments, thought="", log="")

        return None


# ---------------------------------------------------------------------------
# BoundLLM — 绑定了工具的 LLM 包装器
# ---------------------------------------------------------------------------

class BoundLLM:
    """给 LLM 绑定工具描述，使其支持结构化工具调用。

    这是 LangChain `llm.bind_tools([...])` 的等价实现。

    对于不同 LLM 后端：
    - CloudLLM：在 API 请求体中加入 tools + tool_choice 字段。
    - RKLLM：通过 chat_template 注入工具描述，或使用 Qwen 的 function call 模板。

    使用方式：
        bound = BoundLLM(llm_engine, tools)
        response = bound.chat(messages)
        # response 中可能包含 tool_calls
    """

    def __init__(self, llm: LLMEngine, tools: ToolRegistry) -> None:
        self.llm = llm
        self.tools = tools
        self._schemas = ToolSchemaBuilder.build_all(tools)
        self._tool_names = {t.name for t in tools.list_tools()}

    @property
    def name(self) -> str:
        return f"BoundLLM({self.llm.name})"

    def chat(self, messages: List[LLMMessage], max_new_tokens: int = 256) -> str:
        """非流式调用，尝试传入 tools 参数。

        如果底层 LLM 不支持 tools（如 RKLLM 老版本），
        则回退到在 system prompt 中注入工具描述。
        """
        try:
            return self._chat_with_tools(messages, max_new_tokens)
        except Exception as exc:
            logger.warning("结构化工具调用失败，回退到文本模式: %s", exc)
            return self._chat_with_text_tools(messages, max_new_tokens)

    def stream_chat(
        self, messages: List[LLMMessage], max_new_tokens: int = 256
    ):
        """流式调用。回退到普通 stream_chat。"""
        # 流式 tool call 比较复杂，先回退到普通流式
        for token in self.llm.stream_chat(messages, max_new_tokens):
            yield token

    def _chat_with_tools(self, messages: List[LLMMessage], max_new_tokens: int) -> str:
        """尝试用原生 tools 参数调用。"""
        # 检查 LLM 是否支持工具绑定
        if hasattr(self.llm, "chat_with_tools"):
            return self.llm.chat_with_tools(
                messages, self._schemas, max_new_tokens=max_new_tokens
            )
        # 否则回退
        raise NotImplementedError("底层 LLM 不支持原生 tool calling")

    def _chat_with_text_tools(self, messages: List[LLMMessage], max_new_tokens: int) -> str:
        """在 system prompt 中注入工具描述，使用文本模式调用。"""
        tool_desc = self.tools.format_tool_descriptions()
        schema_text = json.dumps(self._schemas, ensure_ascii=False, indent=2)

        inject = (
            f"你有以下工具可用:\n{tool_desc}\n\n"
            f"工具参数格式 (JSON Schema):\n{schema_text}\n\n"
            "当你需要调用工具时，请只回复一个 JSON 对象，格式如下:\n"
            '{"name": "工具名", "arguments": {"参数名": "参数值"}}\n'
            "不需要工具时，直接回复用户的自然语言问题。"
        )

        # 修改第一条 system 消息，或插入新的 system 消息
        new_messages = []
        injected = False
        for msg in messages:
            if msg.role == "system" and not injected:
                new_messages.append(
                    LLMMessage(role="system", content=msg.content + "\n\n" + inject)
                )
                injected = True
            else:
                new_messages.append(msg)
        if not injected:
            new_messages.insert(0, LLMMessage(role="system", content=inject))

        return self.llm.chat(new_messages, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# ToolCallingAgent — 结构化工具调用 Agent
# ---------------------------------------------------------------------------

class ToolCallingAgent(BaseAgent):
    """基于结构化工具调用的 Agent。

    适合模型：KIMI (moonshot-v1)、Qwen3-1.7B+、GPT-4 等支持 function calling 的模型。

    Args:
        llm_engine: 支持 tool calling 的 LLM。
        tools: 工具注册表。
        max_new_tokens: 每次决策的最大 token 数。
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        tools: ToolRegistry,
        max_new_tokens: int = 256,
    ) -> None:
        self.llm = llm_engine
        self.tools = tools
        self.max_new_tokens = max_new_tokens
        self._bound_llm: Optional[BoundLLM] = None
        self._parser = ToolCallParser()

    @property
    def name(self) -> str:
        return "ToolCallingAgent"

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def return_values(self) -> List[str]:
        return ["text"]

    def _get_bound_llm(self) -> BoundLLM:
        if self._bound_llm is None:
            self._bound_llm = BoundLLM(self.llm, self.tools)
        return self._bound_llm

    def plan(
        self,
        intermediate_steps: List[Any],
        callbacks=None,
        **kwargs: Any,
    ) -> AgentDecision:
        """Agent 决策：调用绑定了工具的 LLM，解析输出。

        流程：
        1. 将 intermediate_steps 格式化为消息历史。
        2. 调用 BoundLLM（自动注入工具描述）。
        3. 解析输出：是 tool_call JSON → AgentAction；是普通文本 → AgentFinish。
        """
        messages = kwargs.get("messages", [])
        user_input = kwargs.get("input", "")

        if not messages:
            # 兜底：构建基本消息
            from zhixia.llm.base import LLMMessage

            messages = [LLMMessage(role="user", content=user_input)]

        bound = self._get_bound_llm()
        response = bound.chat(messages, max_new_tokens=self.max_new_tokens)

        # 先尝试解析为 tool_call
        action = self._parser.parse_from_text(response)
        if action is not None and action.tool in {t.name for t in self.tools.list_tools()}:
            # 补充 thought（如果有原始文本中的思考内容）
            if not action.thought:
                action.thought = f"我需要调用 {action.tool} 来获取信息。"
            return action

        # 不是 tool_call，作为最终答案
        cleaned = response.strip()
        # 如果回复是 JSON 但工具不存在，也作为最终答案
        if action is not None and action.tool not in {t.name for t in self.tools.list_tools()}:
            cleaned = f"抱歉，我没有 '{action.tool}' 这个工具。让我直接回答：{cleaned}"

        return AgentFinish(return_values={"text": cleaned}, log=response)
