"""ReAct 输出解析器 — 从 LLM 原始输出中提取 Action 或 FinalAnswer

对应 LangChain 的 AgentOutputParser。
针对小模型做了容错设计：
1. 优先按 ReAct 标准格式解析（Thought / Action / Action Input / Final Answer）。
2. 支持 Markdown 代码块包裹。
3. 如果 LLM 没有输出 Thought，尝试自动补全或兜底。
4. 如果解析失败，返回一个特殊的 AgentFinish 提示用户格式错误。
"""

import logging
import re
from typing import Optional, Union

from zhixia.agent.base import AgentAction, AgentFinish

logger = logging.getLogger(__name__)

# 正则匹配 ReAct 各字段
_RE_THOUGHT = re.compile(r"Thought\s*[:：]\s*(.*?)(?=\n(?:Action|Final Answer)\s*[:：]|$)", re.DOTALL)
_RE_ACTION = re.compile(r"Action\s*[:：]\s*(\S+)")
_RE_ACTION_INPUT = re.compile(r"Action Input\s*[:：]\s*(.*?)(?=\n(?:Observation|Thought|Final Answer)\s*[:：]|$)", re.DOTALL)
_RE_FINAL_ANSWER = re.compile(r"Final Answer\s*[:：]\s*(.*)", re.DOTALL)


class ReActOutputParser:
    """解析 LLM 的 ReAct 格式输出。"""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """解析文本，返回 AgentAction 或 AgentFinish。

        解析策略：
        1. 如果包含 Final Answer → AgentFinish
        2. 如果包含 Action → AgentAction
        3. 都不包含 → 兜底为 AgentFinish（整段文本作为答案）
        """
        text = text.strip()
        if not text:
            return AgentFinish(return_values={"text": "（无输出）"}, log=text)

        # 尝试提取 Final Answer
        final_match = _RE_FINAL_ANSWER.search(text)
        if final_match:
            answer = final_match.group(1).strip()
            # 去除可能包裹的代码块标记
            answer = self._strip_code_blocks(answer)
            return AgentFinish(return_values={"text": answer}, log=text)

        # 尝试提取 Action
        action_match = _RE_ACTION.search(text)
        if action_match:
            tool = action_match.group(1).strip()
            # 提取 Thought
            thought_match = _RE_THOUGHT.search(text)
            thought = thought_match.group(1).strip() if thought_match else ""
            # 提取 Action Input
            input_match = _RE_ACTION_INPUT.search(text)
            tool_input = input_match.group(1).strip() if input_match else ""
            # 去除代码块
            tool_input = self._strip_code_blocks(tool_input)
            return AgentAction(
                tool=tool,
                tool_input=tool_input,
                thought=thought,
                log=text,
            )

        # 兜底：模型没有遵循格式，直接把整段文本当最终答案
        logger.warning("LLM 未遵循 ReAct 格式，直接作为最终答案: %s", text[:100])
        cleaned = self._strip_code_blocks(text)
        return AgentFinish(return_values={"text": cleaned}, log=text)

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """去除 Markdown 代码块标记 ```json ... ```"""
        text = text.strip()
        if text.startswith("```"):
            # 去掉开头的 ``` 和可能的语言标识
            text = re.sub(r"^```\w*\n?", "", text)
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    def get_format_instructions(self) -> str:
        """返回给 prompt 使用的格式说明（可与 ReActPromptTemplate 配合使用）。"""
        return (
            "回复格式:\n"
            "Thought: 你的思考过程\n"
            "Action: 工具名称（不需要工具则直接写 Final Answer）\n"
            "Action Input: 工具的输入\n"
            "或\n"
            "Thought: 你的思考过程\n"
            "Final Answer: 最终回答用户的内容"
        )
