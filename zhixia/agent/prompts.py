"""Agent Prompt 模板 — ReAct 风格

LangChain 中 prompt 通常是 PromptTemplate + partial variables。
这里为了轻量，用简单的字符串模板 + format 方法。

模板设计针对中文小模型（1~3B）优化：
1. 指令明确、格式简洁，避免过度复杂的 JSON schema。
2. 提供清晰的思考-行动-观察示例。
3. Final Answer 直接用中文回复，不需要额外包装。
"""

from typing import Any, Dict, List, Optional

from zhixia.agent.base import AgentStep


class ReActPromptTemplate:
    """ReAct 风格的 Prompt 模板。

    组装后的 prompt 结构：
        1. System / 角色设定
        2. 工具列表 + 使用格式说明
        3. Few-shot 示例（可选）
        4. 对话历史 / Scratchpad（Thought/Action/Observation）
        5. 用户当前问题

    使用方式：
        template = ReActPromptTemplate(system_prompt="你是助手...")
        prompt = template.format(
            tool_descriptions="...",
            input="今天北京天气？",
            agent_scratchpad="...",
        )
    """

    # 默认 ReAct 格式指令（中文，面向小模型优化）
    DEFAULT_FORMAT_INSTRUCTIONS = (
        "你可以使用以下工具来帮助回答用户问题。\n"
        "工具列表:\n"
        "{tool_descriptions}\n\n"
        "使用工具时，请严格按照以下格式回复：\n"
        "Thought: 我需要...（你的思考过程）\n"
        "Action: 工具名称\n"
        "Action Input: 工具的输入参数\n\n"
        "工具会返回 Observation（观察结果），你可以继续思考并决定下一步。\n"
        "如果你已经知道答案，或不需要工具，直接回复：\n"
        "Thought: 我已经知道答案\n"
        "Final Answer: 你的最终回答\n\n"
        "注意：\n"
        "- 每次只能调用一个工具。\n"
        "- Action Input 尽量简洁。\n"
        "- 如果工具返回错误，尝试换个方式提问或告诉用户无法完成。\n"
    )

    # 可选：Few-shot 示例（帮助小模型理解格式）
    DEFAULT_EXAMPLES = (
        "示例 1:\n"
        "Question: 上海现在多少度？\n"
        "Thought: 用户想知道上海的温度，我需要查询天气工具。\n"
        "Action: weather\n"
        "Action Input: 上海\n"
        "Observation: 上海今天多云，26°C，东南风2级。\n"
        "Thought: 我已经获得了天气信息，可以直接回答用户。\n"
        "Final Answer: 上海今天多云，气温26°C，东南风2级，体感比较舒适。\n\n"
        "示例 2:\n"
        "Question: 你好\n"
        "Thought: 用户在打招呼，不需要调用工具。\n"
        "Final Answer: 你好呀！我是小匣，有什么可以帮你的吗？\n\n"
    )

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        format_instructions: Optional[str] = None,
        examples: Optional[str] = None,
        suffix: str = "Question: {input}\n\n{agent_scratchpad}",
    ) -> None:
        """
        Args:
            system_prompt: 角色设定。为空时使用默认助手设定。
            format_instructions: 工具使用格式说明。为空使用默认 ReAct 指令。
            examples: Few-shot 示例。为空使用默认示例。
            suffix: prompt 末尾模板，需包含 {input} 和 {agent_scratchpad} 占位符。
        """
        self.system_prompt = system_prompt or (
            "你是「小匣」，一个温暖有趣的智能助手。"
            "当用户的问题需要外部信息时，你可以使用工具来获取信息并给出准确回答。"
        )
        self.format_instructions = format_instructions or self.DEFAULT_FORMAT_INSTRUCTIONS
        self.examples = examples or self.DEFAULT_EXAMPLES
        self.suffix = suffix

    def format(
        self,
        tool_descriptions: str,
        input: str,
        agent_scratchpad: str,
        **kwargs: Any,
    ) -> str:
        """组装完整 prompt。"""
        parts = [
            self.system_prompt,
            "",
            self.format_instructions.format(tool_descriptions=tool_descriptions),
            "",
            self.examples,
            "",
            self.suffix.format(input=input, agent_scratchpad=agent_scratchpad),
        ]
        return "\n".join(parts)

    @staticmethod
    def build_scratchpad(steps: List[AgentStep]) -> str:
        """将 AgentStep 列表格式化为 scratchpad 字符串。"""
        lines = []
        for step in steps:
            action = step.action
            lines.append(f"Thought: {action.thought}")
            lines.append(f"Action: {action.tool}")
            lines.append(f"Action Input: {action.tool_input}")
            lines.append(f"Observation: {step.observation}")
            lines.append("")
        return "\n".join(lines)
