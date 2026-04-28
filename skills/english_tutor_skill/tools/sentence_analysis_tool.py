"""长难句解析工具 - 分析复杂句子结构"""

from typing import Optional
from zhixia.agent.tool import Tool


class SentenceAnalysisTool(Tool):
    """长难句解析工具：分析复杂句子结构，帮助理解语法和逻辑。

    适用于阅读理解中的长难句分析
    """

    def __init__(self, llm_engine=None):
        super().__init__(
            name="sentence_analysis",
            description="长难句解析工具：分析复杂英语句子的结构、语法和含义。参数：sentence（需要分析的句子）",
            func=self._execute,
        )
        self._llm_engine = llm_engine

    def _execute(self, sentence: str) -> str:
        """执行长难句解析。

        Args:
            sentence: 需要分析的英语句子

        Returns:
            句子结构分析和翻译
        """
        if self._llm_engine:
            from zhixia.llm.base import LLMMessage
            
            system_prompt = """你是一位专业的英语语法分析专家。请对提供的长难句进行详细分析：

分析内容包括：
1. 句子主干提取（主谓宾/主系表结构）
2. 从句类型识别（定语从句、状语从句、名词性从句等）
3. 非谓语动词分析
4. 特殊语法结构说明
5. 句子逻辑关系梳理
6. 准确的中文翻译
7. 重点词汇和短语解析
8. 同类句型拓展（可选）

请用中文回答，条理清晰，便于理解。"""

            user_prompt = f"请分析以下句子：\n\n{sentence}"

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=1536)

        # 无 LLM 时的回退回答
        return f"【长难句解析】\n\n句子：{sentence}\n\n[提示：LLM引擎未加载，无法生成详细分析。请确保系统配置正确。]"
