"""词汇学习工具 - 词汇记忆策略和方法"""

from typing import Optional
from zhixia.agent.tool import Tool


class VocabularyTool(Tool):
    """词汇学习工具：提供词汇记忆策略、词根词缀分析和联想记忆方法。

    帮助高效记忆英语词汇
    """

    def __init__(self, llm_engine=None):
        super().__init__(
            name="vocabulary_learning",
            description="词汇学习工具：提供词汇记忆策略、词根词缀分析、联想记忆方法。参数：word（单词或词组）、context（使用场景，如考试类型）、method（记忆方法偏好，如词根/联想/语境）",
            func=self._execute,
        )
        self._llm_engine = llm_engine

    def _execute(self, word: str, context: str = "通用", method: str = "综合") -> str:
        """执行词汇学习指导。

        Args:
            word: 需要学习的单词或词组
            context: 使用场景/考试类型
            method: 偏好的记忆方法

        Returns:
            词汇学习内容和记忆建议
        """
        if self._llm_engine:
            from zhixia.llm.base import LLMMessage
            
            system_prompt = """你是一位专业的词汇教学专家。请对提供的单词进行详细讲解：

讲解内容包括：
1. 单词音标和发音提示
2. 核心词义及用法（含英文释义）
3. 词根词缀分析（如有）
4. 同义词/反义词对比
5. 常见搭配和短语
6. 例句展示（2-3个不同语境）
7. 记忆技巧（联想记忆、词根记忆、谐音记忆等）
8. 易混淆词辨析（如有）
9. 在各类考试中的出现频率和常见考点

请用中文回答，内容详实、易于记忆。"""

            user_prompt = f"单词/词组：{word}\n使用场景：{context}\n记忆方法偏好：{method}\n\n请详细讲解这个单词。"

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=1536)

        # 无 LLM 时的回退回答
        return f"【词汇学习】\n\n单词：{word}\n场景：{context}\n方法：{method}\n\n[提示：LLM引擎未加载，无法生成详细讲解。请确保系统配置正确。]"
