"""听力训练工具 - 提供听力材料和练习建议"""

from typing import Optional
from zhixia.agent.tool import Tool


class ListeningTool(Tool):
    """听力训练工具：提供听力材料推荐和听力技巧指导。

    支持各类英语考试听力训练
    """

    def __init__(self, llm_engine=None):
        super().__init__(
            name="listening_training",
            description="听力训练工具：提供听力材料推荐和听力技巧指导。参数：exam_type（考试类型）、focus_area（重点训练领域，如新闻听力/对话理解/讲座听力）、difficulty（难度级别）",
            func=self._execute,
        )
        self._llm_engine = llm_engine

    def _execute(self, exam_type: str = "四六级", focus_area: str = "综合", difficulty: str = "中等") -> str:
        """执行听力训练指导。

        Args:
            exam_type: 考试类型
            focus_area: 重点训练领域
            difficulty: 难度级别

        Returns:
            听力训练建议和材料推荐
        """
        if self._llm_engine:
            from zhixia.llm.base import LLMMessage
            
            system_prompt = """你是一位专业的英语听力训练专家。请根据学生的需求提供：

1. 适合该考试类型的听力材料推荐
2. 针对性的听力训练方法
3. 听力技巧指导（如预测、抓关键词、记笔记等）
4. 常见听力题型分析
5. 每日/每周听力练习建议
6. 听力能力提升的长期规划

请用中文回答，内容实用、可操作性强。"""

            user_prompt = f"考试类型：{exam_type}\n重点训练领域：{focus_area}\n难度级别：{difficulty}\n\n请为我提供听力训练建议和材料推荐。"

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=1536)

        # 无 LLM 时的回退回答
        return f"【听力训练建议】\n\n考试类型：{exam_type}\n重点领域：{focus_area}\n难度级别：{difficulty}\n\n[提示：LLM引擎未加载，无法生成详细建议。请确保系统配置正确。]"
