"""考试规划工具 - 制定个性化备考计划"""

from typing import Optional
from zhixia.agent.tool import Tool


class ExamPlanningTool(Tool):
    """考试规划工具：根据考试类型、时间和基础制定备考计划。

    支持考试类型：四六级、雅思、托福、考研英语等
    """

    def __init__(self, llm_engine=None):
        super().__init__(
            name="exam_planning",
            description="考试规划工具：制定个性化英语考试备考计划。参数：exam_type（考试类型，如四六级/雅思/托福/考研）、time_remaining（剩余时间，如3个月）、current_level（当前水平，如初级/中级/高级）",
            func=self._execute,
        )
        self._llm_engine = llm_engine

    def _execute(self, exam_type: str, time_remaining: str = "3个月", current_level: str = "中级") -> str:
        """执行考试规划。

        Args:
            exam_type: 考试类型（四六级/雅思/托福/考研等）
            time_remaining: 剩余备考时间
            current_level: 当前英语水平（初级/中级/高级）

        Returns:
            个性化备考计划
        """
        if self._llm_engine:
            from zhixia.llm.base import LLMMessage
            
            system_prompt = f"""你是一位专业的英语考试规划专家。请根据学生的具体情况制定详细的备考计划。

请提供：
1. 总体备考策略
2. 分阶段学习计划（按周或月划分）
3. 各科目（听力、阅读、写作、翻译/口语）的时间分配
4. 推荐的学习资料和资源
5. 阶段性目标设定
6. 备考注意事项和建议

请用中文回答，结构清晰，内容实用。"""

            user_prompt = f"考试类型：{exam_type}\n剩余时间：{time_remaining}\n当前水平：{current_level}\n\n请为我制定一份详细的备考计划。"

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=2048)

        # 无 LLM 时的回退回答
        return f"【{exam_type}备考计划】\n\n剩余时间：{time_remaining}\n当前水平：{current_level}\n\n[提示：LLM引擎未加载，无法生成详细计划。请确保系统配置正确。]"
