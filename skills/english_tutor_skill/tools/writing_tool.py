"""作文辅导工具 - 写作指导和范文分析"""

from typing import Optional
from zhixia.agent.tool import Tool


class WritingTool(Tool):
    """作文辅导工具：提供写作思路、范文分析和作文修改建议。

    支持各类英语考试写作辅导
    """

    def __init__(self, llm_engine=None):
        super().__init__(
            name="writing_coaching",
            description="作文辅导工具：提供写作思路、范文分析、作文修改建议。参数：task_type（写作任务类型，如议论文/图表作文/书信等）、topic（作文主题/题目）、essay_content（作文内容，用于批改时提供）",
            func=self._execute,
        )
        self._llm_engine = llm_engine

    def _execute(self, task_type: str = "议论文", topic: str = "", essay_content: str = "") -> str:
        """执行作文辅导。

        Args:
            task_type: 写作任务类型
            topic: 作文主题或题目
            essay_content: 学生写的作文内容（用于批改）

        Returns:
            写作指导或作文批改建议
        """
        if self._llm_engine:
            from zhixia.llm.base import LLMMessage
            
            if essay_content:
                # 作文批改模式
                system_prompt = """你是一位专业的英语写作批改老师。请对学生的作文进行详细批改：

批改内容包括：
1. 总体评价（优点和主要问题）
2. 分数预估（按考试标准）
3. 语法错误纠正（逐句标注）
4. 词汇使用建议（升级替换）
5. 句式多样性评价和改进建议
6. 逻辑结构分析
7. 内容深度评价
8. 修改后的范文参考
9. 针对性提升建议

请用中文回答，既指出问题也给予鼓励。"""
                user_prompt = f"写作任务类型：{task_type}\n题目：{topic}\n\n学生作文：\n{essay_content}\n\n请对这篇作文进行详细批改。"
            else:
                # 写作指导模式
                system_prompt = """你是一位专业的英语写作指导老师。请为学生提供详细的写作指导：

指导内容包括：
1. 该类型作文的写作框架和结构
2. 审题要点和思路拓展
3. 开头、主体、结尾的写作技巧
4. 高分句型和过渡词推荐
5. 常见错误提醒
6. 参考范文（含中文翻译）
7. 写作练习建议

请用中文回答，内容实用、易于操作。"""
                user_prompt = f"写作任务类型：{task_type}\n题目：{topic}\n\n请为我提供写作指导。"

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=2048)

        # 无 LLM 时的回退回答
        mode = "批改" if essay_content else "指导"
        return f"【作文{mode}】\n\n任务类型：{task_type}\n题目：{topic}\n\n[提示：LLM引擎未加载，无法生成详细内容。请确保系统配置正确。]"
