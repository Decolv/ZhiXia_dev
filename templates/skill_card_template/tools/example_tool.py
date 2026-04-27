"""示例工具 —— 展示 Tool 基类的正确用法

注意：工具只应导入 zhixia.* 公共接口，不依赖具体卡片路径。
"""

from typing import Optional
from zhixia.agent.tool import Tool


class ExampleTool(Tool):
    """示例工具：根据用户输入返回示例回答。

    实际开发中，工具可以：
    - 调用外部 API（天气、股票、翻译等）
    - 使用 LLM 引擎基于知识生成答案
    - 操作硬件（GPIO、传感器等）
    - 读写文件或数据库
    """

    def __init__(self, llm_engine=None):
        super().__init__(
            name="example_tool",
            description="示例工具：返回一条示例信息。参数：query（任意文本）",
            func=self._execute,
        )
        self._llm_engine = llm_engine

    def _execute(self, query: str) -> str:
        """执行工具逻辑。

        Args:
            query: 用户输入的查询文本

        Returns:
            工具执行结果文本
        """
        if self._llm_engine:
            # 使用 LLM 智能生成答案（推荐方式）
            from zhixia.llm.base import LLMMessage
            messages = [
                LLMMessage(role="system", content="你是示例助手，用一句话回答。"),
                LLMMessage(role="user", content=query),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=64)

        # 无 LLM 时的回退回答
        return f"收到查询：{query}。这是一个示例工具的回答。"
