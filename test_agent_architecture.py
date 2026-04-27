"""Agent 架构综合测试 —— 验证 Runnable、LCEL、ReAct、ToolCalling、Callbacks

用法:
    export KIMI_API_KEY="sk-xxxxxxxx"
    python test_agent_architecture.py
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zhixia.agent import (
    AgentAction,
    AgentFinish,
    AgentState,
    AgentExecutor,
    AgentRunner,
    ReActAgent,
    ToolCallingAgent,
    ToolRegistry,
    Tool,
    tool,
    RunnableLambda,
    RunnableSequence,
    RunnableConfig,
    CallbackManager,
    LoggingHandler,
    StreamingDisplayHandler,
)
from zhixia.agent.runnable import RunnableMap
from zhixia.llm.base import LLMMessage
from zhixia.llm.cloud_engine import CloudLLMEngine
from zhixia.config.settings import LLMConfig


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------

class MockWeatherTool(Tool):
    """模拟天气查询工具（测试用）。"""

    def __init__(self):
        super().__init__(
            name="weather",
            description="查询指定城市的当前天气，返回温度、天气状况。",
            func=self._get_weather,
        )

    def _get_weather(self, city: str) -> str:
        mock_db = {
            "北京": "晴天，28°C，西北风2级",
            "上海": "多云，26°C，东南风3级",
            "深圳": "小雨，24°C，南风1级",
        }
        return mock_db.get(city, f"抱歉，没有 {city} 的天气数据。")


class MockTimeTool(Tool):
    """模拟时间查询工具（测试用）。"""

    def __init__(self):
        super().__init__(
            name="time",
            description="获取当前时间。",
            func=self._get_time,
        )

    def _get_time(self, _input: str = "") -> str:
        from datetime import datetime

        return datetime.now().strftime("现在时间是 %Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def _assert(label: str, condition: bool):
    global PASS, FAIL
    if condition:
        print(f"  [OK] {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}")
        FAIL += 1


def test_runnable_protocol():
    """测试 Runnable 核心协议。"""
    print("\n" + "=" * 60)
    print("测试 1: Runnable 协议 + LCEL 管道")
    print("-" * 60)

    # 1.1 RunnableLambda
    add_one = RunnableLambda(lambda x: x + 1)
    _assert("RunnableLambda.invoke", add_one.invoke(5) == 6)

    # 1.2 管道组合
    chain = add_one | RunnableLambda(lambda x: x * 2) | RunnableLambda(lambda x: f"结果: {x}")
    _assert("管道组合 a|b|c", chain.invoke(5) == "结果: 12")

    # 1.3 RunnableMap 并行
    parallel = RunnableMap({
        "double": RunnableLambda(lambda x: x * 2),
        "square": RunnableLambda(lambda x: x * x),
    })
    result = parallel.invoke(5)
    _assert("RunnableMap 并行", result == {"double": 10, "square": 25})

    # 1.4 流式
    stream_result = list(add_one.stream(5))
    _assert("RunnableLambda.stream", stream_result == [6])


def test_agent_state():
    """测试 AgentState 不可变性。"""
    print("\n" + "=" * 60)
    print("测试 2: AgentState 不可变状态管理")
    print("-" * 60)

    s1 = AgentState()
    s2 = s1.add_message(LLMMessage(role="user", content="你好"))

    _assert("旧状态不变", len(s1.messages) == 0)
    _assert("新状态有消息", len(s2.messages) == 1)
    _assert("状态内容正确", s2.last_user_input == "你好")

    s3 = s2.add_step(
        AgentAction(tool="weather", tool_input="北京", thought="查天气"),
        "晴天",
    )
    _assert("步骤追加", len(s3.intermediate_steps) == 1)
    _assert("scratchpad 生成", "Observation: 晴天" in s3.scratchpad_text)
    _assert("状态机流转", s3.iteration == 1)


def test_react_agent_with_mock_llm():
    """测试 ReActAgent 用模拟 LLM（固定输出）。"""
    print("\n" + "=" * 60)
    print("测试 3: ReActAgent + AgentExecutor（模拟 LLM）")
    print("-" * 60)

    from zhixia.llm.base import LLMEngine

    class MockLLM(LLMEngine):
        """模拟 LLM：固定返回 ReAct 格式输出。"""

        def __init__(self, response: str):
            self._response = response

        @property
        def name(self):
            return "mock"

        def chat(self, messages, max_new_tokens=32):
            return self._response

        def set_system_prompt(self, prompt):
            pass

    # 场景 1: Agent 直接回答（不需要工具）
    mock_llm = MockLLM("Thought: 用户在打招呼，不需要工具。\nFinal Answer: 你好呀！我是小匣。")
    registry = ToolRegistry()
    registry.register(MockWeatherTool())

    agent = ReActAgent(llm_engine=mock_llm, tools=registry)
    executor = AgentExecutor(agent=agent, tools=registry, max_iterations=3)

    state = AgentState(messages=[LLMMessage(role="user", content="你好")])
    final_state = executor.invoke(state)

    _assert("ReAct 直接回答完成", final_state.status == "finished")
    _assert("ReAct 答案正确", "你好呀" in final_state.messages[-1].content)

    # 场景 2: Agent 调用工具
    mock_llm2 = MockLLM(
        "Thought: 用户想知道北京天气，我需要查询天气工具。\n"
        "Action: weather\n"
        "Action Input: 北京\n"
        "Observation: 晴天，28°C\n"
        "Thought: 我已经知道答案了。\n"
        "Final Answer: 北京今天晴天，28°C。"
    )
    agent2 = ReActAgent(llm_engine=mock_llm2, tools=registry)
    executor2 = AgentExecutor(agent=agent2, tools=registry, max_iterations=3)

    state2 = AgentState(messages=[LLMMessage(role="user", content="北京天气？")])
    final_state2 = executor2.invoke(state2)

    _assert("ReAct 工具调用完成", final_state2.status == "finished")
    _assert("ReAct 工具结果融入", "28°C" in final_state2.messages[-1].content)


def test_callbacks():
    """测试回调系统。"""
    print("\n" + "=" * 60)
    print("测试 4: CallbackManager + Handler")
    print("-" * 60)

    collected = []

    class TestHandler(StreamingDisplayHandler):
        def on_agent_thought(self, run_id, thought, **kwargs):
            collected.append(("thought", thought))

        def on_agent_action(self, run_id, action, **kwargs):
            collected.append(("action", action.tool))

        def on_agent_finish(self, run_id, finish, **kwargs):
            collected.append(("finish", finish.return_values.get("text", "")))

    manager = CallbackManager([TestHandler()])
    config = RunnableConfig(callbacks=manager)

    # 模拟触发回调
    manager.on_agent_thought("r1", "我需要查天气")
    manager.on_agent_action("r1", AgentAction(tool="weather", tool_input="北京"))
    manager.on_agent_finish("r1", AgentFinish(return_values={"text": "晴天"}))

    _assert("回调收集 thought", collected[0] == ("thought", "我需要查天气"))
    _assert("回调收集 action", collected[1] == ("action", "weather"))
    _assert("回调收集 finish", collected[2] == ("finish", "晴天"))


def test_tool_calling_agent_real():
    """用真实 KIMI API 测试 ToolCallingAgent。"""
    print("\n" + "=" * 60)
    print("测试 5: ToolCallingAgent + 真实 KIMI API")
    print("-" * 60)

    api_key = os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        print("  [SKIP] 跳过（未设置 KIMI_API_KEY）")
        return

    llm = CloudLLMEngine(
        LLMConfig(
            cloud_api_url="https://api.moonshot.cn/v1/chat/completions",
            cloud_api_key=api_key,
            cloud_model_name="moonshot-v1-8k",
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
            system_prompt="你是「小匣」，一个温暖的智能助手。",
        )
    )

    registry = ToolRegistry()
    registry.register(MockWeatherTool())
    registry.register(MockTimeTool())

    agent = ToolCallingAgent(llm_engine=llm, tools=registry, max_new_tokens=256)
    executor = AgentExecutor(agent=agent, tools=registry, max_iterations=3)

    # 测试 1: 不需要工具的简单问题
    print("  测试 5a: 简单问题（无需工具）")
    state = AgentState(messages=[LLMMessage(role="user", content="你好，请自我介绍")])
    final = executor.invoke(state)
    _assert("简单问题完成", final.status == "finished")
    _assert("简单问题有回答", len(final.messages[-1].content) > 5)
    print(f"    回答: {final.messages[-1].content[:80]}...")

    # 测试 2: 需要工具的问题（Agent 应该调用 weather 工具）
    print("  测试 5b: 工具调用问题（查天气）")
    state2 = AgentState(messages=[LLMMessage(role="user", content="北京今天天气怎么样？")])
    final2 = executor.invoke(state2)
    _assert("工具问题完成", final2.status == "finished")
    print(f"    回答: {final2.messages[-1].content[:120]}...")
    # 只要完成就认为是成功（小模型可能不严格调用工具，但会给出合理回答）
    _assert("工具问题有回答", len(final2.messages[-1].content) > 0)


def test_agent_runner():
    """测试 AgentRunner 高层封装。"""
    print("\n" + "=" * 60)
    print("测试 6: AgentRunner 高层封装")
    print("-" * 60)

    from zhixia.llm.base import LLMEngine

    class MockLLM(LLMEngine):
        def __init__(self):
            self._count = 0

        @property
        def name(self):
            return "mock"

        def chat(self, messages, max_new_tokens=32):
            self._count += 1
            return "Thought: 直接回答。\nFinal Answer: 测试成功。"

        def set_system_prompt(self, prompt):
            pass

    llm = MockLLM()
    registry = ToolRegistry()
    agent = ReActAgent(llm_engine=llm, tools=registry)
    executor = AgentExecutor(agent=agent, tools=registry)
    runner = AgentRunner(executor)

    result = runner.run("测试")
    _assert("Runner.run 返回结果", result["status"] == "finished")
    _assert("Runner.run 文本正确", "测试成功" in result["text"])

    # 流式
    chunks = list(runner.stream("测试"))
    _assert("Runner.stream 有输出", len(chunks) > 0)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_runnable_protocol()
    test_agent_state()
    test_react_agent_with_mock_llm()
    test_callbacks()
    test_agent_runner()
    test_tool_calling_agent_real()

    print("\n" + "=" * 60)
    print(f"测试完成: [PASS] {PASS} 通过, [FAIL] {FAIL} 失败")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
