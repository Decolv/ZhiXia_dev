"""Runnable 核心协议 —— 参照 LangChain LCEL 架构

设计目标：
1. 所有组件（LLM、Prompt、Agent、Tool）实现统一接口 invoke / stream / transform。
2. 支持管道组合 `runnable1 | runnable2 | runnable3`。
3. 流式透传：上游 yield chunk，下游可实时处理。
4. 零外部依赖，纯 Python 实现。

核心类：
    Runnable          — 抽象基类，定义 invoke/stream/transform/batch
    RunnableSequence  — 管道组合 `a | b | c`
    RunnableConfig    — 运行时配置（callbacks、metadata、tags）
    RunnableMap       — 并行分支 { "key1": r1, "key2": r2 }

使用示例：
    chain = prompt_template | llm | output_parser
    result = chain.invoke({"input": "你好"})
    for chunk in chain.stream({"input": "你好"}):
        print(chunk, end="")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")


# ---------------------------------------------------------------------------
# RunnableConfig — 运行时配置
# ---------------------------------------------------------------------------

class RunnableConfig:
    """Runnable 运行时配置容器。

    对应 LangChain 的 RunnableConfig，但做了大幅精简：
    - callbacks: CallbackManager 实例，用于观测生命周期事件
    - metadata: 任意字典，用于 tracing / 日志
    - tags: 字符串列表，用于过滤日志
    - recursion_limit: 最大递归/迭代深度（AgentExecutor 使用）
    """

    def __init__(
        self,
        callbacks=None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        recursion_limit: int = 10,
        **kwargs: Any,
    ) -> None:
        from zhixia.agent.callbacks import CallbackManager

        self.callbacks = callbacks or CallbackManager()
        self.metadata = metadata or {}
        self.tags = tags or []
        self.recursion_limit = recursion_limit
        self.extras = kwargs

    def copy(self, **overrides: Any) -> RunnableConfig:
        """创建配置的浅拷贝，可覆盖特定字段。"""
        new = RunnableConfig(
            callbacks=overrides.get("callbacks", self.callbacks),
            metadata={**self.metadata, **overrides.get("metadata", {})},
            tags=self.tags + overrides.get("tags", []),
            recursion_limit=overrides.get("recursion_limit", self.recursion_limit),
        )
        new.extras = {**self.extras, **overrides.get("extras", {})}
        return new


# ---------------------------------------------------------------------------
# Runnable — 抽象基类
# ---------------------------------------------------------------------------

class Runnable(Generic[Input, Output], ABC):
    """所有可运行组件的抽象基类。

    子类必须实现：
        - _invoke(self, input, config) -> Output

    可选覆盖：
        - _stream(self, input, config) -> Iterator[Output]
        - _transform(self, iterator, config) -> Iterator[Output]

    公共方法（包装了 callbacks 生命周期）：
        - invoke(input, config=None) -> Output
        - stream(input, config=None) -> Iterator[Output]
        - batch(inputs, config=None) -> List[Output]
    """

    @abstractmethod
    def _invoke(self, input: Input, config: RunnableConfig) -> Output:
        """核心实现。子类覆盖此方法。"""
        ...

    def _stream(self, input: Input, config: RunnableConfig) -> Iterator[Output]:
        """流式实现。默认回退到 invoke 后 yield 完整结果。"""
        yield self._invoke(input, config)

    def _transform(
        self, iterator: Iterator[Input], config: RunnableConfig
    ) -> Iterator[Output]:
        """转换输入迭代器。默认逐条 invoke。"""
        for item in iterator:
            yield self._invoke(item, config)

    # -- 公共包装方法（加入 callbacks 生命周期） --

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        config = config or RunnableConfig()
        run_id = config.callbacks.on_chain_start(self, input, config)
        try:
            output = self._invoke(input, config)
            config.callbacks.on_chain_end(run_id, output, config)
            return output
        except Exception as exc:
            config.callbacks.on_chain_error(run_id, exc, config)
            raise

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        config = config or RunnableConfig()
        run_id = config.callbacks.on_chain_start(self, input, config)
        try:
            for chunk in self._stream(input, config):
                config.callbacks.on_chain_stream(run_id, chunk, config)
                yield chunk
            config.callbacks.on_chain_end(run_id, None, config)
        except Exception as exc:
            config.callbacks.on_chain_error(run_id, exc, config)
            raise

    def batch(
        self, inputs: List[Input], config: Optional[RunnableConfig] = None
    ) -> List[Output]:
        return [self.invoke(inp, config) for inp in inputs]

    # -- LCEL 管道操作符 --

    def __or__(
        self, other: Union[Runnable[Output, Any], Callable[[Output], Any]]
    ) -> RunnableSequence[Input, Any]:
        """管道组合：r1 | r2 | r3

        支持 Runnable 或普通函数：
            chain = prompt | llm | parser
            chain = prompt | llm | lambda x: x.upper()
        """
        if isinstance(other, Runnable):
            return RunnableSequence(self, other)
        # 普通函数包装为 RunnableLambda
        return RunnableSequence(self, RunnableLambda(other))

    def __ror__(
        self, other: Union[Runnable[Any, Input], Callable[[Any], Input]]
    ) -> RunnableSequence[Any, Output]:
        """反向管道：func | runnable"""
        if isinstance(other, Runnable):
            return RunnableSequence(other, self)
        return RunnableSequence(RunnableLambda(other), self)

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# RunnableLambda — 将普通函数包装为 Runnable
# ---------------------------------------------------------------------------

class RunnableLambda(Runnable[Any, Any]):
    """将普通函数/可调用对象包装为 Runnable。

    示例：
        upper = RunnableLambda(lambda x: x.upper())
        chain = prompt | llm | upper
    """

    def __init__(self, func: Callable[[Any], Any], name: Optional[str] = None) -> None:
        self.func = func
        self._name = name or getattr(func, "__name__", "lambda")

    @property
    def name(self) -> str:
        return self._name

    def _invoke(self, input: Any, config: RunnableConfig) -> Any:
        return self.func(input)


# ---------------------------------------------------------------------------
# RunnableSequence — 管道组合 a | b | c
# ---------------------------------------------------------------------------

class RunnableSequence(Runnable[Any, Any]):
    """Runnable 管道序列：按顺序执行多个 Runnable。

    内部实现：
        first._invoke(input) -> mid -> last._invoke(mid) -> output

    流式支持：
        - 如果所有组件都支持 stream，则逐 chunk 透传
        - 否则回退到 invoke（收集完整输出后传给下一个）
    """

    def __init__(self, *steps: Runnable) -> None:
        self.steps = list(steps)
        if len(self.steps) < 2:
            raise ValueError("RunnableSequence 至少需要 2 个步骤")

    @property
    def first(self) -> Runnable:
        return self.steps[0]

    @property
    def last(self) -> Runnable:
        return self.steps[-1]

    @property
    def middle(self) -> List[Runnable]:
        return self.steps[1:-1]

    @property
    def name(self) -> str:
        return " | ".join(s.name for s in self.steps)

    def _invoke(self, input: Any, config: RunnableConfig) -> Any:
        output = self.first.invoke(input, config)
        for step in self.middle:
            output = step.invoke(output, config)
        return self.last.invoke(output, config)

    def _stream(self, input: Any, config: RunnableConfig) -> Iterator[Any]:
        """流式管道：尽可能逐 chunk 透传。

        策略：
        - 如果只有首尾两步支持 stream，中间步骤需要完整输入 → 先 invoke 中间，再 stream 最后
        - 如果只有一步 → 直接 stream 那一步
        """
        if len(self.steps) == 2:
            # 只有两步：尝试让 first stream，last 处理每个 chunk
            # 但如果 last 不支持逐 chunk 处理（如 parser），则回退
            try:
                for chunk in self.first.stream(input, config):
                    yield from self.last.stream(chunk, config)
                return
            except (TypeError, AttributeError):
                pass
            # 回退：先 invoke first，再 stream last
            mid = self.first.invoke(input, config)
            yield from self.last.stream(mid, config)
            return

        # 多步管道：先 invoke 到倒数第二步，最后一步 stream
        output = self.first.invoke(input, config)
        for step in self.middle:
            output = step.invoke(output, config)
        yield from self.last.stream(output, config)


# ---------------------------------------------------------------------------
# RunnableMap — 并行分支
# ---------------------------------------------------------------------------

class RunnableMap(Runnable[Dict[str, Any], Dict[str, Any]]):
    """并行执行多个 Runnable，结果合并为字典。

    对应 LangChain 的 {"key": runnable} 语法。

    示例：
        map_runnable = RunnableMap({
            "summary": summary_chain,
            "sentiment": sentiment_chain,
        })
        result = map_runnable.invoke("今天天气很好")
        # {"summary": "天气晴朗", "sentiment": "positive"}
    """

    def __init__(self, steps: Dict[str, Runnable]) -> None:
        self.steps = steps

    @property
    def name(self) -> str:
        return "RunnableMap({})".format(", ".join(self.steps.keys()))

    def _invoke(self, input: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        # 如果 input 不是 dict，则作为每个 step 的输入
        result = {}
        for key, runnable in self.steps.items():
            step_input = input.get(key) if isinstance(input, dict) else input
            result[key] = runnable.invoke(step_input, config)
        return result

    def _stream(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> Iterator[Dict[str, Any]]:
        # 流式并行：每次 yield 一个包含最新 chunk 的字典
        iterators = {}
        for key, runnable in self.steps.items():
            step_input = input.get(key) if isinstance(input, dict) else input
            iterators[key] = runnable.stream(step_input, config)

        active = dict(iterators)
        while active:
            chunk_map = {}
            finished = []
            for key, it in active.items():
                try:
                    chunk_map[key] = next(it)
                except StopIteration:
                    finished.append(key)
                except Exception:
                    finished.append(key)
                    raise
            for key in finished:
                del active[key]
            if chunk_map:
                yield chunk_map
