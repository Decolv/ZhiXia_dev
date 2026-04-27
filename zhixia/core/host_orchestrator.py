"""HostOrchestrator — 主机编排器

替代原有的固定 VoicePipeline，实现：
1. 动态 Agent 组装：根据当前挂载的 Skill 卡 + Knowledge 卡实时构建 Agent
2. 卡片变化检测：每轮对话前自动扫描槽位
3. 无卡模式支持：未插卡时使用基础 LLM 对话
4. 与原有 Pipeline 兼容：复用 ASR → (Agent/LLM) → TTS → Play 流水线

使用方式：
    orchestrator = HostOrchestrator(config, asr, llm, tts, player)
    orchestrator.initialize_slots()  # 初始化槽位
    
    # 处理一轮对话
    orchestrator.process_turn(audio_path)
    
    # 关机时清理
    orchestrator.shutdown()
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zhixia.agent import (
    AgentExecutor,
    AgentState,
    CallbackManager,
    LoggingHandler,
    ReActAgent,
    ToolCallingAgent,
    ToolRegistry,
)
from zhixia.agent.runnable import RunnableConfig
from zhixia.asr.base import ASREngine, ASRResult
from zhixia.audio.base import AudioPlayer
from zhixia.config.settings import AppSettings
from zhixia.core.card_base import HostContext, KnowledgeHub, PersonaHolder
from zhixia.core.card_loader import CardLoader
from zhixia.display.base import DisplayOutput, DisplayPayload
from zhixia.llm.base import LLMEngine, LLMMessage
from zhixia.llm.output_parser import _strip_thinking_tokens, parse_llm_output
from zhixia.llm.rag.base import RAGContext, RAGRetriever
from zhixia.llm.rag.null_retriever import NullRAGRetriever
from zhixia.memory.conversation_memory import ConversationMemory
from zhixia.tts.base import TTSEngine

logger = logging.getLogger(__name__)

# 复用 Pipeline 的分句正则和常量
_SENTENCE_END = re.compile(r'[。！？!?…]+|(?<=[^0-9])[.]+(?=[^0-9])|[；;]+')
_MIN_CHUNK_LEN = 8
_INTER_CHUNK_GAP_SEC = 0.02
_INITIAL_PLAY_BUFFER_CHUNKS = 2
_SENTINEL = object()


def _split_sentences(text: str) -> List[str]:
    """按句末标点切分，保留标点。"""
    parts = []
    last = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        chunk = text[last:end].strip()
        if chunk:
            parts.append(chunk)
        last = end
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts


class HostOrchestrator:
    """主机编排器 — 插卡式 Agent 的核心控制器。

    Attributes:
        config: 应用配置
        card_loader: 卡片加载器
        asr_engine: 语音识别引擎
        llm_engine: 大语言模型引擎
        tts_engine: 语音合成引擎
        audio_player: 音频播放器
        display: 显示输出
        conversation_memory: 对话记忆
    """

    def __init__(
        self,
        config: AppSettings,
        asr_engine: ASREngine,
        llm_engine: LLMEngine,
        tts_engine: TTSEngine,
        audio_player: AudioPlayer,
        display: Optional[DisplayOutput] = None,
        slot_paths: Optional[Dict[str, Tuple[Path, str]]] = None,
    ) -> None:
        self.config = config
        self.asr_engine = asr_engine
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.audio_player = audio_player

        if display is None:
            from zhixia.display.null_display import NullDisplay
            self.display = NullDisplay()
        else:
            self.display = display

        self.conversation_memory = ConversationMemory()

        # 初始化主机上下文
        tool_registry = ToolRegistry()
        persona_holder = PersonaHolder(config.llm.system_prompt)
        knowledge_hub = KnowledgeHub()
        self.host_context = HostContext(
            tool_registry=tool_registry,
            persona_holder=persona_holder,
            knowledge_hub=knowledge_hub,
            display=self.display,
            config=config,
        )

        # 初始化卡片加载器
        if slot_paths is None:
            project_root = config.project_root
            slot_paths = {
                "skill": (project_root / "cards" / "slot_a", "skill"),
                "knowledge": (project_root / "cards" / "slot_b", "knowledge"),
            }
        self.card_loader = CardLoader(slot_paths, self.host_context)

        # 缓存当前 Agent（避免每轮重复构建）
        self._current_agent: Optional[AgentExecutor] = None
        self._current_agent_signature: str = ""
        self._agent_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def initialize_slots(self) -> Dict[str, Optional[str]]:
        """初始化：扫描槽位并同步卡片。"""
        logger.info("初始化插卡槽位...")
        changes = self.card_loader.scan_and_sync()
        for slot_id, change in changes.items():
            if change:
                logger.info("[%s] %s", slot_id, change)
        self._invalidate_agent_cache()
        return changes

    def check_slots(self) -> Dict[str, Optional[str]]:
        """检查槽位变化，如有变化重新同步。"""
        changes = self.card_loader.scan_and_sync()
        if any(v is not None for v in changes.values()):
            logger.info("检测到卡片变化: %s", changes)
            self._invalidate_agent_cache()
        return changes

    def process_turn(self, audio_path: Path) -> None:
        """处理一轮完整对话（音频输入 → 音频输出）。

        流程：
        1. 检查槽位变化
        2. ASR 语音识别
        3. 构建/复用 Agent
        4. Agent/LLM 执行
        5. TTS + 播放
        6. 保存对话记忆
        """
        t0 = time.perf_counter()
        print("\n" + "=" * 70)
        print("️  ZhiXia 插卡式 Agent — 对话回合")
        print("=" * 70)

        # 1. 检查卡片变化
        changes = self.check_slots()
        mounted = self.card_loader.get_mounted_names()
        if mounted:
            print(f" 已插卡: {', '.join(mounted)}")
        else:
            print(" 无卡模式 — 基础对话")

        # 2. ASR
        print("\n[阶段 1/4] 语音识别 (ASR)")
        print("-" * 70)
        t_asr = time.perf_counter()
        asr_result = self.asr_engine.transcribe(audio_path)
        if not asr_result.text:
            raise RuntimeError("语音识别失败")
        asr_duration = time.perf_counter() - t_asr
        print(f' 识别完成: "{asr_result.text}"')
        print(f"⏱️  ASR 耗时: {asr_duration:.3f}s")

        # 3. 构建 Agent（如果缓存失效）
        print("\n[阶段 2/4] Agent 组装")
        print("-" * 70)
        agent = self._get_or_build_agent()
        if agent:
            print(f" Agent 模式: {agent.agent.name}")
            print(f"️  可用工具: {[t.name for t in self.host_context.tool_registry.list_tools()]}")
        else:
            print(" 直接 LLM 对话模式")

        # 4. 执行（Agent 或 直接 LLM）
        print("\n[阶段 3/4] AI 生成")
        print("-" * 70)
        t_gen = time.perf_counter()

        if agent:
            response_text = self._run_agent(agent, asr_result.text)
        else:
            response_text = self._run_direct_llm(asr_result.text)

        gen_duration = time.perf_counter() - t_gen
        print(f" 生成完成: \"{response_text[:100]}...\"")
        print(f"⏱️  生成耗时: {gen_duration:.3f}s")

        # 5. TTS + Play
        print("\n[阶段 4/4] 语音合成与播放")
        print("-" * 70)
        self._tts_and_play(response_text)

        # 6. 保存记忆
        if self.config.llm.memory_enabled:
            self.conversation_memory.add_message("user", asr_result.text)
            self.conversation_memory.add_message("assistant", response_text)

        # 7. 显示
        parsed = parse_llm_output(response_text)
        self.display.show(DisplayPayload(
            text=parsed.text,
            emotion=parsed.emotion,
            is_thinking=False,
            metadata=parsed.metadata,
        ))

        total = time.perf_counter() - t0
        print("\n" + "=" * 70)
        print(f"⏱️  总耗时: {total:.3f}s")
        print("=" * 70)

    def shutdown(self) -> None:
        """关机：卸载所有卡片，清除痕迹。"""
        logger.info("主机编排器关闭中...")
        self.card_loader.force_unmount_all()
        self.llm_engine.shutdown()
        logger.info("主机编排器已关闭")

    # ------------------------------------------------------------------
    # Agent 构建
    # ------------------------------------------------------------------

    def _get_or_build_agent(self) -> Optional[AgentExecutor]:
        """获取或构建 Agent。

        如果卡片未变化，复用缓存的 Agent。
        线程安全：使用锁保护缓存读写。
        """
        with self._agent_lock:
            signature = self._compute_agent_signature()
            if self._current_agent and self._current_agent_signature == signature:
                return self._current_agent

            tools = self.host_context.tool_registry
            has_tools = len(tools.list_tools()) > 0

            if not has_tools:
                # 无工具 = 无 Skill 卡，使用直接 LLM 模式
                self._current_agent = None
                self._current_agent_signature = signature
                return None

            # 根据配置选择 Agent 类型
            agent_type = getattr(self.config, "agent", None)
            engine_type = getattr(agent_type, "engine", "react") if agent_type else "react"
            max_iter = getattr(agent_type, "max_iterations", 5) if agent_type else 5
            stop_method = getattr(agent_type, "early_stopping_method", "raise") if agent_type else "raise"

            if engine_type == "tool_calling":
                agent = ToolCallingAgent(
                    llm_engine=self.llm_engine,
                    tools=tools,
                    max_new_tokens=self.config.llm.max_new_tokens,
                )
            else:
                agent = ReActAgent(
                    llm_engine=self.llm_engine,
                    tools=tools,
                    max_new_tokens=self.config.llm.max_new_tokens,
                )

            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=max_iter,
                early_stopping_method=stop_method,
            )

            self._current_agent = executor
            self._current_agent_signature = signature
            return executor

    def _compute_agent_signature(self) -> str:
        """计算当前 Agent 配置的签名（用于缓存）。

        包含卡片名称和工具的类名，防止同名但不同实现的工具导致缓存命中错误。
        """
        names = sorted(self.card_loader.get_mounted_names())
        tool_signatures = sorted([
            f"{t.name}:{t.__class__.__name__}"
            for t in self.host_context.tool_registry.list_tools()
        ])
        return f"cards:{names}:tools:{tool_signatures}"

    def _invalidate_agent_cache(self) -> None:
        """使 Agent 缓存失效。"""
        with self._agent_lock:
            self._current_agent = None
            self._current_agent_signature = ""

    # ------------------------------------------------------------------
    # 执行模式
    # ------------------------------------------------------------------

    def _run_agent(self, agent: AgentExecutor, user_text: str) -> str:
        """Agent 模式执行。"""
        # 构建初始状态
        system_prompt = self.host_context.persona_holder.current_persona
        messages = [LLMMessage(role="system", content=system_prompt)]

        # 添加知识检索（如果 Knowledge 卡已挂载）
        knowledge_chunks = self.host_context.knowledge_hub.retrieve(user_text, top_k=3)
        if knowledge_chunks:
            context_block = "\n".join(knowledge_chunks)
            messages.append(LLMMessage(
                role="system",
                content=f"参考信息:\n{context_block}\n请基于以上参考信息回答。",
            ))

        # 添加对话记忆
        if self.config.llm.memory_enabled:
            history = self.conversation_memory.get_history(
                max_rounds=self.config.llm.max_memory_rounds,
                max_tokens=self.config.llm.max_memory_tokens,
            )
            for hist_msg in history:
                messages.append(LLMMessage(role=hist_msg["role"], content=hist_msg["content"]))

        messages.append(LLMMessage(role="user", content=user_text))

        # 执行 Agent
        state = AgentState(messages=messages)
        callbacks = CallbackManager([LoggingHandler()])
        config = RunnableConfig(callbacks=callbacks, recursion_limit=5)

        final_state = agent.invoke(state, config)

        # 提取最终答案
        for msg in reversed(final_state.messages):
            if msg.role == "assistant":
                return msg.content
        return "抱歉，我没有得到答案。"

    def _run_direct_llm(self, user_text: str) -> str:
        """直接 LLM 模式（无卡 / 无工具时）。"""
        system_prompt = self.host_context.persona_holder.current_persona
        messages = [LLMMessage(role="system", content=system_prompt)]

        # 知识检索（知识卡可独立使用）
        knowledge_chunks = self.host_context.knowledge_hub.retrieve(user_text, top_k=3)
        if knowledge_chunks:
            context_block = "\n".join(knowledge_chunks)
            messages.append(LLMMessage(
                role="system",
                content=f"参考信息:\n{context_block}\n请基于以上参考信息回答。",
            ))

        if self.config.llm.memory_enabled:
            history = self.conversation_memory.get_history(
                max_rounds=self.config.llm.max_memory_rounds,
                max_tokens=self.config.llm.max_memory_tokens,
            )
            for hist_msg in history:
                messages.append(LLMMessage(role=hist_msg["role"], content=hist_msg["content"]))

        messages.append(LLMMessage(role="user", content=user_text))

        response = self.llm_engine.chat(messages, max_new_tokens=self.config.llm.max_new_tokens)
        return response.strip()

    # ------------------------------------------------------------------
    # TTS + Play
    # ------------------------------------------------------------------

    def _tts_and_play(self, text: str) -> None:
        """TTS 合成并播放。简化版（非流式，适合短文本）。"""
        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]

        print(f"️  合成 {len(sentences)} 句...")
        for i, sentence in enumerate(sentences):
            clean = _strip_thinking_tokens(sentence)
            if not clean or len(clean) < _MIN_CHUNK_LEN:
                continue
            try:
                wav = self.tts_engine.synthesize_to_bytes(clean)
                if wav:
                    self.audio_player.play_bytes(wav, blocking=True)
                    if i < len(sentences) - 1:
                        time.sleep(_INTER_CHUNK_GAP_SEC)
            except Exception as exc:
                logger.exception("TTS/播放失败: %s", clean[:30])
