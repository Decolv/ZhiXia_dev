"""HostOrchestrator — 主机编排器（纯净版）

主机与卡片深度解耦：
1. 主机不包含任何卡片特定逻辑（无 __NAV_DATA__ 解析，无特定工具名称硬编码）
2. Agent 类型通过 HostContext.agent_configurator 动态配置
3. 响应后处理通过 ResponsePostProcessor 扩展点
4. 无卡时主机仅包含基础 LLM 对话，不泄露卡片内容

使用方式：
    orchestrator = HostOrchestrator(config, asr, llm, tts, player)
    orchestrator.initialize_slots()
    orchestrator.process_turn(audio_path)
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
    BaseCallbackHandler,
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
from zhixia.memory.conversation_memory import ConversationMemory
from zhixia.tts.base import TTSEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DisplayCallbackHandler — 支持思考播报的回调处理器
# ---------------------------------------------------------------------------

class DisplayCallbackHandler(BaseCallbackHandler):
    """将Agent思考过程实时显示到Display的回调处理器。

    同时支持 Live2D 眼睛动画联动。
    主机不关心具体工具类型，仅做通用展示。
    """

    def __init__(self, display: DisplayOutput) -> None:
        self.display = display
        self._current_run_id: Optional[str] = None

    def on_thinking_start(self, run_id: str, **kwargs: Any) -> None:
        self._current_run_id = run_id
        self.display.update_thinking(True, "正在思考...")
        self.display.show(DisplayPayload(
            text="", emotion="thinking", is_thinking=True,
            thinking_text="正在思考...", eye_state="thinking",
        ))
        print("[思考] 开始分析问题...")

    def on_thinking_end(self, run_id: str, **kwargs: Any) -> None:
        self.display.update_thinking(False)
        self.display.set_eye_state("neutral")
        print("[思考] 分析完成")

    def on_agent_thought(self, run_id: str, thought: str, **kwargs: Any) -> None:
        self.display.show(DisplayPayload(
            text="", emotion="thinking", is_thinking=True,
            thinking_text=thought, eye_state="thinking",
        ))
        print(f"[思考] {thought}")

    def on_agent_action(self, run_id: str, action: Any, **kwargs: Any) -> None:
        from zhixia.agent.base import AgentAction
        if isinstance(action, AgentAction):
            action_text = f"正在调用 {action.tool} 工具..."
            self.display.show(DisplayPayload(
                text="", emotion="working", is_thinking=True,
                thinking_text=action_text, eye_state="working",
            ))
            print(f"[工具] 调用 {action.tool}")

    def on_agent_finish(self, run_id: str, finish: Any, **kwargs: Any) -> None:
        from zhixia.agent.base import AgentFinish
        if isinstance(finish, AgentFinish):
            text = finish.return_values.get("text", "")
            self.display.show(DisplayPayload(
                text=text, emotion="neutral", is_thinking=False,
                thinking_text="", eye_state="neutral",
                blink_override=True,
            ))


# ---------------------------------------------------------------------------
# 流式输出相关常量
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r'[。！？!?…]+|(?<=[^0-9])[.]+(?=[^0-9])|[；;]+')
_MIN_CHUNK_LEN = 8
_INTER_CHUNK_GAP_SEC = 0.02
_INITIAL_PLAY_BUFFER_CHUNKS = 2
_SENTINEL = object()


def _split_sentences(text: str) -> List[str]:
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
    """主机编排器 — 插卡式 Agent 的核心控制器（纯净版）。

    主机职责：
    1. 管理卡片生命周期（挂载/卸载）
    2. 根据卡片注册的工具构建 Agent
    3. 执行对话流水线（ASR → Agent/LLM → TTS → Play）
    4. 调用 ResponsePostProcessor 处理响应（扩展点）

    主机不关心：
    - 具体有哪些工具
    - 工具的响应格式
    - 特定 UI 展示逻辑
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
        enable_live2d_eyes: bool = True,
    ) -> None:
        self.config = config
        self.asr_engine = asr_engine
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.audio_player = audio_player

        # 初始化显示输出
        if display is None:
            if enable_live2d_eyes:
                try:
                    from zhixia.display.live2d_display import Live2dEyeDisplay
                    self.display = Live2dEyeDisplay(auto_start=True)
                    logger.info("Live2D 眼睛显示已启用")
                except Exception as exc:
                    logger.warning("Live2D 眼睛显示启动失败: %s，使用空显示", exc)
                    from zhixia.display.null_display import NullDisplay
                    self.display = NullDisplay()
            else:
                from zhixia.display.null_display import NullDisplay
                self.display = NullDisplay()
        else:
            self.display = display

        self.conversation_memory = ConversationMemory()

        # 初始化主机上下文（纯净，不包含卡片特定内容）
        tool_registry = ToolRegistry()
        persona_holder = PersonaHolder(config.llm.system_prompt)
        knowledge_hub = KnowledgeHub()
        self.host_context = HostContext(
            tool_registry=tool_registry,
            persona_holder=persona_holder,
            knowledge_hub=knowledge_hub,
            display=self.display,
            config=config,
            llm_engine=self.llm_engine,
        )

        # 初始化卡片加载器
        if slot_paths is None:
            project_root = config.project_root
            slot_paths = {
                "skill": (project_root / "cards" / "slot_a", "skill"),
                "knowledge": (project_root / "cards" / "slot_b", "knowledge"),
            }
        self.card_loader = CardLoader(slot_paths, self.host_context)

        # Agent 缓存
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
        """检查槽位变化。"""
        changes = self.card_loader.scan_and_sync()
        if any(v is not None for v in changes.values()):
            logger.info("检测到卡片变化: %s", changes)
            self._invalidate_agent_cache()
        return changes

    def process_turn(self, audio_path: Path) -> None:
        """处理一轮完整对话（音频输入 → 音频输出）。"""
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

        # 5. 响应后处理（卡片注册的处理器）
        cleaned_text = self._run_response_processors(response_text)

        # 6. TTS + Play
        print("\n[阶段 4/4] 语音合成与播放")
        print("-" * 70)
        self._tts_and_play(cleaned_text)

        # 7. 保存记忆
        if self.config.llm.memory_enabled:
            self.conversation_memory.add_message("user", asr_result.text)
            self.conversation_memory.add_message("assistant", cleaned_text)

        # 8. 显示
        parsed = parse_llm_output(cleaned_text)
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
        """关机：卸载所有卡片，清除痕迹，停止显示。"""
        logger.info("主机编排器关闭中...")
        self.card_loader.force_unmount_all()
        if hasattr(self.display, "stop"):
            self.display.stop()
        self.llm_engine.shutdown()
        logger.info("主机编排器已关闭")

    # ------------------------------------------------------------------
    # Agent 构建
    # ------------------------------------------------------------------

    def _get_or_build_agent(self) -> Optional[AgentExecutor]:
        """获取或构建 Agent。"""
        with self._agent_lock:
            signature = self._compute_agent_signature()
            if self._current_agent and self._current_agent_signature == signature:
                return self._current_agent

            tools = self.host_context.tool_registry
            has_tools = len(tools.list_tools()) > 0

            if not has_tools:
                self._current_agent = None
                self._current_agent_signature = signature
                return None

            # 使用卡片配置的 Agent 类型
            agent_config = self.host_context.agent_configurator.get_config()
            agent_type = agent_config["agent_type"]
            max_iter = agent_config["max_iterations"]
            stop_method = agent_config["early_stopping_method"]

            if agent_type == "tool_calling":
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
        """计算 Agent 配置签名（用于缓存）。"""
        names = sorted(self.card_loader.get_mounted_names())
        tool_signatures = sorted([
            f"{t.name}:{t.__class__.__name__}"
            for t in self.host_context.tool_registry.list_tools()
        ])
        return f"cards={names};tools={tool_signatures}"

    def _invalidate_agent_cache(self) -> None:
        """使 Agent 缓存失效。"""
        with self._agent_lock:
            self._current_agent = None
            self._current_agent_signature = ""

    # ------------------------------------------------------------------
    # 响应后处理（扩展点）
    # ------------------------------------------------------------------

    def _run_response_processors(self, response_text: str) -> str:
        """运行卡片注册的响应后处理器。

        主机不关心处理器做了什么，仅负责调用。
        处理器可以修改响应文本、触发 UI 展示等。
        """
        cleaned = response_text
        for processor in self.host_context.response_processors:
            try:
                cleaned, handled = processor.process(cleaned)
                if handled:
                    logger.debug("响应已被处理器处理: %s", processor.name)
            except Exception as exc:
                logger.warning("响应后处理器执行失败 [%s]: %s", processor.name, exc)
        return cleaned

    # ------------------------------------------------------------------
    # 执行模式
    # ------------------------------------------------------------------

    def _run_agent(self, agent: AgentExecutor, user_text: str) -> str:
        """Agent 模式执行。"""
        # 构建 system prompt
        system_prompt = self.host_context.persona_holder.current_persona
        
        # 注入用户画像
        user_profile = self.host_context.user_profile
        if user_profile:
            profile_text = user_profile.to_prompt_text()
            if profile_text:
                system_prompt += f"\n\n{profile_text}"

        # 使用卡片自定义 system prompt（如果有）
        agent_config = self.host_context.agent_configurator.get_config()
        if agent_config.get("custom_system_prompt"):
            system_prompt = agent_config["custom_system_prompt"]

        messages = [LLMMessage(role="system", content=system_prompt)]

        # 知识检索
        knowledge_chunks = self.host_context.knowledge_hub.retrieve(user_text, top_k=3)
        if knowledge_chunks:
            context_block = "\n".join(knowledge_chunks)
            messages.append(LLMMessage(
                role="system",
                content=f"参考信息:\n{context_block}\n请基于以上参考信息回答。",
            ))

        # 对话记忆
        if self.config.llm.memory_enabled:
            history = self.conversation_memory.get_history(
                max_rounds=self.config.llm.max_memory_rounds,
                max_tokens=self.config.llm.max_memory_tokens,
            )
            for hist_msg in history:
                messages.append(LLMMessage(role=hist_msg["role"], content=hist_msg["content"]))

        messages.append(LLMMessage(role="user", content=user_text))

        # 执行
        state = AgentState(messages=messages)
        callbacks = CallbackManager([LoggingHandler(), DisplayCallbackHandler(self.display)])
        config = RunnableConfig(callbacks=callbacks, recursion_limit=5)

        callbacks.on_thinking_start("agent_run")
        final_state = agent.invoke(state, config)
        callbacks.on_thinking_end("agent_run")

        for msg in reversed(final_state.messages):
            if msg.role == "assistant":
                return msg.content
        return "抱歉，我没有得到答案。"

    def _run_direct_llm(self, user_text: str) -> str:
        """直接 LLM 模式（无卡 / 无工具时）。"""
        system_prompt = self.host_context.persona_holder.current_persona
        
        # 注入用户画像
        user_profile = self.host_context.user_profile
        if user_profile:
            profile_text = user_profile.to_prompt_text()
            if profile_text:
                system_prompt += f"\n\n{profile_text}"

        messages = [LLMMessage(role="system", content=system_prompt)]

        # 知识检索
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
        """TTS 合成并播放。"""
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
