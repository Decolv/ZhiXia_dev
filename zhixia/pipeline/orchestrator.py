"""VoicePipeline — ASR → (RAG) → LLM → TTS → play 流水线

延迟优化策略：
1. LLM 流式输出，边生成边分句
2. 分句后立即送 TTS 合成（内存合成，无磁盘 I/O）
3. TTS 合成完一句立即播放，不等后续句子
4. LLM / TTS / Play 三阶段通过队列并发，首句延迟 = ASR + LLM首句 + TTS首句
"""

import logging
import queue
import re
import threading
import time
from pathlib import Path
from typing import Optional

from zhixia.asr.base import ASREngine, ASRResult
from zhixia.audio.base import AudioPlayer
from zhixia.config.settings import AppSettings
from zhixia.display.base import DisplayOutput, DisplayPayload
from zhixia.llm.base import LLMEngine, LLMMessage, StructuredOutput
from zhixia.llm.output_parser import _strip_thinking_tokens, get_format_instruction, parse_llm_output
from zhixia.llm.rag.base import RAGContext, RAGRetriever
from zhixia.tts.base import TTSEngine

logger = logging.getLogger(__name__)

# 分句正则：中文句末标点 + 英文句末标点
_SENTENCE_END = re.compile(r'[。！？!?…]+|(?<=[^0-9])[.]+(?=[^0-9])|[；;]+')
# 最短触发 TTS 的字符数（避免太短的片段频繁合成）
_MIN_CHUNK_LEN = 8
# 流水线结束哨兵
_SENTINEL = object()


def _split_sentences(text: str) -> list[str]:
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


class VoicePipeline:

    def __init__(
        self,
        config: AppSettings,
        asr_engine: ASREngine,
        llm_engine: LLMEngine,
        tts_engine: TTSEngine,
        audio_player: AudioPlayer,
        rag_retriever: Optional[RAGRetriever] = None,
        display: Optional[DisplayOutput] = None,
    ) -> None:
        self.config = config
        self.asr_engine = asr_engine
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.audio_player = audio_player

        if rag_retriever is None:
            from zhixia.llm.rag.null_retriever import NullRAGRetriever
            self.rag_retriever = NullRAGRetriever()
        else:
            self.rag_retriever = rag_retriever

        if display is None:
            from zhixia.display.null_display import NullDisplay
            self.display = NullDisplay()
        else:
            self.display = display

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def process_audio(self, audio_path: Path) -> None:
        t0 = time.perf_counter()
        print("\n" + "=" * 60)
        print("🎙️ 语音助手处理开始（流水线模式）")
        print("=" * 60)

        # 1. ASR
        t_asr = time.perf_counter()
        print("\n[1/3] 语音识别中...")
        asr_result = self.asr_engine.transcribe(audio_path)
        if not asr_result.text:
            raise RuntimeError("语音识别失败")
        print(f"✅ 识别文本: {asr_result.text!r}  ({time.perf_counter()-t_asr:.2f}s)")

        # 2. RAG（可选）
        rag_context = None
        if self.config.rag.enabled:
            rag_context = self.rag_retriever.retrieve(asr_result.text, self.config.rag.top_k)

        # 3. LLM → TTS → Play 流水线
        print("\n[2/3] AI 思考 + 语音合成（流水线）...")
        full_text = self._run_streaming_pipeline(asr_result.text, rag_context)

        # 4. 解析最终输出（用于 display）
        structured = parse_llm_output(full_text)
        print(f"\n✅ AI 完整回复: {structured.text!r}")

        # 更新 display：显示完整的 text 和最终的 emotion
        payload = DisplayPayload(
            text=structured.text,
            emotion=structured.emotion,
            is_thinking=False,
            metadata=structured.metadata,
        )
        self.display.show(payload)

        print(f"\n⏱️  总耗时: {time.perf_counter()-t0:.2f}s")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 流水线核心
    # ------------------------------------------------------------------

    def _run_streaming_pipeline(self, user_text: str, rag_context: Optional[RAGContext]) -> str:
        """
        三线程流水线：
          Thread-A (LLM)  → tts_queue
          Thread-B (TTS)  → play_queue
          Thread-C (Play) → 播放
        主线程等待 Thread-C 结束。
        返回 LLM 完整原始输出。
        """
        tts_queue: queue.Queue = queue.Queue(maxsize=4)
        play_queue: queue.Queue = queue.Queue(maxsize=4)
        full_output_holder: list[str] = []
        errors: list[Exception] = []

        messages = self._build_messages(user_text, rag_context)
        # 如果启用了结构化输出，禁用分句（避免破坏 JSON 格式）
        enable_sentence_split = not self.config.llm.enable_structured_output

        def llm_worker():
            try:
                buffer = ""
                raw_chunks = []
                emotion_shown = False  # 标记 emotion 是否已显示
                # 结构化输出流式状态
                in_text_value = False   # 是否正在接收 "text" 值内容
                text_buf = ""           # text 值的流式缓冲区
                sent_chars = 0          # 已送入 TTS 的字符数
                for token in self.llm_engine.stream_chat(messages, self.config.llm.max_new_tokens):
                    raw_chunks.append(token)
                    buffer += token

                    # 如果启用了结构化输出，尝试立即提取并显示 emotion
                    if not emotion_shown and self.config.llm.enable_structured_output and '"emotion"' in buffer:
                        # 尝试提取 emotion 值
                        emo_idx = buffer.find('"emotion"')
                        if emo_idx >= 0:
                            rest = buffer[emo_idx + 9:]
                            quote_idx = rest.find('"')
                            if quote_idx >= 0:
                                value_start = emo_idx + 9 + quote_idx + 1
                                value_end = -1
                                for i in range(value_start, len(buffer)):
                                    if buffer[i] == '"' and (i == 0 or buffer[i - 1] != '\\'):
                                        value_end = i
                                        break
                                if value_end > 0:
                                    emotion_value = buffer[value_start:value_end]
                                    payload = DisplayPayload(
                                        text="",
                                        emotion=emotion_value,
                                        is_thinking=False,
                                        metadata={}
                                    )
                                    self.display.show(payload)
                                    emotion_shown = True
                                    logger.debug("立即显示 emotion: %s", emotion_value)

                    if enable_sentence_split:
                        # 非结构化输出：直接分句
                        sentences = _split_sentences(buffer)
                        if len(sentences) > 1:
                            for s in sentences[:-1]:
                                clean = _strip_thinking_tokens(s)
                                if clean and len(clean) >= _MIN_CHUNK_LEN:
                                    tts_queue.put(clean)
                            buffer = sentences[-1]
                    else:
                        # 结构化输出：流式 JSON 解析
                        if not in_text_value:
                            # 还没进入 text 值，检测 "text" 字段的开始引号
                            if '"text"' in buffer:
                                idx = buffer.find('"text"')
                                rest = buffer[idx + 6:]
                                quote_idx = rest.find('"')
                                if quote_idx >= 0:
                                    # 找到了 text 值的开始引号，进入流式模式
                                    text_start = idx + 6 + quote_idx + 1
                                    in_text_value = True
                                    text_buf = buffer[text_start:]
                                    sent_chars = 0
                                    buffer = ""  # 清空 buffer，后续 token 不再追加到 buffer
                        else:
                            # 正在接收 text 值内容，追加到 text_buf
                            text_buf += token
                            # 检查新追加的部分是否包含闭合引号（text 值结束）
                            close_idx = -1
                            # 从上次检查位置开始，避免重复检查
                            search_start = max(0, len(text_buf) - len(token) - 1)
                            for i in range(search_start, len(text_buf)):
                                if text_buf[i] == '"' and (i == 0 or text_buf[i - 1] != '\\'):
                                    close_idx = i
                                    break
                            if close_idx >= 0:
                                # text 值结束，发送剩余未送出的内容
                                remaining = text_buf[sent_chars:close_idx]
                                if remaining.strip():
                                    clean = _strip_thinking_tokens(remaining)
                                    if clean:
                                        tts_queue.put(clean)
                                in_text_value = False
                                text_buf = ""
                                # buffer 保持为空，丢弃后续 JSON 碎片（emotion 已提取）
                                buffer = ""
                            else:
                                # text 值还在继续，尝试流式分句
                                current = text_buf[sent_chars:]
                                sentences = _split_sentences(current)
                                if len(sentences) > 1:
                                    for s in sentences[:-1]:
                                        clean = _strip_thinking_tokens(s)
                                        if clean and len(clean) >= _MIN_CHUNK_LEN:
                                            tts_queue.put(clean)
                                    sent_chars = len(text_buf) - len(sentences[-1])

                # 处理剩余
                if in_text_value and text_buf[sent_chars:].strip():
                    # text 值未正常闭合（LLM 截断），发送剩余内容
                    clean = _strip_thinking_tokens(text_buf[sent_chars:])
                    if clean:
                        tts_queue.put(clean)
                elif enable_sentence_split and buffer.strip():
                    # 非结构化模式：发送 buffer 中剩余的未完成句子
                    clean = _strip_thinking_tokens(buffer)
                    if clean:
                        tts_queue.put(clean)

                full_output_holder.append("".join(raw_chunks))
            except Exception as e:
                logger.exception("LLM worker 异常")
                errors.append(e)
            finally:
                tts_queue.put(_SENTINEL)

        def tts_worker():
            try:
                while True:
                    item = tts_queue.get()
                    if item is _SENTINEL:
                        break
                    # item 已经是纯文本（由 llm_worker 提取），直接合成
                    t = time.perf_counter()
                    wav = self.tts_engine.synthesize_to_bytes(item)
                    logger.debug("TTS 合成 %.2fs: %s", time.perf_counter() - t, item[:30])
                    if wav:
                        play_queue.put(wav)
            except Exception as e:
                logger.exception("TTS worker 异常")
                errors.append(e)
            finally:
                play_queue.put(_SENTINEL)

        def play_worker():
            first = True
            try:
                while True:
                    item = play_queue.get()
                    if item is _SENTINEL:
                        break
                    if first:
                        print(f"🔊 首句播放开始")
                        first = False
                    self.audio_player.play_bytes(item, blocking=True)
            except Exception as e:
                logger.exception("Play worker 异常")
                errors.append(e)

        t_llm = threading.Thread(target=llm_worker, daemon=True, name="llm-worker")
        t_tts = threading.Thread(target=tts_worker, daemon=True, name="tts-worker")
        t_play = threading.Thread(target=play_worker, daemon=True, name="play-worker")

        t_llm.start()
        t_tts.start()
        t_play.start()

        t_llm.join()
        t_tts.join()
        t_play.join()

        if errors:
            raise errors[0]

        return full_output_holder[0] if full_output_holder else ""

    # ------------------------------------------------------------------
    # 消息构建
    # ------------------------------------------------------------------

    def _build_messages(self, user_text: str, rag_context: Optional[RAGContext]) -> list[LLMMessage]:
        messages = []
        system_prompt = self.config.llm.system_prompt
        # 低延迟场景：关闭结构化输出（避免 LLM 生成 JSON 包装）
        if self.config.llm.enable_structured_output:
            system_prompt = system_prompt + get_format_instruction()
        messages.append(LLMMessage(role="system", content=system_prompt))

        if rag_context and rag_context.chunks:
            context_block = "\n".join(rag_context.chunks)
            messages.append(LLMMessage(
                role="system",
                content=f"参考信息:\n{context_block}\n请基于以上参考信息回答用户问题。",
            ))

        messages.append(LLMMessage(role="user", content=user_text))
        return messages
