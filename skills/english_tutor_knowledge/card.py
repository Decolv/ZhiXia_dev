"""英语考试知识卡

提供英语考试相关的知识检索功能，包括：
- 听力技巧与真题解析
- 写作模板与范文
- 词汇表与记忆方法
- 长难句分析与理解

实现 KnowledgeProvider 接口，支持技能卡解耦访问。
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from zhixia.core.card_base import (
    CardManifest,
    ContentAwareKnowledgeCard,
    HostContext,
    ListeningMaterial,
    Sentence,
    VocabularyItem,
    WritingExample,
)
from zhixia.llm.rag.base import RAGContext, RAGRetriever


class SimpleKeywordRetriever(RAGRetriever):
    """简单关键词检索器（示例实现）。

    生产环境建议替换为 ChromaStore 等向量检索器。
    """

    def __init__(self, documents: Dict[str, str]) -> None:
        self.documents = documents

    @property
    def name(self) -> str:
        return "simple_keyword"

    def retrieve(self, query: str, top_k: int = 3) -> RAGContext:
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) >= 2]
        if not keywords:
            keywords = list(query_lower)

        scored = []
        for title, content in self.documents.items():
            score = sum(1 for k in keywords if k in content.lower())
            if score > 0:
                scored.append((score, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [content for _, content in scored[:top_k]]
        return RAGContext(chunks=chunks)


class EnglishTutorKnowledge(ContentAwareKnowledgeCard):
    """英语考试知识卡。

    实现 KnowledgeProvider 接口，提供标准化的知识内容访问。
    支持技能卡通过接口获取内容，实现解耦。
    """

    # 内容类型声明
    CONTENT_TYPES = ["listening", "sentences", "writing", "vocabulary"]
    SUPPORTED_EXAMS = ["cet4", "cet6", "ielts"]

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        self._retriever: Optional[SimpleKeywordRetriever] = None
        # 缓存解析后的内容
        self._listening_cache: Optional[List[ListeningMaterial]] = None
        self._sentences_cache: Optional[List[Sentence]] = None
        self._writing_cache: Optional[List[WritingExample]] = None
        self._vocabulary_cache: Optional[Dict[str, List[VocabularyItem]]] = None

    @property
    def content_types(self) -> List[str]:
        """返回知识卡提供的内容类型列表。"""
        return self.CONTENT_TYPES

    @property
    def supported_exams(self) -> List[str]:
        """返回支持的考试类型列表。"""
        return self.SUPPORTED_EXAMS

    def on_mount(self, host: HostContext) -> None:
        """插卡时：加载文档 -> 构建检索器 -> 注册到主机。"""
        docs = self._load_documents()
        self._retriever = SimpleKeywordRetriever(docs)
        host.knowledge_hub.register_retriever(self.name, self._retriever)

        print(f"[MOUNT] 英语考试知识卡已插入: {self.display_name}")
        print(f"   文档: {list(docs.keys())}")
        print(f"   检索: 关键词检索 ({len(docs)} 篇文档)")
        print(f"   内容类型: {', '.join(self.content_types)}")
        print(f"   支持考试: {', '.join(self.supported_exams)}")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销知识检索器。"""
        host.knowledge_hub.unregister_retriever(self.name)
        self._retriever = None
        # 清空缓存
        self._listening_cache = None
        self._sentences_cache = None
        self._writing_cache = None
        self._vocabulary_cache = None
        print(f"[UNMOUNT] 英语考试知识卡已拔出: {self.display_name}")

    def get_retriever(self) -> RAGRetriever:
        if self._retriever is None:
            docs = self._load_documents()
            return SimpleKeywordRetriever(docs)
        return self._retriever

    def get_assets(self) -> Dict[str, Path]:
        """返回资源映射。"""
        return {}

    # -------------------------------------------------------------------------
    # KnowledgeProvider 接口实现
    # -------------------------------------------------------------------------

    def get_listening_materials(
        self,
        exam_type: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> List[ListeningMaterial]:
        """获取听力材料。

        Args:
            exam_type: 考试类型过滤 (cet4/cet6/ielts)
            difficulty: 难度过滤 (beginner/intermediate/advanced)

        Returns:
            听力材料列表
        """
        if self._listening_cache is None:
            self._listening_cache = self._parse_listening_materials()

        materials = self._listening_cache

        # 应用过滤条件
        if exam_type:
            materials = [m for m in materials if m.exam_type == exam_type.lower()]
        if difficulty:
            materials = [m for m in materials if m.difficulty == difficulty.lower()]

        return materials

    def get_sentences(
        self,
        difficulty: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Sentence]:
        """获取长难句。

        Args:
            difficulty: 难度过滤 (beginner/intermediate/advanced)
            source: 来源过滤 (economist/nytimes)

        Returns:
            长难句列表
        """
        if self._sentences_cache is None:
            self._sentences_cache = self._parse_sentences()

        sentences = self._sentences_cache

        # 应用过滤条件
        if difficulty:
            sentences = [s for s in sentences if s.difficulty == difficulty.lower()]
        if source:
            sentences = [s for s in sentences if s.source == source.lower()]

        return sentences

    def get_writing_examples(
        self,
        exam_type: Optional[str] = None,
        essay_type: Optional[str] = None
    ) -> List[WritingExample]:
        """获取作文范文。

        Args:
            exam_type: 考试类型过滤 (cet4/cet6/ielts)
            essay_type: 作文类型过滤 (argumentation/narration/exposition)

        Returns:
            作文范文列表
        """
        if self._writing_cache is None:
            self._writing_cache = self._parse_writing_examples()

        examples = self._writing_cache

        # 应用过滤条件
        if exam_type:
            examples = [e for e in examples if e.exam_type == exam_type.lower()]
        if essay_type:
            examples = [e for e in examples if e.essay_type == essay_type.lower()]

        return examples

    def get_vocabulary(
        self,
        exam_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[VocabularyItem]:
        """获取词汇。

        Args:
            exam_type: 考试类型过滤 (cet4/cet6/ielts)
            category: 词汇分类

        Returns:
            词汇列表
        """
        if self._vocabulary_cache is None:
            self._vocabulary_cache = self._parse_vocabulary()

        # 确定要查询的词汇表
        if exam_type and exam_type.lower() in self._vocabulary_cache:
            vocab_list = self._vocabulary_cache[exam_type.lower()]
        else:
            # 返回所有词汇
            vocab_list = []
            for vlist in self._vocabulary_cache.values():
                vocab_list.extend(vlist)

        return vocab_list

    # -------------------------------------------------------------------------
    # 内容解析方法
    # -------------------------------------------------------------------------

    def _load_documents(self) -> Dict[str, str]:
        """加载 docs/ 目录及其子目录下的所有 Markdown 文档。"""
        docs_dir = self.card_root / "docs"
        documents = {}
        if not docs_dir.exists() or not docs_dir.is_dir():
            return documents

        for doc_path in sorted(docs_dir.rglob("*.md")):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    relative_path = doc_path.relative_to(docs_dir)
                    documents[str(relative_path.with_suffix(""))] = f.read()
            except Exception as exc:
                print(f"   警告: 无法读取文档 {doc_path}: {exc}")

        return documents

    def _parse_listening_materials(self) -> List[ListeningMaterial]:
        """解析听力材料。"""
        materials = []
        listening_dir = self.card_root / "docs" / "listening"

        if not listening_dir.exists():
            return materials

        for exam_type in self.SUPPORTED_EXAMS:
            exam_dir = listening_dir / exam_type
            if not exam_dir.exists():
                continue

            for file_path in exam_dir.glob("*.md"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 解析内容
                    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
                    title = title_match.group(1) if title_match else file_path.stem

                    original_match = re.search(
                        r"## 原文\s*\n+(.+?)(?=\n## 翻译|$)", content, re.DOTALL
                    )
                    original = original_match.group(1).strip() if original_match else ""

                    translation_match = re.search(
                        r"## 翻译\s*\n+(.+?)(?=\n## 元数据|$)", content, re.DOTALL
                    )
                    translation = translation_match.group(1).strip() if translation_match else ""

                    # 解析元数据
                    metadata = {}
                    metadata_match = re.search(r"## 元数据\s*\n(.+)$", content, re.DOTALL)
                    if metadata_match:
                        for line in metadata_match.group(1).split("\n"):
                            if line.strip().startswith("-"):
                                parts = line.strip()[1:].strip().split(":", 1)
                                if len(parts) == 2:
                                    metadata[parts[0].strip()] = parts[1].strip()

                    material = ListeningMaterial(
                        id=f"{exam_type}_{file_path.stem}",
                        title=title,
                        exam_type=exam_type,
                        original_text=original,
                        translation=translation,
                        difficulty=metadata.get("难度", "intermediate"),
                        topic=metadata.get("话题", ""),
                        material_type=metadata.get("类型", ""),
                    )
                    materials.append(material)

                except Exception as e:
                    print(f"   警告: 解析听力材料失败 {file_path}: {e}")

        return materials

    def _parse_sentences(self) -> List[Sentence]:
        """解析长难句。"""
        sentences = []
        sentences_dir = self.card_root / "docs" / "sentences"

        if not sentences_dir.exists():
            return sentences

        # 按难度解析
        difficulty_dir = sentences_dir / "by_difficulty"
        if difficulty_dir.exists():
            for diff_file in difficulty_dir.glob("*.md"):
                difficulty = diff_file.stem  # beginner/intermediate/advanced
                try:
                    with open(diff_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 解析句子块
                    sentence_blocks = re.split(r"## 句子\d+", content)
                    for i, block in enumerate(sentence_blocks[1:], 1):
                        original_match = re.search(r"### 原句\s*\n([^\n]+)", block)
                        translation_match = re.search(r"### 翻译\s*\n([^#]+)", block)
                        grammar_match = re.search(r"### 语法分析\s*\n([^#]+)", block)

                        vocab_list = []
                        vocab_section = re.search(r"### 重点词汇\s*\n(.+?)(?=##|$)", block, re.DOTALL)
                        if vocab_section:
                            for match in re.findall(r"-\s*([^:]+):\s*(.+)", vocab_section.group(1)):
                                vocab_list.append({"word": match[0].strip(), "meaning": match[1].strip()})

                        if original_match:
                            sentence = Sentence(
                                id=f"sentence_{difficulty}_{i}",
                                original=original_match.group(1).strip(),
                                translation=translation_match.group(1).strip() if translation_match else "",
                                grammar_analysis=grammar_match.group(1).strip() if grammar_match else "",
                                vocabulary=vocab_list,
                                difficulty=difficulty,
                                source="",
                            )
                            sentences.append(sentence)

                except Exception as e:
                    print(f"   警告: 解析长难句失败 {diff_file}: {e}")

        return sentences

    def _parse_writing_examples(self) -> List[WritingExample]:
        """解析作文范文。"""
        examples = []
        writing_dir = self.card_root / "docs" / "writing" / "examples"

        if not writing_dir.exists():
            return examples

        for exam_type in self.SUPPORTED_EXAMS:
            exam_dir = writing_dir / exam_type
            if not exam_dir.exists():
                continue

            for file_path in exam_dir.glob("*.md"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 简单解析标题和内容
                    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
                    title = title_match.group(1) if title_match else file_path.stem

                    example = WritingExample(
                        id=f"{exam_type}_{file_path.stem}",
                        exam_type=exam_type,
                        essay_type="argumentation",  # 默认类型
                        title=title,
                        content=content,
                    )
                    examples.append(example)

                except Exception as e:
                    print(f"   警告: 解析作文范文失败 {file_path}: {e}")

        return examples

    def _parse_vocabulary(self) -> Dict[str, List[VocabularyItem]]:
        """解析词汇库。"""
        vocab_dict: Dict[str, List[VocabularyItem]] = {}
        vocab_dir = self.card_root / "docs" / "vocabulary"

        if not vocab_dir.exists():
            return vocab_dict

        vocab_files = {
            "cet4": "cet4_core.md",
            "cet6": "cet6_core.md",
            "ielts": "ielts_academic.md",
        }

        for exam_type, filename in vocab_files.items():
            file_path = vocab_dir / filename
            if not file_path.exists():
                continue

            vocab_list = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 解析词汇条目
                pattern = r"###\s+(\w+)\n- 单词:\s*(.+?)\n- 音标:\s*(.+?)\n- 词性:\s*(.+?)\n- 释义:\s*(.+?)\n- 例句:\s*(.+?)\n- 翻译:\s*(.+?)\n- 记忆法:\s*(.+?)(?=\n###|\Z)"
                matches = re.findall(pattern, content, re.DOTALL)

                for match in matches:
                    word_id, word, phonetic, pos, meaning, example, translation, memory = match
                    vocab_item = VocabularyItem(
                        id=word_id.strip(),
                        word=word.strip(),
                        phonetic=phonetic.strip(),
                        pos=pos.strip(),
                        meaning=meaning.strip(),
                        example=example.strip(),
                        translation=translation.strip(),
                        memory_tip=memory.strip(),
                    )
                    vocab_list.append(vocab_item)

                vocab_dict[exam_type] = vocab_list

            except Exception as e:
                print(f"   警告: 解析词汇失败 {file_path}: {e}")

        return vocab_dict
