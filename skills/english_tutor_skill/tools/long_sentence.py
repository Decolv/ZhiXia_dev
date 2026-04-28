"""长难句助力器工具 - 帮助用户学习和理解长难句"""

from typing import Optional, Dict, List, Any
from zhixia.agent.tool import Tool


class KnowledgeProvider:
    """知识提供者接口定义"""

    def get_sentences(
        self,
        difficulty: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取句子列表

        Args:
            difficulty: 难度级别
            source: 来源

        Returns:
            句子列表
        """
        raise NotImplementedError


class LongSentenceTool(Tool):
    """长难句助力器工具：获取、解析和讲解长难句，帮助用户逐步理解复杂句子结构。

    支持三种操作：
    - get_sentence: 获取长难句及基础解析
    - explain: 详细讲解特定语法点
    - analyze: 完整分析句子结构
    """

    # 难度级别映射
    DIFFICULTY_LEVELS = {
        "beginner": "初级",
        "intermediate": "中级",
        "advanced": "高级"
    }

    # 来源映射
    SOURCES = {
        "economist": "经济学人",
        "nytimes": "纽约时报"
    }

    def __init__(
        self,
        llm_engine=None,
        knowledge_provider: Optional[KnowledgeProvider] = None
    ):
        super().__init__(
            name="long_sentence",
            description="""长难句助力器工具：获取、解析和讲解英语长难句。

参数说明：
- action: 操作类型 (get_sentence/explain/analyze)
  * get_sentence: 获取长难句及基础解析
  * explain: 详细讲解特定语法点
  * analyze: 完整分析句子结构
- difficulty: 难度级别 (beginner/intermediate/advanced)
- source: 来源 (economist/nytimes)
- sentence_id: 句子ID (如 "sentence1", "sentence2" 等)
- grammar_point: 语法点名称 (explain时使用)

使用示例：
1. 获取中级难度的长难句：action=get_sentence, difficulty=intermediate
2. 获取经济学人的长难句：action=get_sentence, source=economist
3. 讲解特定语法点：action=explain, grammar_point=定语从句
4. 分析特定句子：action=analyze, sentence_id=sentence1""",
            func=self._execute,
        )
        self._llm_engine = llm_engine
        self._knowledge_provider = knowledge_provider
        self._sentences_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _execute(
        self,
        action: str = "get_sentence",
        difficulty: Optional[str] = None,
        source: Optional[str] = None,
        sentence_id: Optional[str] = None,
        grammar_point: Optional[str] = None
    ) -> str:
        """执行长难句工具操作。

        Args:
            action: 操作类型 (get_sentence/explain/analyze)
            difficulty: 难度级别 (beginner/intermediate/advanced)
            source: 来源 (economist/nytimes)
            sentence_id: 句子ID
            grammar_point: 语法点名称

        Returns:
            根据action返回相应的结果字符串
        """
        action = action.lower()

        if action == "get_sentence":
            return self._get_sentence(difficulty, source, sentence_id)
        elif action == "explain":
            return self._explain_grammar(grammar_point)
        elif action == "analyze":
            return self._analyze_sentence(difficulty, source, sentence_id)
        else:
            return f"【错误】不支持的操作类型：{action}。请使用 get_sentence、explain 或 analyze。"

    def _get_sentences(
        self,
        difficulty: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取句子列表（带缓存）。

        Args:
            difficulty: 难度级别
            source: 来源

        Returns:
            句子列表
        """
        cache_key = f"{difficulty or 'all'}_{source or 'all'}"

        if cache_key in self._sentences_cache:
            return self._sentences_cache[cache_key]

        sentences = []
        if self._knowledge_provider:
            sentences = self._knowledge_provider.get_sentences(difficulty, source)

        self._sentences_cache[cache_key] = sentences
        return sentences

    def _get_sentence(
        self,
        difficulty: Optional[str],
        source: Optional[str],
        sentence_id: Optional[str]
    ) -> str:
        """获取长难句及基础解析。

        Args:
            difficulty: 难度级别
            source: 来源
            sentence_id: 句子ID

        Returns:
            格式化的句子信息
        """
        if not self._knowledge_provider:
            return self._get_fallback_response()

        sentences = self._get_sentences(difficulty, source)

        if not sentences:
            return self._get_no_data_response(difficulty, source)

        # 如果指定了sentence_id，查找特定句子
        if sentence_id:
            target = None
            for s in sentences:
                if s.get("id") == sentence_id.lower():
                    target = s
                    break
            if target:
                return self._format_sentence_basic(target)
            else:
                available_ids = ", ".join([s.get("id", "unknown") for s in sentences])
                return f"【错误】未找到句子 {sentence_id}。\n可用句子ID：{available_ids}"

        # 否则返回第一个句子
        return self._format_sentence_basic(sentences[0])

    def _get_fallback_response(self) -> str:
        """获取降级响应（无知识提供者时）。

        Returns:
            降级提示信息
        """
        return """【提示】长难句知识库暂不可用。

可能的原因：
  • 知识提供者未配置
  • 知识服务暂时不可用

您可以：
  1. 稍后重试
  2. 使用 explain 操作学习通用语法知识（无需知识库）
  3. 直接提供句子，我可以使用AI能力进行分析

示例：action=explain, grammar_point=定语从句"""

    def _get_no_data_response(
        self,
        difficulty: Optional[str],
        source: Optional[str]
    ) -> str:
        """获取无数据时的响应。

        Args:
            difficulty: 难度级别
            source: 来源

        Returns:
            无数据提示信息
        """
        filters = []
        if difficulty:
            filters.append(f"难度: {difficulty}")
        if source:
            filters.append(f"来源: {source}")

        filter_str = f"（{'，'.join(filters)}）" if filters else ""

        return f"""【提示】未找到符合条件的长难句{filter_str}。

可用选项：
  • 难度级别：beginner(初级), intermediate(中级), advanced(高级)
  • 来源：economist(经济学人), nytimes(纽约时报)

您可以：
  1. 更换筛选条件重试
  2. 使用 action=explain 学习通用语法知识
  3. 直接提供句子，我可以使用AI能力进行分析"""

    def _format_sentence_basic(self, sentence: Dict[str, Any]) -> str:
        """格式化基础句子信息。

        Args:
            sentence: 句子数据字典

        Returns:
            格式化的字符串
        """
        result = [
            "📚 长难句学习",
            "",
            f"【原句】{sentence.get('original', '')}",
            "",
            f"【翻译】{sentence.get('translation', '')}",
            "",
            "【语法结构】",
            sentence.get('grammar_analysis', ''),
            "",
            "【重点词汇】"
        ]

        vocabulary = sentence.get('vocabulary', [])
        if vocabulary:
            for vocab in vocabulary:
                word = vocab.get('word', '') if isinstance(vocab, dict) else str(vocab)
                meaning = vocab.get('meaning', '') if isinstance(vocab, dict) else ''
                result.append(f"  • {word}: {meaning}")
        else:
            result.append("  （无重点词汇）")

        result.extend([
            "",
            "💡 学习提示：",
            "  1. 先通读原句，尝试理解大意",
            "  2. 对照翻译，理解句子结构",
            "  3. 学习重点词汇和语法点",
            "  4. 如需详细分析，使用 action=analyze",
            f"  5. 句子ID: {sentence.get('id', 'unknown')}，可用于后续分析"
        ])

        return "\n".join(result)

    def _explain_grammar(self, grammar_point: Optional[str]) -> str:
        """详细讲解语法知识点。

        Args:
            grammar_point: 语法点名称

        Returns:
            语法讲解内容
        """
        if not grammar_point:
            return "【提示】请指定要讲解的语法点，例如：grammar_point=定语从句"

        # 语法知识库
        grammar_knowledge = {
            "定语从句": {
                "definition": "定语从句是用来修饰名词或代词的从句，通常由关系代词(who, whom, whose, which, that)或关系副词(when, where, why)引导。",
                "types": [
                    "限制性定语从句：对先行词起限定作用，去掉后主句意思不完整",
                    "非限制性定语从句：对先行词起补充说明作用，用逗号隔开"
                ],
                "examples": [
                    "The book that I bought yesterday is very interesting.",
                    "The man who is standing there is my teacher."
                ],
                "tips": "识别定语从句的关键是找到关系词，然后确定它指代的先行词是什么。"
            },
            "状语从句": {
                "definition": "状语从句在复合句中充当状语，修饰主句的动词、形容词或副词，表示时间、地点、原因、条件、目的、结果等。",
                "types": [
                    "时间状语从句：when, while, as, before, after, since, until等",
                    "条件状语从句：if, unless, as long as等",
                    "原因状语从句：because, since, as等",
                    "让步状语从句：although, though, even though等"
                ],
                "examples": [
                    "If it rains tomorrow, we will stay at home.",
                    "Although he is young, he knows a lot."
                ],
                "tips": "状语从句可以放在主句前或主句后，放在前面时常用逗号隔开。"
            },
            "名词性从句": {
                "definition": "名词性从句在句中充当名词的作用，可以作主语、宾语、表语或同位语。",
                "types": [
                    "主语从句：在句中作主语",
                    "宾语从句：在句中作宾语",
                    "表语从句：在句中作表语",
                    "同位语从句：解释说明名词的内容"
                ],
                "examples": [
                    "What he said is true. (主语从句)",
                    "I believe that he will come. (宾语从句)"
                ],
                "tips": "名词性从句通常由that, whether, if或疑问词引导。"
            },
            "非谓语动词": {
                "definition": "非谓语动词是指在句中不作谓语的动词形式，包括不定式、动名词和分词。",
                "types": [
                    "不定式(to do)：表示目的、将来或具体动作",
                    "动名词(doing)：表示一般性、习惯性动作",
                    "现在分词(doing)：表示主动、进行",
                    "过去分词(done)：表示被动、完成"
                ],
                "examples": [
                    "To learn English well is important.",
                    "I enjoy reading books.",
                    "The man standing there is my teacher."
                ],
                "tips": "非谓语动词可以充当多种句子成分，是英语语法中的重点和难点。"
            },
            "倒装": {
                "definition": "倒装是指将句子的主语和谓语位置互换，或将谓语的一部分提到主语之前。",
                "types": [
                    "完全倒装：整个谓语放在主语之前",
                    "部分倒装：只将助动词/情态动词放在主语之前"
                ],
                "examples": [
                    "Here comes the bus. (完全倒装)",
                    "Never have I seen such a beautiful sight. (部分倒装)"
                ],
                "tips": "否定词开头、only+状语开头、so/neither/nor等情况下常使用倒装。"
            },
            "虚拟语气": {
                "definition": "虚拟语气用来表示与事实相反、不可能发生或可能性很小的情况。",
                "types": [
                    "与现在事实相反：if + 过去式, would + 动词原形",
                    "与过去事实相反：if + had done, would have done",
                    "与将来事实相反：if + should/were to, would + 动词原形"
                ],
                "examples": [
                    "If I were you, I would study harder.",
                    "If he had come yesterday, he would have met her."
                ],
                "tips": "虚拟语气中，be动词通常用were，不用was（第一、三人称单数也如此）。"
            },
            "独立主格": {
                "definition": "独立主格结构是由名词/代词+非谓语动词/形容词/副词/介词短语构成，在句中作状语，表示时间、原因、条件、伴随等。",
                "structure": "名词/代词 + (doing/done/to do/adj./adv./prep. phrase)",
                "examples": [
                    "Weather permitting, we will go camping.",
                    "All things considered, his proposal is reasonable."
                ],
                "tips": "独立主格结构有自己的逻辑主语，与主句主语不同。"
            },
            "强调句": {
                "definition": "强调句用于突出句子的某一成分，基本结构为：It is/was + 被强调部分 + that/who + 其他。",
                "usage": "可以强调主语、宾语、状语，但不能强调谓语。",
                "examples": [
                    "It was John who broke the window.",
                    "It was yesterday that I met him."
                ],
                "tips": "去掉It is/was...that后，剩下的部分应该能构成完整的句子。"
            },
            "插入语": {
                "definition": "插入语是插在句子中间的成分，对句子进行补充说明，去掉后不影响句子的完整性。",
                "types": [
                    "副词作插入语：however, therefore, obviously等",
                    "短语作插入语：in fact, for example等",
                    "从句作插入语：I think, I believe等"
                ],
                "examples": [
                    "This book, however, is too difficult for beginners.",
                    "He is, in fact, a very talented musician."
                ],
                "tips": "阅读时可以先忽略插入语，抓住句子主干。"
            },
            "同位语": {
                "definition": "同位语是对名词或代词进行解释说明的成分，与被修饰词指同一人或事物。",
                "types": [
                    "名词作同位语",
                    "短语作同位语",
                    "从句作同位语（同位语从句）"
                ],
                "examples": [
                    "My friend Tom is coming.",
                    "The news that he won the prize surprised us."
                ],
                "tips": "同位语从句通常由that引导，that在从句中不充当成分。"
            }
        }

        # 查找匹配的语法点
        matched_point = None
        matched_name = None
        for key, value in grammar_knowledge.items():
            if key in grammar_point or grammar_point in key:
                matched_point = value
                matched_name = key
                break

        if not matched_point:
            available_points = ", ".join(grammar_knowledge.keys())
            return f"【提示】暂未找到'{grammar_point}'的详细讲解。\n\n可用语法点：{available_points}\n\n您可以使用LLM获取更详细的讲解。"

        # 格式化输出
        result = [
            f"📖 语法点讲解：{matched_name}",
            "",
            f"【定义】{matched_point['definition']}",
            ""
        ]

        if 'types' in matched_point:
            result.append("【分类/用法】")
            for t in matched_point['types']:
                result.append(f"  • {t}")
            result.append("")

        if 'structure' in matched_point:
            result.append(f"【结构】{matched_point['structure']}")
            result.append("")

        if 'usage' in matched_point:
            result.append(f"【用法】{matched_point['usage']}")
            result.append("")

        result.append("【例句】")
        for i, ex in enumerate(matched_point['examples'], 1):
            result.append(f"  {i}. {ex}")
        result.append("")

        result.append(f"💡 学习技巧：{matched_point['tips']}")

        return "\n".join(result)

    def _analyze_sentence(
        self,
        difficulty: Optional[str],
        source: Optional[str],
        sentence_id: Optional[str]
    ) -> str:
        """完整分析句子结构。

        Args:
            difficulty: 难度级别
            source: 来源
            sentence_id: 句子ID

        Returns:
            详细的句子分析报告
        """
        if not self._knowledge_provider:
            return self._get_fallback_response()

        sentences = self._get_sentences(difficulty, source)

        if not sentences:
            return self._get_no_data_response(difficulty, source)

        # 查找目标句子
        target = None
        if sentence_id:
            for s in sentences:
                if s.get("id") == sentence_id.lower():
                    target = s
                    break
            if not target:
                available_ids = ", ".join([s.get("id", "unknown") for s in sentences])
                return f"【错误】未找到句子 {sentence_id}。\n可用句子ID：{available_ids}"
        else:
            target = sentences[0]

        # 构建详细分析
        result = [
            "🔍 长难句深度分析",
            "",
            f"【原句】{target.get('original', '')}",
            "",
            f"【翻译】{target.get('translation', '')}",
            "",
            "=" * 50,
            "",
            "📋 逐步解析：",
            "",
            "第一步：识别句子主干",
            "  • 找出主语、谓语、宾语（或表语）",
            "  • 暂时忽略修饰成分，理解核心意思",
            "",
            "第二步：分析从句和修饰成分",
            f"  {target.get('grammar_analysis', '')}",
            "",
            "第三步：理解逻辑关系",
            "  • 确定各成分之间的逻辑联系",
            "  • 理解作者想表达的核心信息",
            "",
            "=" * 50,
            "",
            "📚 重点词汇详解："
        ]

        vocabulary = target.get('vocabulary', [])
        if vocabulary:
            for vocab in vocabulary:
                word = vocab.get('word', '') if isinstance(vocab, dict) else str(vocab)
                meaning = vocab.get('meaning', '') if isinstance(vocab, dict) else ''
                result.append(f"  • {word}")
                result.append(f"    释义：{meaning}")
                result.append("")
        else:
            result.append("  （无重点词汇）")
            result.append("")

        result.extend([
            "=" * 50,
            "",
            "💡 学习建议：",
            "  1. 背诵原句，培养语感",
            "  2. 模仿造句，运用所学语法",
            "  3. 定期复习，巩固记忆",
            "  4. 如需讲解特定语法点，使用 action=explain",
            f"  5. 句子ID: {target.get('id', 'unknown')}"
        ])

        # 如果有LLM引擎，添加智能分析
        if self._llm_engine:
            result.extend([
                "",
                "🤖 AI 深度分析：",
                "  正在生成..."
            ])
            # 这里可以调用LLM进行更深入的分析

        return "\n".join(result)

    def get_all_sentences(
        self,
        difficulty: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取所有句子列表（用于外部调用）。

        Args:
            difficulty: 难度级别
            source: 来源

        Returns:
            句子列表
        """
        return self._get_sentences(difficulty, source)

    def get_sentence_by_id(
        self,
        sentence_id: str,
        difficulty: Optional[str] = None,
        source: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """根据ID获取特定句子（用于外部调用）。

        Args:
            sentence_id: 句子ID
            difficulty: 难度级别
            source: 来源

        Returns:
            句子数据或None
        """
        sentences = self.get_all_sentences(difficulty, source)
        for s in sentences:
            if s.get("id") == sentence_id.lower():
                return s
        return None
