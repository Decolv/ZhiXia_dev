"""作文辅导器工具 - 提供范文、思路建议、万能句推荐和作文润色"""

import os
import re
from typing import Optional
from zhixia.agent.tool import Tool


class WritingAssistantTool(Tool):
    """作文辅导器工具：提供范文获取、思路建议、万能句推荐和作文润色功能。

    支持 CET-4、CET-6、IELTS 等英语考试写作辅导
    """

    KNOWLEDGE_BASE_PATH = "d:\\Code\\ZhiXia_dev\\skills\\english_tutor_knowledge\\docs\\writing"

    def __init__(self, llm_engine=None):
        super().__init__(
            name="writing_assistant",
            description="作文辅导器工具：获取范文、提供写作思路、推荐万能句、润色作文。参数：action（操作类型：get_example/get_ideas/get_sentences/polish）、exam_type（考试类型：cet4/cet6/ielts）、topic（作文题目/话题）、essay_type（作文类型：argumentation/narration/exposition）、user_essay（用户作文，polish时使用）",
            func=self._execute,
        )
        self._llm_engine = llm_engine

    def _execute(
        self,
        action: str = "get_example",
        exam_type: str = "cet4",
        topic: str = "",
        essay_type: str = "argumentation",
        user_essay: str = "",
    ) -> str:
        """执行作文辅导功能。

        Args:
            action: 操作类型 - get_example(获取范文)/get_ideas(思路建议)/get_sentences(万能句推荐)/polish(作文润色)
            exam_type: 考试类型 - cet4/cet6/ielts
            topic: 作文题目或话题
            essay_type: 作文类型 - argumentation(议论文)/narration(记叙文)/exposition(说明文)
            user_essay: 用户作文内容（polish时使用）

        Returns:
            相应的辅导内容
        """
        action = action.lower()
        exam_type = exam_type.lower()
        essay_type = essay_type.lower()

        if action == "get_example":
            return self._get_example(exam_type, topic)
        elif action == "get_ideas":
            return self._get_ideas(exam_type, topic, essay_type)
        elif action == "get_sentences":
            return self._get_sentences(exam_type, essay_type, topic)
        elif action == "polish":
            return self._polish_essay(exam_type, topic, user_essay)
        else:
            return f"【错误】不支持的操作类型：{action}。请使用：get_example/get_ideas/get_sentences/polish"

    def _get_example(self, exam_type: str, topic: str) -> str:
        """获取范文及点评。"""
        example_path = os.path.join(self.KNOWLEDGE_BASE_PATH, "examples", exam_type)

        if not os.path.exists(example_path):
            return f"【获取范文】\n\n考试类型：{exam_type.upper()}\n题目：{topic}\n\n[提示：暂无可用的范文示例]"

        # 读取该考试类型的所有范文
        examples = []
        for filename in sorted(os.listdir(example_path)):
            if filename.endswith(".md"):
                filepath = os.path.join(example_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        examples.append(content)
                except Exception as e:
                    continue

        if not examples:
            return f"【获取范文】\n\n考试类型：{exam_type.upper()}\n题目：{topic}\n\n[提示：暂无可用的范文示例]"

        # 组合输出
        result = f"【{exam_type.upper()} 范文示例】\n"
        if topic:
            result += f"\n您的题目：{topic}\n"
        result += "\n" + "=" * 50 + "\n"

        for i, example in enumerate(examples, 1):
            result += f"\n### 范文 {i}\n\n{example}\n"
            if i < len(examples):
                result += "\n" + "-" * 50 + "\n"

        result += "\n" + "=" * 50 + "\n"
        result += "\n💡 学习建议：\n"
        result += "1. 仔细分析范文的结构和逻辑\n"
        result += "2. 学习其中的高级词汇和句型\n"
        result += "3. 注意段落之间的衔接和过渡\n"
        result += "4. 尝试仿写，将学到的技巧应用到自己的作文中\n"

        return result

    def _get_ideas(self, exam_type: str, topic: str, essay_type: str) -> str:
        """针对题材提供写作思路。"""
        template_path = os.path.join(self.KNOWLEDGE_BASE_PATH, "templates", f"{essay_type}.md")

        result = f"【写作思路建议】\n\n"
        result += f"考试类型：{exam_type.upper()}\n"
        result += f"作文类型：{self._get_essay_type_name(essay_type)}\n"
        if topic:
            result += f"题目：{topic}\n"
        result += "\n" + "=" * 50 + "\n"

        # 读取模板
        template_content = ""
        if os.path.exists(template_path):
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    template_content = f.read()
            except Exception as e:
                template_content = ""

        if template_content:
            result += f"\n{template_content}\n"
        else:
            result += "\n[提示：暂无该类型的详细模板]\n"

        # 针对不同考试类型的建议
        result += "\n" + "=" * 50 + "\n"
        result += f"\n【{exam_type.upper()} 写作要点】\n"
        result += self._get_exam_tips(exam_type, essay_type)

        # 针对具体题目的思路拓展
        if topic and self._llm_engine:
            result += "\n" + "=" * 50 + "\n"
            result += f"\n【针对题目的思路拓展】\n"
            result += self._generate_topic_ideas(exam_type, essay_type, topic)

        return result

    def _get_sentences(self, exam_type: str, essay_type: str, topic: str) -> str:
        """推荐适用的万能句。"""
        sentences_path = os.path.join(self.KNOWLEDGE_BASE_PATH, "universal_sentences.md")

        result = f"【万能句推荐】\n\n"
        result += f"考试类型：{exam_type.upper()}\n"
        result += f"作文类型：{self._get_essay_type_name(essay_type)}\n"
        if topic:
            result += f"题目：{topic}\n"
        result += "\n" + "=" * 50 + "\n"

        # 读取万能句库
        sentences_content = ""
        if os.path.exists(sentences_path):
            try:
                with open(sentences_path, "r", encoding="utf-8") as f:
                    sentences_content = f.read()
            except Exception as e:
                sentences_content = ""

        if sentences_content:
            # 根据作文类型推荐重点句型
            result += self._extract_relevant_sentences(sentences_content, essay_type, topic)
        else:
            result += "\n[提示：暂无可用的万能句库]\n"

        # 添加使用建议
        result += "\n" + "=" * 50 + "\n"
        result += "\n💡 万能句使用建议：\n"
        result += "1. 开头句：根据话题选择合适的引入方式（现象/争议/名言）\n"
        result += "2. 过渡句：使用递进、转折、因果等连接词使文章连贯\n"
        result += "3. 论证句：用观点表达、举例、数据等增强说服力\n"
        result += "4. 高级句型：适当使用强调句、倒装句、从句等提升语言水平\n"
        result += "5. 结尾句：总结观点、提出建议或展望未来\n"

        return result

    def _polish_essay(self, exam_type: str, topic: str, user_essay: str) -> str:
        """分析作文并提供词汇升级建议。"""
        if not user_essay:
            return "【作文润色】\n\n错误：请提供需要润色的作文内容（user_essay参数）"

        result = f"【作文润色与词汇升级】\n\n"
        result += f"考试类型：{exam_type.upper()}\n"
        if topic:
            result += f"题目：{topic}\n"
        result += "\n" + "=" * 50 + "\n"
        result += "\n【您的作文】\n\n"
        result += user_essay
        result += "\n\n" + "=" * 50 + "\n"

        # 使用 LLM 进行智能润色
        if self._llm_engine:
            result += self._generate_polish_suggestions(exam_type, topic, user_essay)
        else:
            # 基础润色建议（无 LLM 时）
            result += "\n【基础润色建议】\n\n"
            result += "1. 词汇升级建议：\n"
            result += "   - 使用更正式的学术词汇替代口语化表达\n"
            result += "   - 避免重复使用同一词汇，使用同义词替换\n"
            result += "   - 适当使用高级词汇提升文章档次\n\n"
            result += "2. 句式改进建议：\n"
            result += "   - 长短句结合，避免句式单一\n"
            result += "   - 适当使用复合句、从句增加句式多样性\n"
            result += "   - 尝试使用倒装句、强调句等特殊句式\n\n"
            result += "3. 结构优化建议：\n"
            result += "   - 确保开头、主体、结尾完整\n"
            result += "   - 段落之间使用过渡词连接\n"
            result += "   - 每个段落有明确的主题句\n\n"
            result += "[提示：配置 LLM 引擎可获得更详细的润色建议]\n"

        return result

    def _get_essay_type_name(self, essay_type: str) -> str:
        """获取作文类型的中文名称。"""
        type_names = {
            "argumentation": "议论文",
            "narration": "记叙文",
            "exposition": "说明文",
        }
        return type_names.get(essay_type, essay_type)

    def _get_exam_tips(self, exam_type: str, essay_type: str) -> str:
        """获取针对考试类型的写作建议。"""
        tips = {
            "cet4": {
                "argumentation": """
• 字数要求：120-180词
• 结构：三段式（开头-主体-结尾）
• 重点：观点明确、论据充分、逻辑清晰
• 时间分配：5分钟构思，20分钟写作，5分钟检查
• 注意：避免语法错误，保持卷面整洁""",
                "narration": """
• 字数要求：120-180词
• 结构：时间顺序或事件发展顺序
• 重点：事件完整、细节生动、情感真实
• 常用时态：一般过去时为主
• 注意：交代清楚时间、地点、人物、事件""",
                "exposition": """
• 字数要求：120-180词
• 结构：介绍-说明-总结
• 重点：条理清晰、说明方法得当
• 常用方法：举例、对比、分类、因果
• 注意：客观准确，避免主观臆断""",
            },
            "cet6": {
                "argumentation": """
• 字数要求：150-200词
• 结构：四段式（开头-主体1-主体2-结尾）
• 重点：论证深入、例证丰富、语言高级
• 时间分配：5分钟构思，25分钟写作，5分钟检查
• 注意：使用高级词汇和复杂句式""",
                "narration": """
• 字数要求：150-200词
• 结构：起承转合
• 重点：情节曲折、描写细腻、主题深刻
• 常用技巧：倒叙、插叙、悬念
• 注意：开头吸引人，结尾有余味""",
                "exposition": """
• 字数要求：150-200词
• 结构：总-分-总
• 重点：说明全面、数据准确、逻辑严密
• 常用方法：定义、举例、比较、过程说明
• 注意：专业术语使用准确""",
            },
            "ielts": {
                "argumentation": """
• Task 2 要求：250词以上，40分钟
• 结构：四段式（开头-主体1-主体2-结尾）
• 重点：观点平衡、论证充分、词汇多样
• 评分标准：任务回应、连贯衔接、词汇资源、语法范围
• 注意：回应题目所有部分，避免模板化""",
                "narration": """
• 较少单独考查，可能出现在 Task 2 中
• 重点：故事完整、情感表达、反思深刻
• 注意：即使是记叙也要体现思考深度
• 建议：结合个人经历说明观点""",
                "exposition": """
• Task 1 可能涉及图表说明（数据类）
• 重点：数据描述准确、趋势分析清晰
• 常用表达：上升、下降、波动、稳定等
• 注意：客观描述，不要加入个人观点
• 字数：至少150词，建议20分钟完成""",
            },
        }

        exam_tips = tips.get(exam_type, tips["cet4"])
        return exam_tips.get(essay_type, exam_tips["argumentation"])

    def _extract_relevant_sentences(self, content: str, essay_type: str, topic: str) -> str:
        """根据作文类型和话题提取相关万能句。"""
        result = ""

        # 根据作文类型推荐重点句型
        if essay_type == "argumentation":
            result += "\n【议论文重点句型】\n"
            result += "\n1️⃣ 开头引入句：\n"
            result += "   • In recent years, the issue of... has sparked a heated debate.\n"
            result += "   • When it comes to..., opinions vary from person to person.\n"
            result += "   • It is universally acknowledged that...\n\n"
            result += "2️⃣ 观点表达句：\n"
            result += "   • From my perspective, I firmly believe that...\n"
            result += "   • I am convinced that... plays a vital role in...\n"
            result += "   • There is no denying that...\n\n"
            result += "3️⃣ 论证支持句：\n"
            result += "   • A case in point is...\n"
            result += "   • This is supported by the fact that...\n"
            result += "   • According to recent statistics, ...\n\n"
            result += "4️⃣ 让步转折句：\n"
            result += "   • Admittedly, ... However, ...\n"
            result += "   • While it is true that..., I believe...\n\n"
            result += "5️⃣ 结尾总结句：\n"
            result += "   • Taking all these factors into consideration, ...\n"
            result += "   • Only in this way can we...\n"

        elif essay_type == "narration":
            result += "\n【记叙文重点句型】\n"
            result += "\n1️⃣ 时间顺序句：\n"
            result += "   • It all began when...\n"
            result += "   • The moment I..., I realized...\n"
            result += "   • Years ago, when I was...\n\n"
            result += "2️⃣ 场景描写句：\n"
            result += "   • The scene was so... that...\n"
            result += "   • Surrounded by..., I felt...\n"
            result += "   • It was a... day, with...\n\n"
            result += "3️⃣ 情感表达句：\n"
            result += "   • Never had I felt so...\n"
            result += "   • Words cannot describe how...\n"
            result += "   • It dawned on me that...\n\n"
            result += "4️⃣ 反思感悟句：\n"
            result += "   • Looking back, I now understand...\n"
            result += "   • This experience taught me that...\n"
            result += "   • It was not until then that I...\n"

        elif essay_type == "exposition":
            result += "\n【说明文重点句型】\n"
            result += "\n1️⃣ 定义说明句：\n"
            result += "   • ...can be defined as...\n"
            result += "   • By... we mean...\n"
            result += "   • ...refers to...\n\n"
            result += "2️⃣ 分类说明句：\n"
            result += "   • ...can be classified into... categories.\n"
            result += "   • Generally speaking, there are... types of...\n\n"
            result += "3️⃣ 过程说明句：\n"
            result += "   • The process of... involves... steps.\n"
            result += "   • First of all, ... Then, ... Finally, ...\n\n"
            result += "4️⃣ 因果说明句：\n"
            result += "   • The main reason why... is that...\n"
            result += "   • ...is attributed to...\n"
            result += "   • As a result, ...\n"

        # 添加高级句型推荐
        result += "\n【高分句型推荐】\n"
        result += "\n✨ 强调句：\n"
        result += "   • It is... that...\n"
        result += "   • What matters most is...\n\n"
        result += "✨ 倒装句：\n"
        result += "   • Only by doing so can we...\n"
        result += "   • Not only... but also...\n\n"
        result += "✨ 虚拟语气：\n"
        result += "   • Were it not for..., ...\n"
        result += "   • It is high time that... (过去式)\n\n"
        result += "✨ 从句结构：\n"
        result += "   • There is no doubt that...\n"
        result += "   • The reason why... is that...\n"

        return result

    def _generate_topic_ideas(self, exam_type: str, essay_type: str, topic: str) -> str:
        """使用 LLM 生成针对具体题目的思路建议。"""
        if not self._llm_engine:
            return ""

        from zhixia.llm.base import LLMMessage

        system_prompt = f"""你是一位专业的英语写作指导老师，擅长{exam_type.upper()}考试写作辅导。

请针对给定的作文题目，提供详细的写作思路建议，包括：
1. 审题分析：题目要求和核心议题
2. 立意角度：可选的切入角度和观点
3. 结构框架：推荐的段落安排
4. 论点建议：每个段落可以展开的内容
5. 素材推荐：可用的例子、数据或名言
6. 注意事项：该题目的易错点和提分技巧

请用中文回答，内容实用、具体、可操作。"""

        user_prompt = f"考试类型：{exam_type.upper()}\n作文类型：{self._get_essay_type_name(essay_type)}\n题目：{topic}\n\n请提供详细的写作思路建议。"

        try:
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=2048)
        except Exception as e:
            return f"[生成思路建议时出错：{str(e)}]"

    def _generate_polish_suggestions(self, exam_type: str, topic: str, user_essay: str) -> str:
        """使用 LLM 生成作文润色建议。"""
        if not self._llm_engine:
            return ""

        from zhixia.llm.base import LLMMessage

        system_prompt = f"""你是一位专业的英语作文批改老师，擅长{exam_type.upper()}考试作文评分和润色。

请对学生的作文进行详细分析和润色，包括：

1. 【总体评价】
   - 作文的优点和亮点
   - 存在的主要问题
   - 预估分数（按{exam_type.upper()}标准）

2. 【词汇升级建议】
   - 标出可替换的基础词汇
   - 提供高级替换选项（至少3-5处）
   - 解释为什么新词汇更好

3. 【句式改进建议】
   - 标出句式单一的地方
   - 建议如何增加句式多样性
   - 提供具体改写示例

4. 【语法错误纠正】
   - 列出发现的语法错误
   - 给出正确表达
   - 简要说明语法规则

5. 【结构和逻辑】
   - 评价文章结构
   - 指出逻辑衔接问题
   - 提供改进建议

6. 【修改后的范文】
   - 提供润色后的完整作文
   - 标注主要修改之处

7. 【针对性提升建议】
   - 针对该学生的具体问题给出练习建议

请用中文回答，既指出问题也给予鼓励，让学生有明确的改进方向。"""

        user_prompt = f"考试类型：{exam_type.upper()}\n"
        if topic:
            user_prompt += f"题目：{topic}\n"
        user_prompt += f"\n学生作文：\n{user_essay}\n\n请对这篇作文进行详细批改和润色。"

        try:
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            return self._llm_engine.chat(messages, max_new_tokens=3072)
        except Exception as e:
            return f"\n【润色建议生成失败】\n错误信息：{str(e)}\n\n请检查 LLM 引擎配置。"
