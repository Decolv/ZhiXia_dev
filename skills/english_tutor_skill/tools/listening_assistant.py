"""听力辅助器工具 - 提供听力材料获取、播放、理解测试和建议

支持从知识卡获取听力内容，模拟播放，翻译对照测试等功能。
"""

import os
import re
from typing import Optional, Dict, List
from zhixia.agent.tool import Tool


class ListeningAssistantTool(Tool):
    """听力辅助器工具：获取听力材料、模拟播放、理解测试和改进建议。

    功能：
    - get_material: 获取可用的听力材料列表
    - play: 模拟播放听力材料（返回原文）
    - check: 接收用户中文翻译，对照正确翻译给出反馈
    """

    def __init__(self, llm_engine=None, knowledge_provider: Optional["KnowledgeProvider"] = None):
        super().__init__(
            name="listening_assistant",
            description="听力辅助器工具：获取听力材料(get_material)、播放听力(play)、理解测试(check)。参数：action(必需), exam_type, difficulty, material_id, user_translation",
            func=self._execute,
        )
        self._llm_engine = llm_engine
        self._knowledge_provider = knowledge_provider
        self._materials_cache: Dict[str, Dict] = {}

    def _execute(
        self,
        action: str,
        exam_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        material_id: Optional[str] = None,
        user_translation: Optional[str] = None,
    ) -> str:
        """执行听力辅助器功能。

        Args:
            action: 操作类型 - "get_material"/"play"/"check"
            exam_type: 考试类型 (cet4/cet6/ielts)
            difficulty: 难度 (beginner/intermediate/advanced)
            material_id: 材料ID（play/check时使用）
            user_translation: 用户的中文翻译（check时使用）

        Returns:
            根据action返回对应的结果
        """
        if action == "get_material":
            return self._get_materials(exam_type, difficulty)
        elif action == "play":
            return self._play_material(material_id)
        elif action == "check":
            return self._check_translation(material_id, user_translation)
        else:
            return f"错误：未知的action '{action}'。支持的action：get_material, play, check"

    def _get_materials(
        self, exam_type: Optional[str], difficulty: Optional[str]
    ) -> str:
        """获取可用的听力材料列表。"""
        # 无知识卡时的降级处理
        if self._knowledge_provider is None:
            return "【提示】知识卡未挂载，无法获取听力材料"

        # 通过知识提供者获取材料
        materials = self._knowledge_provider.get_listening_materials(exam_type, difficulty)

        if not materials:
            filter_info = []
            if exam_type:
                filter_info.append(f"考试类型: {exam_type}")
            if difficulty:
                filter_info.append(f"难度: {difficulty}")
            filters = "，".join(filter_info) if filter_info else "无"
            return f"未找到符合条件的听力材料（筛选条件：{filters}）"

        # 缓存材料内容
        for material in materials:
            self._materials_cache[material["id"]] = material

        # 格式化输出
        lines = ["📚 可用听力材料列表", "=" * 40]

        for m in materials:
            lines.append(f"\n📝 {m['title']}")
            lines.append(f"   ID: {m['id']}")
            lines.append(f"   考试类型: {m['exam_type'].upper()}")
            lines.append(f"   难度: {self._translate_difficulty(m['difficulty'])}")
            lines.append(f"   话题: {m.get('topic', '未标注')}")
            lines.append(f"   类型: {m.get('type', '未标注')}")

        lines.append(f"\n共找到 {len(materials)} 个材料")
        lines.append("\n使用说明：")
        lines.append("- 播放听力：action='play', material_id='材料ID'")
        lines.append("- 翻译测试：action='check', material_id='材料ID', user_translation='你的中文翻译'")

        return "\n".join(lines)

    def _play_material(self, material_id: Optional[str]) -> str:
        """模拟播放听力材料（返回原文）。"""
        if not material_id:
            return "错误：请提供 material_id 参数"

        # 如果缓存中没有，尝试加载
        if material_id not in self._materials_cache:
            self._load_material_by_id(material_id)

        material = self._materials_cache.get(material_id)
        if not material:
            return f"错误：未找到材料 '{material_id}'。请先使用 action='get_material' 查看可用材料。"

        lines = [
            f"🎧 正在播放：{material['title']}",
            "=" * 50,
            f"[考试类型: {material['exam_type'].upper()} | 难度: {self._translate_difficulty(material['difficulty'])}]",
            "",
            "📖 听力原文：",
            "-" * 40,
            material.get("original", "[原文内容缺失]"),
            "",
            "-" * 40,
            "✅ 播放完成！",
            "",
            "💡 提示：你可以尝试翻译这段内容，然后使用 action='check' 来测试你的理解程度。",
        ]

        return "\n".join(lines)

    def _check_translation(
        self, material_id: Optional[str], user_translation: Optional[str]
    ) -> str:
        """检查用户的翻译并给出反馈。"""
        if not material_id:
            return "错误：请提供 material_id 参数"
        if not user_translation:
            return "错误：请提供 user_translation 参数（你的中文翻译）"

        # 如果缓存中没有，尝试加载
        if material_id not in self._materials_cache:
            self._load_material_by_id(material_id)

        material = self._materials_cache.get(material_id)
        if not material:
            return f"错误：未找到材料 '{material_id}'。请先使用 action='get_material' 查看可用材料。"

        correct_translation = material.get("translation", "")
        original_text = material.get("original", "")

        # 使用 LLM 进行智能对比分析
        if self._llm_engine:
            return self._llm_check_translation(
                original_text, correct_translation, user_translation, material
            )

        # 无 LLM 时的基础对比
        return self._basic_check_translation(
            original_text, correct_translation, user_translation, material
        )

    def _llm_check_translation(
        self,
        original: str,
        correct: str,
        user: str,
        material: Dict,
    ) -> str:
        """使用 LLM 进行智能翻译评估。"""
        from zhixia.llm.base import LLMMessage

        system_prompt = """你是一位专业的英语听力理解评估专家。请对比用户的翻译与正确翻译，提供详细的反馈。

评估维度：
1. 理解准确度：核心信息是否理解正确
2. 细节完整性：关键细节是否遗漏或误解
3. 表达流畅度：中文表达是否自然通顺

反馈格式：
- 总体评价（优秀/良好/需改进）
- 理解准确度分析
- 遗漏或误解的关键点
- 具体改进建议
- 鼓励性结语

请用中文回答，语气友好、鼓励性但专业。"""

        user_prompt = f"""听力材料：{material['title']}
难度：{self._translate_difficulty(material['difficulty'])}

【听力原文】
{original}

【正确翻译】
{correct}

【用户翻译】
{user}

请评估用户的听力理解程度并给出反馈和建议。"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        feedback = self._llm_engine.chat(messages, max_new_tokens=2048)

        lines = [
            f"📝 翻译评估报告 - {material['title']}",
            "=" * 50,
            "",
            "【你的翻译】",
            user,
            "",
            "【评估结果】",
            "-" * 40,
            feedback,
        ]

        return "\n".join(lines)

    def _basic_check_translation(
        self,
        original: str,
        correct: str,
        user: str,
        material: Dict,
    ) -> str:
        """基础翻译对比（无 LLM 时）。"""
        # 简单的关键词匹配来评估
        user_keywords = set(re.findall(r"\w+", user.lower()))
        correct_keywords = set(re.findall(r"\w+", correct.lower()))

        # 计算简单相似度
        if len(correct_keywords) > 0:
            overlap = len(user_keywords & correct_keywords)
            similarity = overlap / len(correct_keywords)
        else:
            similarity = 0

        # 根据相似度给出评价
        if similarity >= 0.7:
            evaluation = "优秀"
            suggestion = "你的理解非常准确！继续保持，可以尝试更高难度的材料。"
        elif similarity >= 0.4:
            evaluation = "良好"
            suggestion = "你理解了大部分内容，但还有一些细节需要加强。建议多听几遍，注意关键词。"
        else:
            evaluation = "需改进"
            suggestion = "理解还有较大提升空间。建议先听较简单的材料，或者分段听写练习。"

        lines = [
            f"📝 翻译评估报告 - {material['title']}",
            "=" * 50,
            "",
            "【你的翻译】",
            user,
            "",
            "【正确翻译】",
            correct,
            "",
            "【评估结果】",
            "-" * 40,
            f"总体评价：{evaluation}",
            f"关键词匹配度：{similarity*100:.1f}%",
            "",
            "【改进建议】",
            suggestion,
            "",
            "💡 提示：系统配置 LLM 后可获得更智能的详细分析。",
        ]

        return "\n".join(lines)

    def _parse_material_file(
        self, file_path: str, exam_type: str, filename: str
    ) -> Dict:
        """解析听力材料文件。"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取标题
        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else filename.replace(".md", "")

        # 提取原文
        original_match = re.search(
            r"## 原文\s*\n+(.+?)(?=\n## 翻译|$)", content, re.DOTALL
        )
        original = original_match.group(1).strip() if original_match else ""

        # 提取翻译
        translation_match = re.search(
            r"## 翻译\s*\n+(.+?)(?=\n## 元数据|$)", content, re.DOTALL
        )
        translation = translation_match.group(1).strip() if translation_match else ""

        # 提取元数据
        metadata = {}
        metadata_match = re.search(r"## 元数据\s*\n(.+)$", content, re.DOTALL)
        if metadata_match:
            meta_text = metadata_match.group(1)
            for line in meta_text.split("\n"):
                if line.strip().startswith("-"):
                    parts = line.strip()[1:].strip().split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metadata[key] = value

        # 生成材料ID
        material_id = f"{exam_type}_{filename.replace('.md', '')}"

        return {
            "id": material_id,
            "title": title,
            "exam_type": exam_type,
            "filename": filename,
            "original": original,
            "translation": translation,
            "difficulty": metadata.get("难度", "unknown"),
            "topic": metadata.get("话题", ""),
            "type": metadata.get("类型", ""),
        }

    def _load_material_by_id(self, material_id: str) -> bool:
        """根据ID加载材料到缓存。"""
        # 解析ID格式: exam_type_filename
        parts = material_id.split("_", 1)
        if len(parts) != 2:
            return False

        exam_type, filename_base = parts
        file_path = os.path.join(
            self.KNOWLEDGE_BASE_PATH, exam_type, f"{filename_base}.md"
        )

        if os.path.exists(file_path):
            material_info = self._parse_material_file(file_path, exam_type, f"{filename_base}.md")
            self._materials_cache[material_id] = material_info
            return True

        return False

    def _translate_difficulty(self, difficulty: str) -> str:
        """将难度翻译成中文。"""
        mapping = {
            "beginner": "初级",
            "intermediate": "中级",
            "advanced": "高级",
            "unknown": "未知",
        }
        return mapping.get(difficulty, difficulty)
