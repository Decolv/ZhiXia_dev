"""词汇复习器工具 - 基于艾宾浩斯遗忘曲线的词汇记忆系统

功能：
1. 制定计划：根据考试类型制定词汇记忆计划
2. 滚动复习：按记忆曲线安排复习
3. 定期检测：测试用户词汇掌握情况
4. 进度汇总：统计学习进度并给出建议

依赖：
- 通过 KnowledgeProvider 接口获取词汇内容，实现与知识卡的解耦
"""

import json
import os
import random
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Any
from zhixia.agent.tool import Tool

if TYPE_CHECKING:
    from zhixia.core.card_base import KnowledgeProvider, VocabularyItem


class VocabularyReviewerTool(Tool):
    """词汇复习器工具：提供基于艾宾浩斯遗忘曲线的词汇记忆系统。

    支持功能：
    - create_plan: 根据考试类型制定词汇记忆计划
    - review: 按记忆曲线返回今日复习单词列表
    - test: 生成测试题目或评分
    - progress: 统计学习进度并给出建议

    依赖：
    - 通过 KnowledgeProvider 接口获取词汇内容，实现与知识卡的解耦
    - 支持动态切换不同考试类型的词汇库
    """

    # 艾宾浩斯遗忘曲线复习间隔（天数）
    REVIEW_INTERVALS = [1, 2, 4, 7, 15, 30]

    # 支持的考试类型
    SUPPORTED_EXAMS = ["cet4", "cet6", "ielts"]

    def __init__(
        self,
        llm_engine=None,
        project_root: str = None,
        knowledge_provider: Optional["KnowledgeProvider"] = None
    ):
        super().__init__(
            name="vocabulary_reviewer",
            description="词汇复习器工具：制定词汇记忆计划、滚动复习、定期检测、进度汇总。参数：action(create_plan/review/test/progress)、exam_type(cet4/cet6/ielts)、daily_count(每日学习量)、word_id(单词ID)、test_results(测试结果)",
            func=self._execute,
        )
        self._llm_engine = llm_engine
        self._project_root = project_root or self._find_project_root()
        self._knowledge_provider = knowledge_provider
        self._user_data_dir = os.path.join(self._project_root, "data", "vocabulary_reviewer")
        os.makedirs(self._user_data_dir, exist_ok=True)

    def set_knowledge_provider(self, knowledge_provider: Optional["KnowledgeProvider"]) -> None:
        """动态设置或切换知识提供者。

        Args:
            knowledge_provider: 知识提供者实例，用于获取词汇内容
        """
        self._knowledge_provider = knowledge_provider

    def _find_project_root(self) -> str:
        """查找项目根目录"""
        current = os.getcwd()
        while current and current != os.path.dirname(current):
            if os.path.exists(os.path.join(current, "skills")):
                return current
            current = os.path.dirname(current)
        return os.getcwd()

    def _execute(
        self,
        action: str,
        exam_type: str = "",
        daily_count: int = 20,
        word_id: str = "",
        test_results: str = "",
    ) -> str:
        """执行词汇复习器功能。

        Args:
            action: 操作类型 - create_plan/review/test/progress
            exam_type: 考试类型 - cet4/cet6/ielts
            daily_count: 每日学习量（默认20）
            word_id: 单词ID（review时使用）
            test_results: 测试结果JSON字符串（test时使用）

        Returns:
            功能执行结果文本
        """
        action = action.lower()

        if action == "create_plan":
            return self._create_plan(exam_type, daily_count)
        elif action == "review":
            return self._review(exam_type, word_id)
        elif action == "test":
            return self._test(exam_type, test_results)
        elif action == "progress":
            return self._progress(exam_type)
        else:
            return f"【错误】未知的操作类型：{action}\n\n支持的操作：create_plan, review, test, progress"

    def _load_vocabulary(self, exam_type: str) -> List[Dict[str, str]]:
        """加载词汇库。

        优先通过 KnowledgeProvider 接口获取词汇内容，
        如果未提供 knowledge_provider，则返回空列表。

        Args:
            exam_type: 考试类型 (cet4/cet6/ielts)

        Returns:
            词汇列表，每个词汇为字典格式
        """
        if not self._knowledge_provider:
            return []

        try:
            vocab_items: List["VocabularyItem"] = self._knowledge_provider.get_vocabulary(
                exam_type=exam_type
            )

            # 将 VocabularyItem 转换为字典格式
            words = []
            for item in vocab_items:
                words.append({
                    "id": item.id,
                    "word": item.word,
                    "phonetic": item.phonetic,
                    "pos": item.pos,
                    "meaning": item.meaning,
                    "example": item.example,
                    "translation": item.translation,
                    "memory": item.memory_tip,
                })
            return words
        except Exception as e:
            print(f"通过 KnowledgeProvider 加载词汇库失败: {e}")
            return []

    def _get_user_data_path(self, exam_type: str) -> str:
        """获取用户数据文件路径"""
        return os.path.join(self._user_data_dir, f"{exam_type}_plan.json")

    def _load_user_data(self, exam_type: str) -> Dict:
        """加载用户学习数据"""
        data_path = self._get_user_data_path(exam_type)
        if os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "exam_type": exam_type,
            "created_at": datetime.now().isoformat(),
            "daily_count": 20,
            "total_words": 0,
            "learned_words": [],
            "review_schedule": {},
            "test_history": [],
            "word_status": {},  # word_id -> {"status": "new/learning/mastered", "next_review": date, "review_count": int}
        }

    def _save_user_data(self, exam_type: str, data: Dict):
        """保存用户学习数据"""
        data_path = self._get_user_data_path(exam_type)
        try:
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户数据失败: {e}")

    def _create_plan(self, exam_type: str, daily_count: int) -> str:
        """制定词汇记忆计划

        Args:
            exam_type: 考试类型
            daily_count: 每日学习量

        Returns:
            学习计划文本
        """
        if not exam_type:
            return "【错误】请指定考试类型（cet4/cet6/ielts）"

        exam_type = exam_type.lower()
        if exam_type not in self.SUPPORTED_EXAMS:
            return f"【错误】不支持的考试类型：{exam_type}\n\n支持的类型：{', '.join(self.SUPPORTED_EXAMS)}"

        # 检查知识提供者是否可用
        if not self._knowledge_provider:
            return (
                f"【错误】词汇知识卡未加载\n\n"
                f"词汇复习器需要英语考试知识卡提供词汇内容。\n"
                f"请先加载 english_tutor_knowledge 知识卡。"
            )

        # 加载词汇库
        vocabulary = self._load_vocabulary(exam_type)
        if not vocabulary:
            return (
                f"【错误】无法加载 {exam_type} 词汇库\n\n"
                f"可能原因：\n"
                f"1. 知识卡中未包含 {exam_type} 的词汇内容\n"
                f"2. KnowledgeProvider 接口返回空数据"
            )

        total_words = len(vocabulary)
        days_needed = (total_words + daily_count - 1) // daily_count

        # 创建或更新学习计划
        user_data = self._load_user_data(exam_type)
        user_data["daily_count"] = daily_count
        user_data["total_words"] = total_words
        user_data["plan_created_at"] = datetime.now().isoformat()
        user_data["estimated_completion"] = (datetime.now() + timedelta(days=days_needed)).isoformat()

        # 为每个单词分配学习日期和复习计划
        for i, word in enumerate(vocabulary):
            word_id = word["id"]
            if word_id not in user_data["word_status"]:
                study_day = i // daily_count + 1
                study_date = (datetime.now() + timedelta(days=study_day - 1)).strftime("%Y-%m-%d")

                # 计算复习日期
                review_dates = []
                for interval in self.REVIEW_INTERVALS:
                    review_date = (datetime.now() + timedelta(days=study_day - 1 + interval)).strftime("%Y-%m-%d")
                    review_dates.append(review_date)

                user_data["word_status"][word_id] = {
                    "status": "new",
                    "study_day": study_day,
                    "study_date": study_date,
                    "review_dates": review_dates,
                    "next_review": review_dates[0] if review_dates else None,
                    "review_count": 0,
                    "correct_count": 0,
                    "test_count": 0,
                }

        self._save_user_data(exam_type, user_data)

        # 生成计划报告
        plan_report = f"""📚 【词汇记忆计划】

考试类型：{exam_type.upper()}
词汇总量：{total_words} 词
每日学习：{daily_count} 词
预计完成：{days_needed} 天
计划开始：{datetime.now().strftime("%Y-%m-%d")}
预计结束：{(datetime.now() + timedelta(days=days_needed)).strftime("%Y-%m-%d")}

📅 学习安排：
- 第 1 阶段（第1-7天）：新词学习 + 第1、2、4天复习
- 第 2 阶段（第8-15天）：新词学习 + 第7天复习
- 第 3 阶段（第16-30天）：新词学习 + 第15天复习
- 第 4 阶段（第30天+）：第30天最终复习

🔄 复习机制（艾宾浩斯遗忘曲线）：
- 第1次复习：学习后1天
- 第2次复习：学习后2天
- 第3次复习：学习后4天
- 第4次复习：学习后7天
- 第5次复习：学习后15天
- 第6次复习：学习后30天

✅ 计划已创建！使用 action="review" 开始今日复习。"""

        return plan_report

    def _review(self, exam_type: str, word_id: str = "") -> str:
        """返回今日复习单词列表

        Args:
            exam_type: 考试类型
            word_id: 如果提供，标记该单词为已复习

        Returns:
            今日复习列表文本
        """
        if not exam_type:
            return "【错误】请指定考试类型（cet4/cet6/ielts）"

        exam_type = exam_type.lower()
        user_data = self._load_user_data(exam_type)

        if not user_data.get("word_status"):
            return f"【提示】尚未创建 {exam_type} 的学习计划。请先使用 action='create_plan' 创建计划。"

        today = datetime.now().strftime("%Y-%m-%d")

        # 如果提供了 word_id，标记为已复习
        if word_id and word_id in user_data["word_status"]:
            word_status = user_data["word_status"][word_id]
            word_status["review_count"] = word_status.get("review_count", 0) + 1
            word_status["status"] = "learning"

            # 更新下次复习日期
            review_idx = word_status["review_count"]
            if review_idx < len(self.REVIEW_INTERVALS):
                next_interval = self.REVIEW_INTERVALS[review_idx]
                word_status["next_review"] = (datetime.now() + timedelta(days=next_interval)).strftime("%Y-%m-%d")
            else:
                word_status["status"] = "mastered"
                word_status["next_review"] = None

            self._save_user_data(exam_type, user_data)
            return f"✅ 单词 [{word_id}] 已标记为已复习（第 {word_status['review_count']} 次）"

        # 获取今日需要学习的单词（新词）
        new_words = []
        for word_id, status in user_data["word_status"].items():
            if status["status"] == "new" and status["study_date"] == today:
                new_words.append(word_id)

        # 获取今日需要复习的单词
        review_words = []
        for word_id, status in user_data["word_status"].items():
            if status["status"] == "learning" and status.get("next_review") == today:
                review_words.append(word_id)

        # 加载词汇详情
        vocabulary = self._load_vocabulary(exam_type)
        vocab_dict = {w["id"]: w for w in vocabulary}

        # 生成复习列表
        result_lines = [f"📖 【今日词汇任务】{today}", ""]

        # 新词学习
        daily_count = user_data.get("daily_count", 20)
        new_words = new_words[:daily_count]
        if new_words:
            result_lines.append(f"🆕 今日新词（{len(new_words)} 个）：")
            for i, wid in enumerate(new_words, 1):
                word = vocab_dict.get(wid, {})
                if word:
                    result_lines.append(f"\n{i}. {word['word']} {word['phonetic']}")
                    result_lines.append(f"   词性：{word['pos']}")
                    result_lines.append(f"   释义：{word['meaning']}")
                    result_lines.append(f"   例句：{word['example']}")
                    result_lines.append(f"   翻译：{word['translation']}")
                    result_lines.append(f"   💡 记忆法：{word['memory']}")
        else:
            result_lines.append("🆕 今日新词：无（已完成当日新词学习）")

        result_lines.append("")

        # 复习单词
        if review_words:
            result_lines.append(f"🔄 今日复习（{len(review_words)} 个）：")
            for i, wid in enumerate(review_words, 1):
                word = vocab_dict.get(wid, {})
                status = user_data["word_status"].get(wid, {})
                if word:
                    review_count = status.get("review_count", 0)
                    result_lines.append(f"\n{i}. {word['word']} {word['phonetic']}（第 {review_count + 1} 次复习）")
                    result_lines.append(f"   释义：{word['meaning']}")
                    result_lines.append(f"   例句：{word['example']}")
        else:
            result_lines.append("🔄 今日复习：无（没有需要复习的单词）")

        # 统计信息
        total_learned = sum(1 for s in user_data["word_status"].values() if s["status"] != "new")
        total_mastered = sum(1 for s in user_data["word_status"].values() if s["status"] == "mastered")

        result_lines.extend([
            "",
            "📊 学习统计：",
            f"   已学习：{total_learned} / {user_data['total_words']} 词",
            f"   已掌握：{total_mastered} 词",
            f"   掌握率：{total_mastered / user_data['total_words'] * 100:.1f}%" if user_data['total_words'] > 0 else "   掌握率：0%",
        ])

        return "\n".join(result_lines)

    def _test(self, exam_type: str, test_results: str = "") -> str:
        """生成测试题目或评分

        Args:
            exam_type: 考试类型
            test_results: 测试结果JSON字符串（如果有则评分，否则生成题目）

        Returns:
            测试题目或评分结果
        """
        if not exam_type:
            return "【错误】请指定考试类型（cet4/cet6/ielts）"

        exam_type = exam_type.lower()
        user_data = self._load_user_data(exam_type)

        if not user_data.get("word_status"):
            return f"【提示】尚未创建 {exam_type} 的学习计划。请先使用 action='create_plan' 创建计划。"

        # 如果有测试结果，进行评分
        if test_results:
            try:
                results = json.loads(test_results)
                return self._grade_test(exam_type, results, user_data)
            except json.JSONDecodeError:
                return "【错误】测试结果格式不正确，请提供有效的JSON字符串"

        # 否则生成测试题目
        vocabulary = self._load_vocabulary(exam_type)
        vocab_dict = {w["id"]: w for w in vocabulary}

        # 选择已学习的单词进行测试（最多10个）
        learned_words = [
            wid for wid, status in user_data["word_status"].items()
            if status["status"] != "new"
        ]

        if len(learned_words) < 5:
            return "【提示】已学习的单词不足，请先进行更多学习后再测试。"

        test_words = random.sample(learned_words, min(10, len(learned_words)))

        # 生成测试题目
        questions = []
        for i, word_id in enumerate(test_words, 1):
            word = vocab_dict.get(word_id, {})
            if not word:
                continue

            # 随机选择题型
            question_type = random.choice(["meaning", "example", "fill_blank"])

            if question_type == "meaning":
                # 选择释义
                # 生成干扰项
                distractors = random.sample(
                    [w["meaning"] for w in vocabulary if w["id"] != word_id],
                    min(3, len(vocabulary) - 1)
                )
                options = [word["meaning"]] + distractors
                random.shuffle(options)

                questions.append({
                    "id": i,
                    "word_id": word_id,
                    "word": word["word"],
                    "type": "meaning",
                    "question": f"{word['word']} {word['phonetic']} 的词义是？",
                    "options": options,
                    "answer": word["meaning"],
                })

            elif question_type == "example":
                # 理解例句
                questions.append({
                    "id": i,
                    "word_id": word_id,
                    "word": word["word"],
                    "type": "example",
                    "question": f"请根据例句理解 '{word['word']}' 的含义：\n   {word['example']}",
                    "hint": f"词性：{word['pos']}",
                    "answer": word["meaning"],
                })

            else:  # fill_blank
                # 填空题
                example = word["example"]
                # 将单词替换为下划线
                blank_example = example.replace(word["word"], "______")
                questions.append({
                    "id": i,
                    "word_id": word_id,
                    "word": word["word"],
                    "type": "fill_blank",
                    "question": f"请填入正确的单词：\n   {blank_example}",
                    "hint": f"词性：{word['pos']}，释义：{word['meaning']}",
                    "answer": word["word"],
                })

        # 格式化输出
        result_lines = ["📝 【词汇检测】", ""]
        result_lines.append(f"考试类型：{exam_type.upper()}")
        result_lines.append(f"题目数量：{len(questions)} 题")
        result_lines.append("=" * 50)

        for q in questions:
            result_lines.append(f"\n【第 {q['id']} 题】{q['type']}")
            result_lines.append(f"题目：{q['question']}")

            if "options" in q:
                result_lines.append("选项：")
                for j, opt in enumerate(q["options"], 1):
                    result_lines.append(f"  {j}. {opt}")

            if "hint" in q:
                result_lines.append(f"提示：{q['hint']}")

            result_lines.append(f"答案：{q['answer']}")

        result_lines.extend([
            "",
            "=" * 50,
            "📋 测试说明：",
            "1. 请记录你的答案",
            "2. 完成后使用 action='test' 并传入 test_results 参数提交答案",
            "3. test_results 格式：JSON字符串，包含 question_id 和 user_answer",
        ])

        # 保存当前测试到用户数据
        user_data["current_test"] = questions
        self._save_user_data(exam_type, user_data)

        return "\n".join(result_lines)

    def _grade_test(self, exam_type: str, results: Dict, user_data: Dict) -> str:
        """评分测试

        Args:
            exam_type: 考试类型
            results: 用户答案
            user_data: 用户数据

        Returns:
            评分结果文本
        """
        current_test = user_data.get("current_test", [])
        if not current_test:
            return "【错误】没有找到当前测试，请先生成测试题目"

        user_answers = results.get("answers", [])
        if not user_answers:
            return "【错误】请提供答案列表"

        correct_count = 0
        wrong_answers = []

        for user_ans in user_answers:
            q_id = user_ans.get("question_id")
            user_answer = user_ans.get("answer", "").strip().lower()

            # 找到对应题目
            question = next((q for q in current_test if q["id"] == q_id), None)
            if question:
                correct_answer = question["answer"].strip().lower()
                word_id = question["word_id"]

                # 更新单词测试记录
                if word_id not in user_data["word_status"]:
                    user_data["word_status"][word_id] = {}

                word_status = user_data["word_status"][word_id]
                word_status["test_count"] = word_status.get("test_count", 0) + 1

                if user_answer == correct_answer:
                    correct_count += 1
                    word_status["correct_count"] = word_status.get("correct_count", 0) + 1
                else:
                    wrong_answers.append({
                        "word": question["word"],
                        "user_answer": user_ans.get("answer"),
                        "correct_answer": question["answer"],
                    })

        # 保存测试历史
        total_questions = len(current_test)
        score = correct_count / total_questions * 100 if total_questions > 0 else 0

        test_record = {
            "date": datetime.now().isoformat(),
            "score": score,
            "correct": correct_count,
            "total": total_questions,
        }
        user_data["test_history"].append(test_record)
        user_data["current_test"] = []
        self._save_user_data(exam_type, user_data)

        # 生成评分报告
        result_lines = ["📊 【测试结果】", ""]
        result_lines.append(f"得分：{score:.1f} 分")
        result_lines.append(f"正确：{correct_count} / {total_questions} 题")

        if score >= 90:
            result_lines.append("\n🌟 优秀！词汇掌握非常好！")
        elif score >= 70:
            result_lines.append("\n👍 良好！继续保持！")
        elif score >= 60:
            result_lines.append("\n💪 及格！还需要加强复习哦！")
        else:
            result_lines.append("\n📚 需要努力！建议多复习后再测试！")

        if wrong_answers:
            result_lines.append("\n❌ 错题回顾：")
            for wa in wrong_answers:
                result_lines.append(f"   - {wa['word']}: 你的答案「{wa['user_answer']}」→ 正确答案「{wa['correct_answer']}」")

        # 学习建议
        result_lines.append("\n💡 学习建议：")
        if score < 60:
            result_lines.append("   - 建议降低每日学习量，确保每个单词都充分掌握")
            result_lines.append("   - 增加复习频率，重点关注错题单词")
        elif score < 80:
            result_lines.append("   - 继续保持当前学习节奏")
            result_lines.append("   - 针对错题单词进行专项复习")
        else:
            result_lines.append("   - 可以适当增加每日学习量")
            result_lines.append("   - 尝试学习更高级别的词汇")

        return "\n".join(result_lines)

    def _progress(self, exam_type: str) -> str:
        """统计学习进度并给出建议

        Args:
            exam_type: 考试类型

        Returns:
            进度统计和建议文本
        """
        if not exam_type:
            return "【错误】请指定考试类型（cet4/cet6/ielts）"

        exam_type = exam_type.lower()
        user_data = self._load_user_data(exam_type)

        if not user_data.get("word_status"):
            return f"【提示】尚未创建 {exam_type} 的学习计划。请先使用 action='create_plan' 创建计划。"

        total_words = user_data.get("total_words", 0)
        word_status = user_data.get("word_status", {})

        # 统计各类状态单词数量
        new_words = sum(1 for s in word_status.values() if s["status"] == "new")
        learning_words = sum(1 for s in word_status.values() if s["status"] == "learning")
        mastered_words = sum(1 for s in word_status.values() if s["status"] == "mastered")
        learned_words = learning_words + mastered_words

        # 计算学习天数
        plan_created = user_data.get("plan_created_at")
        if plan_created:
            start_date = datetime.fromisoformat(plan_created)
            days_elapsed = (datetime.now() - start_date).days + 1
        else:
            days_elapsed = 0

        # 计算平均每日学习量
        avg_daily = learned_words / days_elapsed if days_elapsed > 0 else 0

        # 预计完成时间
        daily_count = user_data.get("daily_count", 20)
        remaining_words = total_words - learned_words
        estimated_days = (remaining_words + daily_count - 1) // daily_count if daily_count > 0 else 0

        # 测试历史统计
        test_history = user_data.get("test_history", [])
        if test_history:
            avg_score = sum(t["score"] for t in test_history) / len(test_history)
            latest_score = test_history[-1]["score"]
        else:
            avg_score = 0
            latest_score = 0

        # 生成进度报告
        result_lines = [f"📈 【学习进度报告】{exam_type.upper()}", ""]

        # 总体进度
        progress_pct = learned_words / total_words * 100 if total_words > 0 else 0
        mastered_pct = mastered_words / total_words * 100 if total_words > 0 else 0

        result_lines.append("📊 总体进度：")
        result_lines.append(f"   总词汇量：{total_words} 词")
        result_lines.append(f"   已学习：{learned_words} 词 ({progress_pct:.1f}%)")
        result_lines.append(f"   学习中：{learning_words} 词")
        result_lines.append(f"   已掌握：{mastered_words} 词 ({mastered_pct:.1f}%)")
        result_lines.append(f"   未学习：{new_words} 词")

        # 学习统计
        result_lines.append("\n📅 学习统计：")
        result_lines.append(f"   学习天数：{days_elapsed} 天")
        result_lines.append(f"   日均学习：{avg_daily:.1f} 词")
        result_lines.append(f"   预计完成：还需约 {estimated_days} 天")

        # 测试统计
        result_lines.append("\n📝 测试统计：")
        result_lines.append(f"   测试次数：{len(test_history)} 次")
        if test_history:
            result_lines.append(f"   平均得分：{avg_score:.1f} 分")
            result_lines.append(f"   最近得分：{latest_score:.1f} 分")

        # 进度条可视化
        bar_length = 20
        filled = int(progress_pct / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        result_lines.append(f"\n📊 进度条：[{bar}] {progress_pct:.1f}%")

        # 学习建议
        result_lines.append("\n💡 学习建议：")

        if progress_pct < 10:
            result_lines.append("   - 🚀 学习刚刚开始，保持耐心，每天坚持！")
        elif progress_pct < 50:
            result_lines.append("   - 📚 学习进行顺利，继续保持当前节奏！")
        elif progress_pct < 80:
            result_lines.append("   - 💪 已完成大部分学习，冲刺阶段加油！")
        else:
            result_lines.append("   - 🎯 即将完成全部学习，准备总复习！")

        if avg_score > 0:
            if avg_score < 60:
                result_lines.append("   - ⚠️ 测试平均分较低，建议加强复习频率")
                result_lines.append("   - 🔄 重点关注掌握不牢固的单词")
            elif avg_score < 80:
                result_lines.append("   - 👍 测试成绩良好，继续保持！")
            else:
                result_lines.append("   - 🌟 测试成绩优秀，可以考虑挑战更高难度！")

        if avg_daily < daily_count * 0.5:
            result_lines.append("   - ⏰ 近期学习量偏低，建议增加学习时间")
        elif avg_daily > daily_count * 1.5:
            result_lines.append("   - 🔥 学习热情高涨！注意劳逸结合")

        # 下一步行动建议
        result_lines.append("\n📋 下一步行动：")
        result_lines.append("   1. 使用 action='review' 进行今日复习")
        if len(test_history) == 0 or (datetime.now() - datetime.fromisoformat(test_history[-1]["date"])).days >= 7:
            result_lines.append("   2. 使用 action='test' 进行定期检测")
        result_lines.append(f"   3. 每天坚持学习 {daily_count} 个新词")

        return "\n".join(result_lines)
