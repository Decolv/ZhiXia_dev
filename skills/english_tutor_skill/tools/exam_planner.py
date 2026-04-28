"""考试准备计划器工具 - 创建个性化备考计划

功能：
1. 创建备考计划：接收考试类型、截止日期、当前水平，生成个性化计划
2. 薄弱点分析：分析单词、长难句、作文、听力四个维度
3. 排期规划：根据剩余天数分配学习任务
4. 计划存储：将计划保存到JSON文件
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from zhixia.agent.tool import Tool


class ExamPlannerTool(Tool):
    """考试准备计划器工具：创建和管理个性化英语考试备考计划。

    支持考试类型：CET4/CET6/IELTS/TOEFL
    支持操作：create/view/update/analyze
    """

    # 考试配置信息
    EXAM_CONFIG = {
        "cet4": {
            "name": "大学英语四级",
            "total_score": 710,
            "pass_score": 425,
            "sections": ["writing", "listening", "reading", "translation"],
            "duration_days": 60,
        },
        "cet6": {
            "name": "大学英语六级",
            "total_score": 710,
            "pass_score": 425,
            "sections": ["writing", "listening", "reading", "translation"],
            "duration_days": 90,
        },
        "ielts": {
            "name": "雅思",
            "total_score": 9.0,
            "pass_score": 6.0,
            "sections": ["listening", "reading", "writing", "speaking"],
            "duration_days": 90,
        },
        "toefl": {
            "name": "托福",
            "total_score": 120,
            "pass_score": 80,
            "sections": ["reading", "listening", "speaking", "writing"],
            "duration_days": 90,
        },
    }

    # 水平配置
    LEVEL_CONFIG = {
        "beginner": {"daily_hours": 3, "intensity": "高强度", "focus": "基础巩固"},
        "intermediate": {"daily_hours": 2, "intensity": "中等强度", "focus": "能力提升"},
        "advanced": {"daily_hours": 1.5, "intensity": "维持强度", "focus": "冲刺突破"},
    }

    # 薄弱点对应的学习任务
    WEAK_POINT_TASKS = {
        "vocabulary": {
            "name": "词汇",
            "daily_tasks": [
                "背诵50-100个核心词汇",
                "复习昨日生词",
                "完成词汇练习题20道",
            ],
            "tips": [
                "使用艾宾浩斯遗忘曲线复习",
                "结合真题语境记忆",
                "重点掌握高频词汇",
            ],
        },
        "listening": {
            "name": "听力",
            "daily_tasks": [
                "精听1篇真题听力",
                "泛听30分钟英语材料",
                "听写练习15分钟",
            ],
            "tips": [
                "第一遍抓大意，第二遍抓细节",
                "注意连读和弱读现象",
                "积累听力高频词汇",
            ],
        },
        "sentence": {
            "name": "长难句",
            "daily_tasks": [
                "分析5个长难句结构",
                "翻译练习3个复杂句子",
                "语法知识点复习",
            ],
            "tips": [
                "先找主干，再分析修饰成分",
                "注意从句嵌套结构",
                "多做拆分和重组练习",
            ],
        },
        "writing": {
            "name": "写作",
            "daily_tasks": [
                "背诵1篇范文",
                "仿写1个段落",
                "积累5个高级句型",
            ],
            "tips": [
                "建立个人写作模板",
                "多使用连接词和过渡句",
                "注意段落结构和逻辑",
            ],
        },
    }

    def __init__(self, llm_engine=None, storage_dir: str = None):
        super().__init__(
            name="exam_planner",
            description="""考试准备计划器工具：创建和管理个性化英语考试备考计划。
参数：
- action: 操作类型 (create/view/update/analyze)
- exam_type: 考试类型 (cet4/cet6/ielts/toefl)
- exam_date: 考试日期 (YYYY-MM-DD格式)
- current_level: 当前水平 (beginner/intermediate/advanced)
- target_score: 目标分数
- weak_points: 薄弱点列表 ["vocabulary", "listening", "sentence", "writing"]""",
            func=self._execute,
        )
        self._llm_engine = llm_engine
        # 默认存储路径
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self._storage_dir = storage_dir
        os.makedirs(self._storage_dir, exist_ok=True)
        self._plan_file = os.path.join(self._storage_dir, "exam_plan.json")

    def _execute(
        self,
        action: str = "create",
        exam_type: str = None,
        exam_date: str = None,
        current_level: str = "intermediate",
        target_score: float = None,
        weak_points: List[str] = None,
    ) -> str:
        """执行工具逻辑。

        Args:
            action: 操作类型 (create/view/update/analyze)
            exam_type: 考试类型
            exam_date: 考试日期
            current_level: 当前水平
            target_score: 目标分数
            weak_points: 薄弱点列表

        Returns:
            工具执行结果文本
        """
        action = action.lower()

        if action == "create":
            return self._create_plan(exam_type, exam_date, current_level, target_score, weak_points)
        elif action == "view":
            return self._view_plan()
        elif action == "update":
            return self._update_plan(exam_type, exam_date, current_level, target_score, weak_points)
        elif action == "analyze":
            return self._analyze_weak_points(exam_type, current_level, weak_points)
        else:
            return f"❌ 不支持的操作类型: {action}。支持的操作: create/view/update/analyze"

    def _create_plan(
        self,
        exam_type: str,
        exam_date: str,
        current_level: str,
        target_score: float,
        weak_points: List[str],
    ) -> str:
        """创建新的备考计划。"""
        # 验证参数
        if not exam_type or exam_type.lower() not in self.EXAM_CONFIG:
            return f"❌ 请提供有效的考试类型: {list(self.EXAM_CONFIG.keys())}"

        if not exam_date:
            return "❌ 请提供考试日期 (YYYY-MM-DD格式)"

        try:
            exam_date_obj = datetime.strptime(exam_date, "%Y-%m-%d").date()
        except ValueError:
            return "❌ 考试日期格式错误，请使用 YYYY-MM-DD 格式"

        today = datetime.now().date()
        if exam_date_obj <= today:
            return "❌ 考试日期必须在未来"

        exam_type = exam_type.lower()
        current_level = current_level.lower() if current_level else "intermediate"

        if current_level not in self.LEVEL_CONFIG:
            return f"❌ 无效的当前水平: {current_level}。可选: {list(self.LEVEL_CONFIG.keys())}"

        # 计算剩余天数
        days_remaining = (exam_date_obj - today).days

        # 设置默认目标分数
        if target_score is None:
            target_score = self.EXAM_CONFIG[exam_type]["pass_score"]

        # 设置默认薄弱点
        if weak_points is None:
            weak_points = []

        # 创建计划数据
        plan = {
            "exam_type": exam_type,
            "exam_name": self.EXAM_CONFIG[exam_type]["name"],
            "exam_date": exam_date,
            "created_date": today.isoformat(),
            "current_level": current_level,
            "target_score": target_score,
            "weak_points": weak_points,
            "days_remaining": days_remaining,
            "daily_schedule": self._generate_daily_schedule(exam_type, current_level, weak_points, days_remaining),
            "progress": {"completed_days": 0, "total_days": days_remaining},
        }

        # 保存计划
        self._save_plan(plan)

        # 生成计划文本
        return self._format_plan_text(plan)

    def _view_plan(self) -> str:
        """查看当前备考计划。"""
        plan = self._load_plan()
        if not plan:
            return "📋 暂无备考计划。请使用 action='create' 创建新计划。"

        # 更新剩余天数
        exam_date = datetime.strptime(plan["exam_date"], "%Y-%m-%d").date()
        today = datetime.now().date()
        plan["days_remaining"] = max(0, (exam_date - today).days)

        return self._format_plan_text(plan)

    def _update_plan(
        self,
        exam_type: str = None,
        exam_date: str = None,
        current_level: str = None,
        target_score: float = None,
        weak_points: List[str] = None,
    ) -> str:
        """更新现有备考计划。"""
        plan = self._load_plan()
        if not plan:
            return "❌ 暂无备考计划。请先使用 action='create' 创建计划。"

        # 更新字段
        if exam_type and exam_type.lower() in self.EXAM_CONFIG:
            plan["exam_type"] = exam_type.lower()
            plan["exam_name"] = self.EXAM_CONFIG[exam_type.lower()]["name"]

        if exam_date:
            try:
                exam_date_obj = datetime.strptime(exam_date, "%Y-%m-%d").date()
                if exam_date_obj > datetime.now().date():
                    plan["exam_date"] = exam_date
                    plan["days_remaining"] = (exam_date_obj - datetime.now().date()).days
            except ValueError:
                return "❌ 考试日期格式错误，请使用 YYYY-MM-DD 格式"

        if current_level and current_level.lower() in self.LEVEL_CONFIG:
            plan["current_level"] = current_level.lower()

        if target_score is not None:
            plan["target_score"] = target_score

        if weak_points is not None:
            plan["weak_points"] = weak_points

        # 重新生成日程
        plan["daily_schedule"] = self._generate_daily_schedule(
            plan["exam_type"],
            plan["current_level"],
            plan["weak_points"],
            plan["days_remaining"],
        )

        # 保存更新
        self._save_plan(plan)

        return f"✅ 计划已更新！\n\n{self._format_plan_text(plan)}"

    def _analyze_weak_points(
        self,
        exam_type: str,
        current_level: str,
        weak_points: List[str],
    ) -> str:
        """分析薄弱点并提供建议。"""
        if not exam_type or exam_type.lower() not in self.EXAM_CONFIG:
            return f"❌ 请提供有效的考试类型: {list(self.EXAM_CONFIG.keys())}"

        exam_type = exam_type.lower()
        current_level = current_level.lower() if current_level else "intermediate"

        if not weak_points:
            weak_points = ["vocabulary", "listening", "sentence", "writing"]

        exam_name = self.EXAM_CONFIG[exam_type]["name"]
        level_info = self.LEVEL_CONFIG.get(current_level, self.LEVEL_CONFIG["intermediate"])

        result = f"""📊 【{exam_name}】薄弱点分析报告

👤 当前水平: {current_level} ({level_info['focus']})
📚 分析维度: {len(weak_points)} 项

"""

        for wp in weak_points:
            if wp in self.WEAK_POINT_TASKS:
                info = self.WEAK_POINT_TASKS[wp]
                result += f"\n🔍 【{info['name']}】\n"
                result += f"\n📋 每日任务:\n"
                for i, task in enumerate(info["daily_tasks"], 1):
                    result += f"   {i}. {task}\n"
                result += f"\n💡 学习建议:\n"
                for tip in info["tips"]:
                    result += f"   • {tip}\n"
                result += "\n" + "-" * 40 + "\n"

        # 使用 LLM 生成更详细的建议
        if self._llm_engine:
            from zhixia.llm.base import LLMMessage
            prompt = f"""请为{exam_name}考生提供薄弱点改进建议：
当前水平: {current_level}
薄弱点: {[self.WEAK_POINT_TASKS.get(wp, {}).get('name', wp) for wp in weak_points]}

请给出具体、可操作的学习建议，包括：
1. 各薄弱点的学习优先级
2. 推荐的学习资源和方法
3. 时间分配建议"""

            messages = [
                LLMMessage(role="system", content="你是专业的英语考试辅导专家，擅长制定个性化学习计划。"),
                LLMMessage(role="user", content=prompt),
            ]
            llm_advice = self._llm_engine.chat(messages, max_new_tokens=512)
            result += f"\n🤖 AI 专家建议:\n{llm_advice}\n"

        return result

    def _generate_daily_schedule(
        self,
        exam_type: str,
        current_level: str,
        weak_points: List[str],
        days_remaining: int,
    ) -> List[Dict[str, Any]]:
        """生成每日学习日程。"""
        schedule = []
        level_info = self.LEVEL_CONFIG.get(current_level, self.LEVEL_CONFIG["intermediate"])
        daily_hours = level_info["daily_hours"]

        # 基础任务（所有考生都需要）
        base_tasks = [
            "完成1套真题阅读",
            "复习错题笔记",
        ]

        # 根据薄弱点添加针对性任务
        weak_point_tasks = []
        for wp in weak_points:
            if wp in self.WEAK_POINT_TASKS:
                weak_point_tasks.extend(self.WEAK_POINT_TASKS[wp]["daily_tasks"])

        # 分阶段规划
        if days_remaining <= 30:
            # 冲刺阶段
            phase = "冲刺阶段"
            phase_tasks = ["全真模拟考试", "重点复习错题", "背诵高频词汇"]
        elif days_remaining <= 60:
            # 强化阶段
            phase = "强化阶段"
            phase_tasks = ["专项突破练习", "真题精练", "弱项强化"]
        else:
            # 基础阶段
            phase = "基础阶段"
            phase_tasks = ["夯实基础", "系统学习", "积累词汇"]

        # 生成每日计划
        for day in range(1, min(days_remaining + 1, 30)):  # 最多生成30天详细计划
            day_plan = {
                "day": day,
                "phase": phase,
                "daily_hours": daily_hours,
                "tasks": [],
            }

            # 添加基础任务
            day_plan["tasks"].extend(base_tasks[:1])  # 每天1个基础任务

            # 添加薄弱点任务（轮询）
            if weak_point_tasks:
                task_index = (day - 1) % len(weak_point_tasks)
                day_plan["tasks"].append(weak_point_tasks[task_index])

            # 添加阶段任务（每3天一次）
            if day % 3 == 0 and phase_tasks:
                day_plan["tasks"].append(phase_tasks[(day // 3 - 1) % len(phase_tasks)])

            schedule.append(day_plan)

        return schedule

    def _format_plan_text(self, plan: Dict[str, Any]) -> str:
        """格式化计划为可读文本。"""
        exam_name = plan["exam_name"]
        exam_date = plan["exam_date"]
        days_remaining = plan["days_remaining"]
        current_level = plan["current_level"]
        target_score = plan["target_score"]
        weak_points = plan["weak_points"]

        level_info = self.LEVEL_CONFIG.get(current_level, self.LEVEL_CONFIG["intermediate"])

        result = f"""📚 【{exam_name}】备考计划

📅 考试日期: {exam_date}
⏰ 剩余天数: {days_remaining} 天
👤 当前水平: {current_level} ({level_info['focus']})
🎯 目标分数: {target_score}
📊 学习强度: {level_info['intensity']} (每天约 {level_info['daily_hours']} 小时)

🔴 薄弱点重点攻克:
"""

        if weak_points:
            for wp in weak_points:
                name = self.WEAK_POINT_TASKS.get(wp, {}).get("name", wp)
                result += f"   • {name}\n"
        else:
            result += "   • 暂无指定薄弱点，将进行全面复习\n"

        result += "\n📋 近期学习安排:\n"

        # 显示前7天计划
        schedule = plan.get("daily_schedule", [])
        for day_plan in schedule[:7]:
            day = day_plan["day"]
            phase = day_plan["phase"]
            tasks = day_plan["tasks"]
            result += f"\n第 {day} 天 ({phase}):\n"
            for task in tasks:
                result += f"   ✓ {task}\n"

        if len(schedule) > 7:
            result += f"\n... 还有 {len(schedule) - 7} 天的详细计划 ...\n"

        result += "\n💡 使用提示:\n"
        result += "   • 每天按计划完成学习任务\n"
        result += "   • 定期回顾薄弱点，调整学习重点\n"
        result += "   • 使用 action='analyze' 获取薄弱点详细分析\n"

        return result

    def _save_plan(self, plan: Dict[str, Any]) -> None:
        """保存计划到JSON文件。"""
        try:
            with open(self._plan_file, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存计划失败: {e}")

    def _load_plan(self) -> Optional[Dict[str, Any]]:
        """从JSON文件加载计划。"""
        if not os.path.exists(self._plan_file):
            return None
        try:
            with open(self._plan_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载计划失败: {e}")
            return None
