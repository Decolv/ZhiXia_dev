"""专业查询工具 — 使用LLM智能生成专业培养方案

该工具不再使用硬编码专业数据库，而是通过LLM动态生成专业介绍。
专业知识作为上下文注入LLM，使其能够基于知识生成答案。
"""

from typing import TYPE_CHECKING, Optional

from zhixia.agent.tool import Tool

if TYPE_CHECKING:
    from zhixia.agent.callbacks import CallbackManager


class MajorQueryTool(Tool):
    """查询湖南大学本科专业培养方案，使用LLM智能生成答案。"""

    # 专业知识库 - 作为LLM上下文的参考信息，而非硬编码答案
    MAJORS_KNOWLEDGE = """
【计算机科学与技术】
所属学院：信息科学与工程学院
授予学位：工学学士
学制：四年
培养目标：培养具备扎实计算机理论基础、系统掌握计算机软硬件知识、具有创新实践能力的高素质人才
核心课程：数据结构、算法设计、计算机组成原理、操作系统、计算机网络、数据库系统、软件工程、人工智能导论
专业特色：注重编程能力培养，设有ACM竞赛基地，与华为、腾讯等企业深度合作
就业方向：软件开发、算法工程师、系统架构师、人工智能研究员、继续深造

【软件工程】
所属学院：信息科学与工程学院
授予学位：工学学士
学制：四年
培养目标：培养掌握软件工程理论与方法、具备大型软件系统分析设计与开发能力的高级工程人才
核心课程：软件工程、面向对象程序设计、软件测试、项目管理、移动应用开发、云计算技术
专业特色：强调工程实践能力，采用项目驱动教学，毕业设计多为企业真实项目
就业方向：软件工程师、项目经理、技术总监、创业

【土木工程】
所属学院：土木工程学院
授予学位：工学学士
学制：四年
培养目标：培养具备土木工程领域勘察、设计、施工、管理能力的高级工程技术人才
核心课程：结构力学、材料力学、土力学、混凝土结构、钢结构、工程测量、建筑施工
专业特色：百年土木，实力雄厚，拥有国家重点实验室，参与港珠澳大桥等国家级工程
就业方向：结构工程师、施工管理、设计院、房地产企业、继续深造

【机械设计制造及其自动化】
所属学院：机械与运载工程学院
授予学位：工学学士
学制：四年
培养目标：培养掌握机械设计、制造及自动化技术、具备工程创新能力的高级技术人才
核心课程：机械原理、机械设计、控制工程基础、数控技术、机器人技术、CAD/CAM
专业特色：拥有汽车车身先进设计制造国家重点实验室，与三一重工、中联重科紧密合作
就业方向：机械工程师、自动化工程师、汽车行业、智能制造

【工商管理】
所属学院：工商管理学院
授予学位：管理学学士
学制：四年
培养目标：培养具备现代管理理论、掌握企业管理方法、具有国际视野的高级管理人才
核心课程：管理学原理、市场营销、财务管理、战略管理、人力资源管理、运营管理
专业特色：案例教学为主，强调实践能力，设有创业孵化基地
就业方向：企业管理、咨询公司、金融机构、创业、继续深造

【法学】
所属学院：法学院
授予学位：法学学士
学制：四年
培养目标：培养系统掌握法学知识、熟悉我国法律和政策、能在国家机关和企事业单位从事法律工作的人才
核心课程：宪法学、民法学、刑法学、行政法与行政诉讼法、商法、经济法、国际法
专业特色：注重模拟法庭训练，与多家律所和法院建立实习基地
就业方向：律师、法官、检察官、企业法务、公务员

【新闻学】
所属学院：新闻与传播学院
授予学位：文学学士
学制：四年
培养目标：培养具备新闻传播理论知识、掌握全媒体技能、具有社会责任感的传媒人才
核心课程：新闻学概论、传播学概论、新闻采访与写作、新闻编辑、新媒体运营、数据新闻
专业特色：拥有融媒体实验中心，与湖南广电、人民日报湖南分社深度合作
就业方向：记者、编辑、新媒体运营、公关传播、继续深造

【金融学】
所属学院：金融与统计学院
授予学位：经济学学士
学制：四年
培养目标：培养具备金融理论基础和实务操作能力、能在金融机构和企事业单位从事金融工作的人才
核心课程：货币银行学、国际金融、证券投资学、金融风险管理、金融工程、计量经济学
专业特色：quantitative特色鲜明，与长沙银行、方正证券等建立合作
就业方向：银行、证券、保险、基金公司、企业财务、继续深造

【电气工程及其自动化】
所属学院：电气与信息工程学院
授予学位：工学学士
学制：四年
培养目标：培养掌握电气工程领域基础理论和专业技能、具备工程实践能力的高级技术人才
核心课程：电路原理、电机学、电力电子技术、电力系统分析、自动控制原理、高电压技术
专业特色：国家重点学科，与国家电网、南方电网等企业紧密合作
就业方向：电力系统、电气设计、新能源、智能制造

【建筑学】
所属学院：建筑学院
授予学位：建筑学学士
学制：五年
培养目标：培养具备建筑设计、城市设计、室内设计能力的高级建筑设计人才
核心课程：建筑设计基础、建筑构造、建筑历史、城市规划原理、建筑物理
专业特色：五年制，注重设计能力培养，设有建筑模型实验室和数字化设计中心
就业方向：建筑师、城市规划师、室内设计师、房地产
"""

    def __init__(self, llm_engine=None, callbacks: Optional["CallbackManager"] = None):
        super().__init__(
            name="query_major",
            description=(
                "查询湖南大学本科专业的培养方案摘要。"
                "输入专业名称（如'计算机科学与技术'、'土木工程'、'工商管理'），"
                "返回智能生成的专业介绍、培养目标、核心课程等信息。"
            ),
            func=self._query,
        )
        self._llm_engine = llm_engine
        self._callbacks = callbacks

    def set_llm_engine(self, llm_engine) -> None:
        """设置LLM引擎，用于动态生成答案。"""
        self._llm_engine = llm_engine

    def set_callbacks(self, callbacks: "CallbackManager") -> None:
        """设置回调管理器，用于播报思考过程。"""
        self._callbacks = callbacks

    def _query(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "请告诉我你想了解哪个专业？比如'计算机科学与技术'、'土木工程'等。"

        # 触发思考开始回调
        if self._callbacks:
            self._callbacks.on_thinking_start("query_major")

        # 如果没有LLM引擎，返回知识库摘要
        if self._llm_engine is None:
            result = self._generate_fallback_response(query)
            if self._callbacks:
                self._callbacks.on_thinking_end("query_major")
            return result

        # 使用LLM生成答案
        try:
            result = self._generate_with_llm(query)
        except Exception as e:
            # LLM生成失败时回退到知识库匹配
            result = self._generate_fallback_response(query)

        # 触发思考结束回调
        if self._callbacks:
            self._callbacks.on_thinking_end("query_major")

        return result

    def _generate_with_llm(self, query: str) -> str:
        """使用LLM动态生成专业介绍。"""
        from zhixia.llm.base import LLMMessage

        # 构建系统提示词
        system_prompt = f"""你是湖南大学专业咨询助手。请基于以下专业知识，为用户生成个性化的专业介绍。

【专业知识参考】
{self.MAJORS_KNOWLEDGE}

【回答要求】
1. 基于上述知识，针对用户询问的专业生成个性化介绍
2. 回答要自然、友好，像学长学姐给新生介绍专业一样
3. 如果知识库中没有该专业信息，请诚实告知，并推荐相关专业
4. 可以适当添加专业选择建议
5. 回答控制在250字以内，简洁明了但信息完整
"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"请介绍一下{query}专业"),
        ]

        # 触发思考过程回调
        if self._callbacks:
            self._callbacks.on_agent_thought("query_major", f"正在查询'{query}'专业信息...")

        # 调用LLM生成答案
        response = self._llm_engine.chat(messages, max_new_tokens=300)

        # 播报生成完成
        if self._callbacks:
            self._callbacks.on_agent_thought("query_major", "专业介绍生成完成")

        return response.strip()

    def _generate_fallback_response(self, query: str) -> str:
        """当LLM不可用时，基于关键词匹配生成简单回答。"""
        query_lower = query.lower()

        # 简单的关键词匹配，用于确定相关专业
        major_keywords = {
            "计算机": ["计算机", "软件", "编程", "代码", "算法"],
            "土木工程": ["土木", "建筑", "施工", "结构"],
            "机械": ["机械", "自动化", "制造", "机器人"],
            "工商管理": ["工商", "管理", "MBA", "企业"],
            "法学": ["法学", "法律", "律师", "法官"],
            "新闻": ["新闻", "传媒", "记者", "编辑"],
            "金融": ["金融", "经济", "投资", "银行", "证券"],
            "电气": ["电气", "电力", "电网", "电机"],
            "建筑": ["建筑", "设计", "规划"],
        }

        # 确定相关专业
        matched_majors = []
        for major, keywords in major_keywords.items():
            if any(kw in query_lower for kw in keywords):
                matched_majors.append(major)

        if matched_majors:
            return (
                f"我了解到你想了解关于{', '.join(matched_majors)}相关的专业信息。"
                "由于智能生成服务暂时不可用，我建议你：\n"
                "1. 访问湖南大学教务处官网查看专业介绍\n"
                "2. 咨询相关学院的招生办公室\n"
                "3. 联系在读学长学姐了解真实情况"
            )

        return (
            "我理解你想了解专业相关信息。"
            "我可以帮你介绍计算机科学与技术、软件工程、土木工程、机械设计制造及其自动化、"
            "工商管理、法学、新闻学、金融学、电气工程及其自动化、建筑学等热门专业。"
            "请告诉我具体专业名称，我会为你详细介绍。"
        )
