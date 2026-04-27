"""专业查询工具 — 提供湖南大学各专业培养方案摘要"""

from zhixia.agent.tool import Tool


class MajorQueryTool(Tool):
    """查询湖南大学本科专业培养方案、课程设置和毕业要求。"""

    def __init__(self):
        super().__init__(
            name="query_major",
            description=(
                "查询湖南大学本科专业的培养方案摘要。"
                "输入专业名称（如'计算机科学与技术'、'土木工程'、'工商管理'），"
                "返回该专业的培养目标、核心课程、学制学位等信息。"
            ),
            func=self._query,
        )
        self._majors = self._build_major_db()

    def _build_major_db(self) -> dict:
        return {
            "计算机科学与技术": {
                "college": "信息科学与工程学院",
                "degree": "工学学士",
                "duration": "四年",
                "goal": "培养具备扎实计算机理论基础、系统掌握计算机软硬件知识、具有创新实践能力的高素质人才",
                "core_courses": "数据结构、算法设计、计算机组成原理、操作系统、计算机网络、数据库系统、软件工程、人工智能导论",
                "features": "注重编程能力培养，设有ACM竞赛基地，与华为、腾讯等企业深度合作",
                "career": "软件开发、算法工程师、系统架构师、人工智能研究员、继续深造",
            },
            "软件工程": {
                "college": "信息科学与工程学院",
                "degree": "工学学士",
                "duration": "四年",
                "goal": "培养掌握软件工程理论与方法、具备大型软件系统分析设计与开发能力的高级工程人才",
                "core_courses": "软件工程、面向对象程序设计、软件测试、项目管理、移动应用开发、云计算技术",
                "features": "强调工程实践能力，采用项目驱动教学，毕业设计多为企业真实项目",
                "career": "软件工程师、项目经理、技术总监、创业",
            },
            "土木工程": {
                "college": "土木工程学院",
                "degree": "工学学士",
                "duration": "四年",
                "goal": "培养具备土木工程领域勘察、设计、施工、管理能力的高级工程技术人才",
                "core_courses": "结构力学、材料力学、土力学、混凝土结构、钢结构、工程测量、建筑施工",
                "features": "百年土木，实力雄厚，拥有国家重点实验室，参与港珠澳大桥等国家级工程",
                "career": "结构工程师、施工管理、设计院、房地产企业、继续深造",
            },
            "机械设计制造及其自动化": {
                "college": "机械与运载工程学院",
                "degree": "工学学士",
                "duration": "四年",
                "goal": "培养掌握机械设计、制造及自动化技术、具备工程创新能力的高级技术人才",
                "core_courses": "机械原理、机械设计、控制工程基础、数控技术、机器人技术、CAD/CAM",
                "features": "拥有汽车车身先进设计制造国家重点实验室，与三一重工、中联重科紧密合作",
                "career": "机械工程师、自动化工程师、汽车行业、智能制造",
            },
            "工商管理": {
                "college": "工商管理学院",
                "degree": "管理学学士",
                "duration": "四年",
                "goal": "培养具备现代管理理论、掌握企业管理方法、具有国际视野的高级管理人才",
                "core_courses": "管理学原理、市场营销、财务管理、战略管理、人力资源管理、运营管理",
                "features": "案例教学为主，强调实践能力，设有创业孵化基地",
                "career": "企业管理、咨询公司、金融机构、创业、继续深造",
            },
            "法学": {
                "college": "法学院",
                "degree": "法学学士",
                "duration": "四年",
                "goal": "培养系统掌握法学知识、熟悉我国法律和政策、能在国家机关和企事业单位从事法律工作的人才",
                "core_courses": "宪法学、民法学、刑法学、行政法与行政诉讼法、商法、经济法、国际法",
                "features": "注重模拟法庭训练，与多家律所和法院建立实习基地",
                "career": "律师、法官、检察官、企业法务、公务员",
            },
            "新闻学": {
                "college": "新闻与传播学院",
                "degree": "文学学士",
                "duration": "四年",
                "goal": "培养具备新闻传播理论知识、掌握全媒体技能、具有社会责任感的传媒人才",
                "core_courses": "新闻学概论、传播学概论、新闻采访与写作、新闻编辑、新媒体运营、数据新闻",
                "features": "拥有融媒体实验中心，与湖南广电、人民日报湖南分社深度合作",
                "career": "记者、编辑、新媒体运营、公关传播、继续深造",
            },
            "金融学": {
                "college": "金融与统计学院",
                "degree": "经济学学士",
                "duration": "四年",
                "goal": "培养具备金融理论基础和实务操作能力、能在金融机构和企事业单位从事金融工作的人才",
                "core_courses": "货币银行学、国际金融、证券投资学、金融风险管理、金融工程、计量经济学",
                "features": " quantitative 特色鲜明，与长沙银行、方正证券等建立合作",
                "career": "银行、证券、保险、基金公司、企业财务、继续深造",
            },
            "电气工程及其自动化": {
                "college": "电气与信息工程学院",
                "degree": "工学学士",
                "duration": "四年",
                "goal": "培养掌握电气工程领域基础理论和专业技能、具备工程实践能力的高级技术人才",
                "core_courses": "电路原理、电机学、电力电子技术、电力系统分析、自动控制原理、高电压技术",
                "features": "国家重点学科，与国家电网、南方电网等企业紧密合作",
                "career": "电力系统、电气设计、新能源、智能制造",
            },
            "建筑学": {
                "college": "建筑学院",
                "degree": "建筑学学士",
                "duration": "五年",
                "goal": "培养具备建筑设计、城市设计、室内设计能力的高级建筑设计人才",
                "core_courses": "建筑设计基础、建筑构造、建筑历史、城市规划原理、建筑物理",
                "features": "五年制，注重设计能力培养，设有建筑模型实验室和数字化设计中心",
                "career": "建筑师、城市规划师、室内设计师、房地产",
            },
        }

    def _query(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "请告诉我你想了解哪个专业？比如'计算机科学与技术'、'土木工程'等。"

        # 精确匹配
        for name, info in self._majors.items():
            if query in name or name in query:
                return self._format_major(name, info)

        # 模糊匹配
        for name, info in self._majors.items():
            if any(keyword in query for keyword in name.split("及")[0].split("与")[0].split("及其")[0].split("学")[:1]):
                return self._format_major(name, info)

        all_majors = "、".join(self._majors.keys())
        return (
            f"抱歉，我没有找到'{query}'的专业信息。"
            f"我目前知道以下专业：{all_majors}。"
            f"你可以告诉我具体专业名称，我会为你介绍培养方案。"
        )

    def _format_major(self, name: str, info: dict) -> str:
        return (
            f"【{name}】\n"
            f"所属学院：{info['college']}\n"
            f"授予学位：{info['degree']}\n"
            f"学制：{info['duration']}\n"
            f"培养目标：{info['goal']}\n"
            f"核心课程：{info['core_courses']}\n"
            f"专业特色：{info['features']}\n"
            f"就业方向：{info['career']}"
        )
