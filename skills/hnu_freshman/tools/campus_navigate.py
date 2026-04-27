"""校园导航工具 — 提供湖南大学校园地点查询和路线指引"""

from zhixia.agent.tool import Tool


class CampusNavigateTool(Tool):
    """查询校园地点位置和导航路线。"""

    def __init__(self):
        super().__init__(
            name="campus_navigate",
            description=(
                "查询湖南大学校园内的地点位置，提供导航路线指引。"
                "输入地点名称（如'岳麓书院'、'复临舍'、'二食堂'），"
                "返回位置描述和步行路线。"
            ),
            func=self._navigate,
        )
        self._locations = self._build_location_db()

    def _build_location_db(self) -> dict:
        return {
            "岳麓书院": {
                "area": "岳麓山麓",
                "description": "千年学府，湖南大学发源地，中国古代四大书院之一",
                "nearby": "爱晚亭、岳麓山、自卑亭",
                "route": "从东方红广场沿登高路步行约5分钟可达",
            },
            "复临舍": {
                "area": "南校区",
                "description": "湖南大学主教学楼之一，多为理工科课程教室",
                "nearby": "图书馆、综合楼",
                "route": "位于麓山南路与牌楼路交汇处，地铁4号线湖南大学站2号口出站步行3分钟",
            },
            "综合楼": {
                "area": "南校区",
                "description": "新建的综合教学楼，设施现代化，配备多媒体教室",
                "nearby": "复临舍、图书馆",
                "route": "位于麓山南路西侧，从地铁湖南大学站出站后向南步行约5分钟",
            },
            "图书馆": {
                "area": "南校区",
                "description": "湖南大学图书馆总馆，藏书丰富，自习座位充足",
                "nearby": "复临舍、综合楼",
                "route": "位于麓山南路，地铁湖南大学站1号口出站即见",
            },
            "一食堂": {
                "area": "南校区",
                "description": "学生第一食堂，提供早中晚餐，价格实惠",
                "nearby": "学生公寓区",
                "route": "位于牌楼路，靠近德智学生公寓",
            },
            "二食堂": {
                "area": "南校区",
                "description": "学生第二食堂（又名五食堂），菜品丰富，有特色美食档口",
                "nearby": "天马学生公寓",
                "route": "位于麓山南路，天马学生公寓旁",
            },
            "德智学生公寓": {
                "area": "南校区",
                "description": "本科生主要宿舍区之一，四人间，上床下桌",
                "nearby": "一食堂、操场",
                "route": "位于牌楼路与麓山南路之间",
            },
            "天马学生公寓": {
                "area": "南校区",
                "description": "大型学生公寓区，设施完善，生活便利",
                "nearby": "二食堂、天马美食街",
                "route": "位于麓山南路西侧，从地铁湖南大学站向南步行约10分钟",
            },
            "东方红广场": {
                "area": "南校区",
                "description": "校园中心广场，毛主席雕像所在地，重要地标",
                "nearby": "岳麓书院、校办公楼",
                "route": "地铁4号线湖南大学站出站即达",
            },
            "校医院": {
                "area": "南校区",
                "description": "湖南大学校医院，提供基本医疗服务",
                "nearby": "图书馆",
                "route": "位于麓山南路，图书馆北侧",
            },
            "体育场": {
                "area": "南校区",
                "description": "主体育场，设有足球场、跑道",
                "nearby": "德智公寓",
                "route": "位于牌楼路东端，靠近德智学生公寓",
            },
            "研究生院": {
                "area": "南校区",
                "description": "研究生教学楼和办公区",
                "nearby": "图书馆",
                "route": "位于麓山南路，图书馆对面",
            },
        }

    def _navigate(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "请告诉我你想去哪里？比如'岳麓书院'、'复临舍'、'二食堂'等。"

        # 模糊匹配
        best_match = None
        best_score = 0
        for name, info in self._locations.items():
            if query in name or name in query:
                return self._format_location(name, info)
            # 简单评分
            score = sum(1 for c in query if c in name)
            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= len(query) // 2:
            return self._format_location(best_match, self._locations[best_match])

        # 列出所有地点
        all_places = "、".join(self._locations.keys())
        return (
            f"抱歉，我没有找到'{query}'的位置。"
            f"我目前知道以下地点：{all_places}。"
            f"你可以告诉我具体地点名称，我会为你指引路线。"
        )

    def _format_location(self, name: str, info: dict) -> str:
        return (
            f"【{name}】\n"
            f"位置：{info['area']}\n"
            f"简介：{info['description']}\n"
            f"周边：{info['nearby']}\n"
            f"路线：{info['route']}"
        )
