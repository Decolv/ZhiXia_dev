"""校园导航工具 — 使用LLM智能生成导航指引

该工具不再使用硬编码地点数据库，而是通过LLM动态生成导航信息。
校园地点知识作为上下文注入LLM，使其能够基于知识生成答案。
支持地图图片标注，可展示相关目的地图片和路线。
图片素材路径预留，待用户提供素材后填充。
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from zhixia.agent.tool import Tool

if TYPE_CHECKING:
    from zhixia.agent.callbacks import CallbackManager


class CampusNavigateTool(Tool):
    """查询校园地点位置和导航路线，使用LLM智能生成答案，支持地图图片展示。"""

    # 校园地点知识库 - 作为LLM上下文的参考信息，而非硬编码答案
    LOCATIONS_KNOWLEDGE = """
【岳麓书院】
区域：岳麓山麓
简介：千年学府，湖南大学发源地，中国古代四大书院之一
周边：爱晚亭、岳麓山、自卑亭
路线：从东方红广场沿登高路步行约5分钟可达
图片资源：map_image=assets/maps/yuelu_academy_map.png

【复临舍】
区域：南校区
简介：湖南大学主教学楼之一，多为理工科课程教室
周边：图书馆、综合楼
路线：位于麓山南路与牌楼路交汇处，地铁4号线湖南大学站2号口出站步行3分钟
图片资源：map_image=assets/maps/fulinshe_map.png

【综合楼】
区域：南校区
简介：新建的综合教学楼，设施现代化，配备多媒体教室
周边：复临舍、图书馆
路线：位于麓山南路西侧，从地铁湖南大学站出站后向南步行约5分钟
图片资源：map_image=assets/maps/zonghe_building_map.png

【图书馆】
区域：南校区
简介：湖南大学图书馆总馆，藏书丰富，自习座位充足
周边：复临舍、综合楼
路线：位于麓山南路，地铁湖南大学站1号口出站即见
图片资源：map_image=assets/maps/library_map.png

【一食堂】
区域：南校区
简介：学生第一食堂，提供早中晚餐，价格实惠
周边：学生公寓区
路线：位于牌楼路，靠近德智学生公寓
图片资源：map_image=assets/maps/canteen1_map.png

【二食堂】
区域：南校区
简介：学生第二食堂（又名五食堂），菜品丰富，有特色美食档口
周边：天马学生公寓
路线：位于麓山南路，天马学生公寓旁
图片资源：map_image=assets/maps/canteen2_map.png

【德智学生公寓】
区域：南校区
简介：本科生主要宿舍区之一，四人间，上床下桌
周边：一食堂、操场
路线：位于牌楼路与麓山南路之间
图片资源：map_image=assets/maps/dezhi_dorm_map.png

【天马学生公寓】
区域：南校区
简介：大型学生公寓区，设施完善，生活便利
周边：二食堂、天马美食街
路线：位于麓山南路西侧，从地铁湖南大学站向南步行约10分钟
图片资源：map_image=assets/maps/tianma_dorm_map.png

【东方红广场】
区域：南校区
简介：校园中心广场，毛主席雕像所在地，重要地标
周边：岳麓书院、校办公楼
路线：地铁4号线湖南大学站出站即达
图片资源：map_image=assets/maps/east_red_square_map.png

【校医院】
区域：南校区
简介：湖南大学校医院，提供基本医疗服务
周边：图书馆
路线：位于麓山南路，图书馆北侧
图片资源：map_image=assets/maps/hospital_map.png

【体育场】
区域：南校区
简介：主体育场，设有足球场、跑道
周边：德智公寓
路线：位于牌楼路东端，靠近德智学生公寓
图片资源：map_image=assets/maps/stadium_map.png

【研究生院】
区域：南校区
简介：研究生教学楼和办公区
周边：图书馆
路线：位于麓山南路，图书馆对面
图片资源：map_image=assets/maps/grad_school_map.png
"""

    # 图片资源根目录 - 可配置
    ASSETS_ROOT = Path("skills/hnu_freshman/assets")

    def __init__(self, llm_engine=None, callbacks: Optional["CallbackManager"] = None):
        super().__init__(
            name="campus_navigate",
            description=(
                "查询湖南大学校园内的地点位置，提供导航路线指引。"
                "输入地点名称（如'岳麓书院'、'复临舍'、'二食堂'），"
                "返回智能生成的位置描述、步行路线和相关地图图片。"
            ),
            func=self._navigate,
        )
        self._llm_engine = llm_engine
        self._callbacks = callbacks

    def set_llm_engine(self, llm_engine) -> None:
        """设置LLM引擎，用于动态生成答案。"""
        self._llm_engine = llm_engine

    def set_callbacks(self, callbacks: "CallbackManager") -> None:
        """设置回调管理器，用于播报导航过程。"""
        self._callbacks = callbacks

    def _navigate(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "请告诉我你想去哪里？比如'岳麓书院'、'复临舍'、'二食堂'等。"

        # 触发思考开始回调
        if self._callbacks:
            self._callbacks.on_thinking_start("campus_navigate")
            self._callbacks.on_agent_thought("campus_navigate", f"正在查找'{query}'的位置...")

        # 如果没有LLM引擎，返回知识库摘要
        if self._llm_engine is None:
            result = self._generate_fallback_response(query)
            if self._callbacks:
                self._callbacks.on_thinking_end("campus_navigate")
            return result

        # 使用LLM生成答案
        try:
            result = self._generate_with_llm(query)
        except Exception as e:
            # LLM生成失败时回退到知识库匹配
            result = self._generate_fallback_response(query)

        # 触发思考结束回调
        if self._callbacks:
            self._callbacks.on_thinking_end("campus_navigate")

        return result

    def _generate_with_llm(self, query: str) -> str:
        """使用LLM动态生成导航指引。"""
        from zhixia.llm.base import LLMMessage

        # 查找匹配地点
        matched_location = self._find_matching_location(query)

        # 构建系统提示词
        system_prompt = f"""你是湖南大学校园导航助手。请基于以下校园地点知识，为用户提供个性化的导航指引。

【校园地点知识参考】
{self.LOCATIONS_KNOWLEDGE}

【回答要求】
1. 基于上述知识，针对用户询问的地点生成个性化导航指引
2. 回答要自然、友好，像学长学姐给新生指路一样
3. 如果知识库中没有该地点信息，请诚实告知，并推荐相关地点
4. 可以适当添加实用的出行小贴士（如最佳出行时间、注意事项等）
5. 回答控制在200字以内，简洁明了但信息完整
"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"我想去{query}，请告诉我怎么走"),
        ]

        # 触发思考过程回调
        if self._callbacks:
            self._callbacks.on_agent_thought("campus_navigate", f"正在生成'{query}'的导航路线...")

        # 调用LLM生成答案
        response = self._llm_engine.chat(messages, max_new_tokens=256)

        # 播报生成完成
        if self._callbacks:
            self._callbacks.on_agent_thought("campus_navigate", "导航指引生成完成")

        # 如果有匹配的地点，通过回调传递导航数据
        if matched_location and self._callbacks:
            nav_data = self._build_nav_data(matched_location)
            if hasattr(self._callbacks, 'on_nav_data_ready'):
                self._callbacks.on_nav_data_ready("campus_navigate", nav_data)
            # 存储在返回文本的元数据中
            return f"__NAV_DATA__{matched_location}__\n\n{response.strip()}"

        return response.strip()

    def _find_matching_location(self, query: str) -> Optional[str]:
        """查找匹配地点名称。"""
        query_lower = query.lower()
        import re
        location_names = re.findall(r'【(.*?)】', self.LOCATIONS_KNOWLEDGE)

        for name in location_names:
            if name in query or query in name:
                return name

        # 模糊匹配
        for name in location_names:
            if name.lower() in query_lower or query_lower in name.lower():
                return name

        # 关键词匹配
        location_keywords = {
            "岳麓书院": ["岳麓书院", "书院", "爱晚亭", "岳麓山"],
            "复临舍": ["复临舍", "教学楼", "教室"],
            "综合楼": ["综合楼", "新楼"],
            "图书馆": ["图书馆", "借书", "自习"],
            "一食堂": ["一食堂", "德智食堂"],
            "二食堂": ["二食堂", "五食堂", "天马食堂"],
            "德智学生公寓": ["德智", "德智公寓", "宿舍"],
            "天马学生公寓": ["天马", "天马公寓", "宿舍"],
            "东方红广场": ["东方红", "广场", "毛主席", "雕像"],
            "校医院": ["校医院", "医院", "看病", "医务室"],
            "体育场": ["体育场", "操场", "足球", "跑步"],
            "研究生院": ["研究生院", "研究生"],
        }

        for name, keywords in location_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return name

        return None

    def _build_nav_data(self, location_name: str) -> Dict[str, str]:
        """构建导航数据结构。"""
        import re
        pattern = rf'【{location_name}】\n区域：(.*?)\n简介：(.*?)\n周边：(.*?)\n路线：(.*?)(?:\n|$)'
        match = re.search(pattern, self.LOCATIONS_KNOWLEDGE)
        if not match:
            return {
                "destination": location_name,
                "area": "",
                "description": "",
                "route": "",
                "nearby": "",
                "walk_time": "",
            }

        area, description, nearby, route = match.groups()

        # 提取步行时间
        walk_time = ""
        time_match = re.search(r'(\d+)分钟', route)
        if time_match:
            walk_time = f"约{time_match.group(1)}分钟"

        return {
            "destination": location_name,
            "area": area.strip(),
            "description": description.strip(),
            "route": route.strip(),
            "nearby": nearby.strip(),
            "walk_time": walk_time,
        }

    def _generate_fallback_response(self, query: str) -> str:
        """当LLM不可用时，基于关键词匹配生成简单回答。"""
        query_lower = query.lower()

        # 简单的关键词匹配，用于确定相关地点
        location_keywords = {
            "岳麓书院": ["岳麓书院", "书院", "爱晚亭", "岳麓山"],
            "复临舍": ["复临舍", "教学楼", "教室"],
            "综合楼": ["综合楼", "新楼"],
            "图书馆": ["图书馆", "借书", "自习"],
            "一食堂": ["一食堂", "德智食堂"],
            "二食堂": ["二食堂", "五食堂", "天马食堂"],
            "德智学生公寓": ["德智", "德智公寓", "宿舍"],
            "天马学生公寓": ["天马", "天马公寓", "宿舍"],
            "东方红广场": ["东方红", "广场", "毛主席", "雕像"],
            "校医院": ["校医院", "医院", "看病", "医务室"],
            "体育场": ["体育场", "操场", "足球", "跑步"],
            "研究生院": ["研究生院", "研究生"],
        }

        # 确定相关地点
        matched_locations = []
        for location, keywords in location_keywords.items():
            if any(kw in query_lower for kw in keywords):
                matched_locations.append(location)

        if matched_locations:
            return (
                f"我了解到你想去{', '.join(matched_locations)}。"
                "由于智能生成服务暂时不可用，我建议你：\n"
                "1. 使用百度地图或高德地图搜索具体位置\n"
                "2. 询问校园内的同学或保安\n"
                "3. 关注学校官方公众号获取校园地图"
            )

        return (
            "我理解你想了解校园地点导航。"
            "我可以帮你指引岳麓书院、复临舍、综合楼、图书馆、一食堂、二食堂、"
            "德智公寓、天马公寓、东方红广场、校医院、体育场、研究生院等地点的路线。"
            "请告诉我具体想去哪里？"
        )

    def get_location_images(self, location_name: str) -> Tuple[Optional[str], List[str]]:
        """获取地点相关的图片路径。

        Returns:
            (地图图片路径, [实景照片路径列表])
        """
        # 从知识库中解析图片路径
        import re
        match = re.search(rf'【{location_name}】.*?图片资源：(.*?)(?=\n\n|\Z)', 
                         self.LOCATIONS_KNOWLEDGE, re.DOTALL)
        if not match:
            return None, []

        image_info = match.group(1)
        map_match = re.search(r'map_image=(\S+)', image_info)
        map_image = map_match.group(1) if map_match else None

        return map_image, []

    def get_all_locations(self) -> List[str]:
        """获取所有已知地点名称。"""
        import re
        matches = re.findall(r'【(.*?)】', self.LOCATIONS_KNOWLEDGE)
        return matches

    def set_asset_root(self, root_path: Union[str, Path]) -> None:
        """设置图片资源根目录。"""
        self.ASSETS_ROOT = Path(root_path)
