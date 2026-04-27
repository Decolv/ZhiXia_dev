"""校园生活指南工具 — 使用LLM智能生成校园生活相关答案

该工具不再使用硬编码FAQ，而是通过LLM动态生成个性化回答。
校园生活知识作为上下文注入LLM，使其能够基于知识生成答案。
"""

from typing import TYPE_CHECKING, Optional

from zhixia.agent.tool import Tool

if TYPE_CHECKING:
    from zhixia.agent.callbacks import CallbackManager


class CampusLifeGuideTool(Tool):
    """回答湖南大学校园生活相关问题，使用LLM智能生成答案。"""

    # 校园生活知识库 - 作为LLM上下文的参考信息，而非硬编码答案
    CAMPUS_KNOWLEDGE = """
【食堂信息】
湖南大学主要食堂：
- 一食堂（德智园区）：距离德智公寓最近，早餐丰富，价格最实惠
- 二食堂/五食堂（天马园区）：菜品最多，有特色美食档口，推荐麻辣烫和煲仔饭
- 三食堂（北校区）：北校同学的主要就餐地
- 四食堂（财院校区）：财院同学使用
用餐时间：早餐 7:00-9:00，午餐 11:00-13:00，晚餐 17:00-19:00
支付方式：校园一卡通或微信/支付宝
小贴士：天马美食街（二食堂旁）的夜宵非常有名！

【宿舍信息】
本科生宿舍主要分布在：
- 德智园区：四人间，上床下桌，有独立卫生间，距离教学楼近
- 天马园区：四人间/六人间，生活设施完善，楼下有超市和食堂
- 龙王港公寓：部分新生会安排在此
宿舍设施：空调（需租赁）、热水器、洗衣机（公共）、饮水机
门禁时间：周日至周四 23:30，周五周六 24:00
小贴士：天马园区生活最便利，但距离部分教学楼稍远，建议准备自行车

【快递服务】
主要快递点：
- 天马园区：菜鸟驿站（天猫/淘宝/韵达/中通/圆通）
- 德智园区：快递服务中心（顺丰/京东/邮政/申通）
- 综合楼旁：部分快递临时点
取件方式：凭取件码到对应驿站扫码取件
收货地址填写：湖南省长沙市岳麓区麓山南路2号湖南大学 + 所在园区
小贴士：开学季快递爆仓，建议提前3-5天寄出，或报到后再网购

【图书馆】
开放时间：
- 总馆：8:00-22:00（周一至周日）
- 自习区：部分区域延长至 23:00
- 寒暑假：另行通知
借阅规则：本科生可借 15 册，借期 30 天，可续借一次
预约座位：通过图书馆微信公众号或现场选座机预约
电子资源：校园网内免费访问知网、万方、IEEE、Springer 等数据库
小贴士：期末考前座位紧张，建议早上8点前到馆占座

【选课指南】
选课时间：每学期开学前 1-2 周
选课平台：湖南大学教务系统（http://jwc.hnu.edu.cn）
选课阶段：
- 第一轮：志愿式选课（抽签）
- 第二轮：先到先得
- 第三轮：补退选
学分要求：通识课约 30 学分，专业必修课约 60 学分，专业选修课约 20 学分
推荐通识课：《论语》精读、西方哲学史、心理学与生活、摄影艺术
小贴士：热门通识课（如影视鉴赏）秒光，建议提前了解课程评价

【交通出行】
地铁：
- 4号线湖南大学站：位于校园中心，A/B/C/D 四个出口
- 2号线溁湾镇站：可转公交到校园
公交：
- 麓山南路沿线：多条公交线路经过
- 常用线路：大科城1号线、2号线、3号线（校园环线）
自行车/电动车：
- 校园内有共享单车（哈啰、美团）
- 建议自备自行车，校园坡多，电动车更方便
小贴士：早八课前麓山南路非常拥堵，建议提前15分钟出门

【校园网络】
覆盖范围：教学区、图书馆、宿舍区全覆盖
连接方式：
- WiFi：选择 HNU 或 HNU-Stu，用学号登录
- 有线：宿舍有网线接口
收费标准：每月免费流量 20GB，超出部分按量计费
VPN：校外访问校内资源需连接 VPN（vpn.hnu.edu.cn）
小贴士：图书馆网速最快，宿舍区晚上高峰期可能较慢

【医疗服务】
校医院：
- 位置：麓山南路，图书馆北侧
- 门诊时间：工作日 8:00-12:00, 14:30-17:30
- 急诊：24小时（夜间电话值班）
医保：入学后统一购买大学生医保
校外医院：
- 最近三甲：湖南省人民医院（岳麓山院区）、中南大学湘雅三医院
- 湘雅三医院在桐梓坡路，公交约20分钟
小贴士：小病去校医院很方便且便宜，大病建议直接去湘雅三医院

【社团与活动】
社团类型：
- 学术科技类：机器人协会、ACM程序设计协会、电子设计协会
- 文化艺术类：合唱团、话剧社、街舞社、摄影协会
- 体育健身类：篮球协会、羽毛球协会、登山协会、龙舟队
- 公益实践类：青年志愿者协会、支教团
百团大战：每年开学后 1-2 个月，各社团在东方红广场招新
学生组织：学生会、团委、各类学生社团
小贴士：建议加入 1-2 个社团即可，过多会影响学业

【奖助学金】
奖学金：
- 国家奖学金：8000元/年（成绩排名前1%）
- 国家励志奖学金：5000元/年（贫困+成绩优秀）
- 校级奖学金：一等2000元、二等1000元、三等500元
- 专项奖学金：由企业或个人捐赠设立
助学金：
- 国家助学金：2000-4000元/年，按贫困等级评定
- 助学贷款：生源地信用助学贷款，最高12000元/年
申请时间：每学年开学初（9月）
小贴士：成绩是奖学金的核心指标，大一的成绩尤为重要
"""

    def __init__(self, llm_engine=None, callbacks: Optional["CallbackManager"] = None):
        super().__init__(
            name="campus_life_guide",
            description=(
                "解答湖南大学校园生活相关问题。"
                "输入问题（如'食堂在哪里'、'宿舍怎么样'、'怎么取快递'），"
                "返回智能生成的个性化生活指南。"
            ),
            func=self._guide,
        )
        self._llm_engine = llm_engine
        self._callbacks = callbacks

    def set_llm_engine(self, llm_engine) -> None:
        """设置LLM引擎，用于动态生成答案。"""
        self._llm_engine = llm_engine

    def set_callbacks(self, callbacks: "CallbackManager") -> None:
        """设置回调管理器，用于播报思考过程。"""
        self._callbacks = callbacks

    def _guide(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "你想了解校园生活的哪方面？比如食堂、宿舍、快递、图书馆、选课等。"

        # 触发思考开始回调
        if self._callbacks:
            self._callbacks.on_thinking_start("campus_life_guide")

        # 如果没有LLM引擎，返回知识库摘要
        if self._llm_engine is None:
            result = self._generate_fallback_response(query)
            if self._callbacks:
                self._callbacks.on_thinking_end("campus_life_guide")
            return result

        # 使用LLM生成答案
        try:
            result = self._generate_with_llm(query)
        except Exception as e:
            # LLM生成失败时回退到知识库匹配
            result = self._generate_fallback_response(query)

        # 触发思考结束回调
        if self._callbacks:
            self._callbacks.on_thinking_end("campus_life_guide")

        return result

    def _generate_with_llm(self, query: str) -> str:
        """使用LLM动态生成个性化答案。"""
        from zhixia.llm.base import LLMMessage

        # 构建系统提示词
        system_prompt = f"""你是湖南大学校园生活助手。请基于以下校园知识，为用户生成个性化的生活指南回答。

【校园知识参考】
{self.CAMPUS_KNOWLEDGE}

【回答要求】
1. 基于上述知识，针对用户的具体问题生成个性化回答
2. 回答要自然、友好，像学长学姐给新生建议一样
3. 如果知识库中没有相关信息，请诚实告知
4. 可以适当添加实用的生活小贴士
5. 回答控制在200字以内，简洁明了
"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=query),
        ]

        # 触发思考过程回调
        if self._callbacks:
            self._callbacks.on_agent_thought("campus_life_guide", f"正在分析用户问题: {query}")

        # 调用LLM生成答案
        response = self._llm_engine.chat(messages, max_new_tokens=256)

        # 播报生成完成
        if self._callbacks:
            self._callbacks.on_agent_thought("campus_life_guide", "答案生成完成")

        return response.strip()

    def _generate_fallback_response(self, query: str) -> str:
        """当LLM不可用时，基于关键词匹配生成简单回答。"""
        query_lower = query.lower()

        # 简单的关键词匹配，用于确定相关主题
        topics = {
            "食堂": ["食堂", "吃饭", "餐厅", "美食", "吃什么"],
            "宿舍": ["宿舍", "住宿", "寝室", "公寓", "住宿"],
            "快递": ["快递", "包裹", "取件", "菜鸟驿站", "物流"],
            "图书馆": ["图书馆", "借书", "自习", "座位", "学习"],
            "选课": ["选课", "课程", "学分", "教务", "培养方案"],
            "交通": ["交通", "地铁", "公交", "出行", "怎么去"],
            "校园网": ["校园网", "网络", "wifi", "上网", "流量"],
            "医务": ["医务", "医院", "看病", "医保", "校医院", "生病"],
            "社团": ["社团", "活动", "组织", "学生会", "兴趣"],
            "奖学金": ["奖学金", "助学金", "资助", "贷款", "经济"],
        }

        # 确定相关主题
        matched_topics = []
        for topic, keywords in topics.items():
            if any(kw in query_lower for kw in keywords):
                matched_topics.append(topic)

        if matched_topics:
            # 从知识库中提取相关信息
            return (
                f"我了解到你想了解关于{', '.join(matched_topics)}的信息。"
                "由于智能生成服务暂时不可用，我建议你：\n"
                "1. 咨询学长学姐获取第一手经验\n"
                "2. 关注学校官方公众号获取最新信息\n"
                "3. 加入新生群与其他同学交流"
            )

        return (
            "我理解你想了解校园生活相关问题。"
            "我可以帮你解答食堂、宿舍、快递、图书馆、选课、交通、校园网、医务、社团、奖学金等方面的问题。"
            "请告诉我更具体想了解什么？"
        )
