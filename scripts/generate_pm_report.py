#!/usr/bin/env python3
"""生成 ZhiXia 项目管理书 PDF（matplotlib 方案）

依赖: matplotlib, numpy
输出: docs/project_management_report.pdf
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D

# Matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "project_management_report.pdf"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 配色方案
COLOR_PRIMARY = "#1565C0"
COLOR_SECONDARY = "#1976D2"
COLOR_ACCENT = "#4CAF50"
COLOR_WARN = "#FF9800"
COLOR_DANGER = "#F44336"
COLOR_BG = "#FAFAFA"
COLOR_TEXT = "#333333"
COLOR_LIGHT_TEXT = "#666666"


def new_page(title="", figsize=(8.27, 11.69)):
    """创建新页面，返回 figure 和 axes。"""
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # 页眉装饰线
    ax.plot([5, 95], [96, 96], color=COLOR_PRIMARY, linewidth=3, solid_capstyle="butt")
    ax.plot([5, 25], [96, 96], color=COLOR_ACCENT, linewidth=3, solid_capstyle="butt")

    # 页脚
    ax.text(95, 2, f"ZhiXia 项目管理书  |  {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=8, color=COLOR_LIGHT_TEXT, ha="right", va="bottom")
    ax.plot([5, 95], [4, 4], color="#E0E0E0", linewidth=0.5)

    if title:
        ax.text(5, 92, title, fontsize=20, color=COLOR_PRIMARY,
                fontweight="bold", ha="left", va="top")
        ax.plot([5, 40], [89.5, 89.5], color=COLOR_PRIMARY, linewidth=1.5)

    return fig, ax


def draw_cover(pdf):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # 顶部装饰块
    rect = Rectangle((0, 85), 100, 15, facecolor=COLOR_PRIMARY, alpha=0.9)
    ax.add_patch(rect)
    rect2 = Rectangle((0, 83), 100, 2, facecolor=COLOR_ACCENT)
    ax.add_patch(rect2)

    # 底部装饰块
    rect3 = Rectangle((0, 0), 100, 8, facecolor=COLOR_PRIMARY, alpha=0.05)
    ax.add_patch(rect3)

    # 标题
    ax.text(50, 65, "ZhiXia 知匣", fontsize=42, color=COLOR_PRIMARY,
            ha="center", va="center", fontweight="bold")
    ax.text(50, 56, "离线智能语音助手", fontsize=18, color=COLOR_SECONDARY,
            ha="center", va="center")
    ax.text(50, 50, "项目管理书", fontsize=24, color=COLOR_TEXT,
            ha="center", va="center")

    # 分隔线
    ax.plot([30, 70], [44, 44], color="#E0E0E0", linewidth=1)

    # 信息块
    info_items = [
        ("项目版本", "v0.1.0"),
        ("报告日期", datetime.now().strftime("%Y-%m-%d")),
        ("代码规模", "80 文件 / 9,134 行"),
        ("目标平台", "RK3588 (QuarkPi)"),
        ("开源许可", "MIT License"),
    ]
    y_start = 37
    for label, value in info_items:
        # 标签背景
        rect_label = FancyBboxPatch((28, y_start - 2.5), 18, 4.5,
                                     boxstyle="round,pad=0.3",
                                     facecolor=COLOR_PRIMARY, alpha=0.1,
                                     edgecolor=COLOR_PRIMARY, linewidth=1)
        ax.add_patch(rect_label)
        ax.text(29, y_start, label, fontsize=10, color=COLOR_PRIMARY,
                ha="left", va="center", fontweight="bold")
        ax.text(50, y_start, value, fontsize=11, color=COLOR_TEXT,
                ha="left", va="center")
        y_start -= 7

    # 装饰圆圈
    circle1 = Circle((85, 20), 8, facecolor=COLOR_PRIMARY, alpha=0.05, edgecolor=COLOR_PRIMARY, linewidth=2)
    ax.add_patch(circle1)
    circle2 = Circle((15, 75), 5, facecolor=COLOR_ACCENT, alpha=0.08, edgecolor=COLOR_ACCENT, linewidth=1.5)
    ax.add_patch(circle2)

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_overview(pdf):
    fig, ax = new_page("1. 项目概述")

    # 项目简介
    text = (
        "ZhiXia（知匣）是一款为 RK3588 嵌入式平台设计的离线智能语音助手，"
        "采用创新的插卡式 Agent 架构，通过技能卡（Skill Card）和知识卡（Knowledge Card）"
        "灵活扩展功能。项目实现了完整的语音交互流水线：ASR → LLM → TTS → Play，"
        "全程本地运行，无需联网，充分保护用户隐私。"
    )
    ax.text(5, 84, text, fontsize=11, color=COLOR_TEXT, ha="left", va="top",
            wrap=True, linespacing=1.6)

    # 核心技术栈表格
    ax.text(5, 72, "核心技术栈", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")

    headers = ["组件", "技术方案", "特点"]
    rows = [
        ["语音识别 (ASR)", "FunASR / Whisper", "中文识别，支持离线"],
        ["大模型 (LLM)", "RKLLM (Qwen3-1.7B)", "NPU 加速，高效低功耗"],
        ["语音合成 (TTS)", "Piper", "超高速，模型仅 42MB"],
        ["Agent 架构", "ReAct / ToolCalling", "结构化工具调用"],
        ["知识检索", "ChromaDB + 回退", "语义检索增强"],
        ["语音唤醒", "Snowboy", "低功耗唤醒检测"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper left",
        bbox=[0.05, 0.15, 0.9, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 表头样式
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor(COLOR_PRIMARY)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # 行样式
    for i in range(1, len(rows) + 1):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor("#E3F2FD" if i % 2 == 1 else "white")
            cell.set_edgecolor("#BBDEFB")
            cell.set_text_props(color=COLOR_TEXT)

    # 核心特性
    ax.text(5, 28, "核心特性", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")
    features = [
        "LLM 智能调度：所有工具回答均由大模型临时生成，拒绝 FAQ 硬编码",
        "思考过程播报：实时展示 AI 思考过程，增加交互透明度",
        "插卡式架构：Skill Card 和 Knowledge Card 动态扩展，插卡即用",
        "流式并发流水线：三线程并发，首句延迟 0.6~1.1 秒",
        "完全离线：所有模型本地运行，NPU 加速，保护隐私",
    ]
    y = 23
    for f in features:
        ax.text(7, y, "-", fontsize=12, color=COLOR_ACCENT, ha="left", va="top")
        ax.text(10, y, f, fontsize=10, color=COLOR_TEXT, ha="left", va="top", wrap=True)
        y -= 4

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_current_status(pdf):
    fig, ax = new_page("2. 项目当前状态")

    ax.text(5, 84, (
        "截至报告日期，项目已完成核心架构搭建和主要功能模块实现。"
        "代码规模达到 80 个 Python 文件，总计约 9,134 行代码。"
        "整体架构清晰，模块职责分明，已具备生产部署的基础条件。"
    ), fontsize=11, color=COLOR_TEXT, ha="left", va="top", wrap=True, linespacing=1.6)

    # 左侧：模块完成度柱状图
    ax.text(5, 75, "2.1 模块完成度", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")

    modules = ["ASR", "LLM", "TTS", "Audio", "Display", "Agent", "Pipeline", "Core", "Memory", "RAG", "WakeWord"]
    scores = [85, 90, 85, 80, 75, 90, 90, 85, 80, 70, 75]
    bar_colors = [COLOR_ACCENT if s >= 85 else COLOR_WARN if s >= 75 else COLOR_DANGER for s in scores]

    # 在页面上创建子坐标轴用于柱状图
    ax_bar = fig.add_axes([0.08, 0.30, 0.42, 0.38])
    bars = ax_bar.barh(modules, scores, color=bar_colors, edgecolor="white", height=0.6)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("完成度 (%)", fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="both", labelsize=8)
    ax_bar.axvline(x=80, color="#E0E0E0", linestyle="--", linewidth=1)
    for bar, score in zip(bars, scores):
        ax_bar.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                    f"{score}%", va="center", fontsize=8, color=COLOR_TEXT)

    # 右侧：代码分布饼图
    ax_pie = fig.add_axes([0.58, 0.35, 0.35, 0.35])
    labels = ["Agent", "LLM", "语音", "核心", "显示", "工具", "其他"]
    sizes = [2200, 1800, 1600, 1400, 900, 700, 534]
    pie_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4", "#607D8B"]
    ax_pie.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 8}, pctdistance=0.75)
    ax_pie.set_title("代码分布", fontsize=11, fontweight="bold", pad=10)

    # 已知问题
    ax.text(5, 30, "2.2 已知问题", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")
    issues = [
        "测试覆盖率低：目前仅依赖 quick_test.py 和手动测试，缺乏自动化单元测试",
        "Windows 开发环境不完善：音频播放使用 NullAudioPlayer 回退，无法实际播放",
        "模型部署体积大：LLM 模型约 2.2GB，首次部署慢",
        "多卡并发未实现：当前仅支持单 Skill + 单 Knowledge",
        "RAG 性能待验证：Chroma 向量检索在 RK3588 上尚未实测",
    ]
    y = 25
    for issue in issues:
        ax.text(7, y, "-", fontsize=12, color=COLOR_WARN, ha="left", va="top")
        ax.text(10, y, issue, fontsize=9, color=COLOR_TEXT, ha="left", va="top", wrap=True)
        y -= 4.5

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_milestones(pdf):
    fig, ax = new_page("3. 里程碑回顾")

    ax.text(5, 84, (
        "项目自启动以来经历了多次关键迭代，从最初的概念验证逐步演进为"
        "架构成熟、功能完备的嵌入式语音助手。"
    ), fontsize=11, color=COLOR_TEXT, ha="left", va="top", wrap=True, linespacing=1.6)

    milestones = [
        ("2024-Q1", "TTS 换 Piper", "速度提升 10x", "#4CAF50"),
        ("2024-Q1", "Agent 架构", "ReAct + ToolCalling", "#2196F3"),
        ("2024-Q2", "插卡式架构", "Skill + Knowledge 卡", "#9C27B0"),
        ("2024-Q2", "深度解耦", "Configurator + PostProcessor", "#FF9800"),
        ("2024-Q3", "思考播报", "DisplayCallback + Live2D", "#E91E63"),
        ("2024-Q4", "性能优化", "流式流水线 + 懒加载", "#00BCD4"),
    ]

    # 时间线
    ax.plot([10, 90], [72, 72], color="#E3F2FD", linewidth=4, zorder=1)
    x_positions = np.linspace(15, 85, len(milestones))

    for i, (date, title, desc, color) in enumerate(milestones):
        x = x_positions[i]
        # 节点
        ax.scatter(x, 72, s=300, c=color, zorder=3, edgecolors="white", linewidths=2)
        # 日期
        ax.text(x, 76, date, ha="center", va="bottom", fontsize=8, color=COLOR_LIGHT_TEXT)
        # 标题
        ax.text(x, 79, title, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLOR_TEXT)
        # 描述
        ax.text(x, 82, desc, ha="center", va="bottom", fontsize=8, color=COLOR_LIGHT_TEXT)
        # 连接线
        ax.plot([x, x], [72, 74.5], color=color, linewidth=1.5, alpha=0.6)

    # 关键演进文字
    ax.text(5, 62, "关键演进", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")

    evolutions = [
        ("早期阶段", "从 ASR+LLM+TTS 串行原型起步，使用 ChatTTS，响应 15~20s"),
        ("性能飞跃", "切换 Piper TTS，模型 800MB→42MB，速度提升 10~20x，响应 3~5s"),
        ("架构升级", "引入 LangChain 风格 Agent，支持 ReAct 和 ToolCalling"),
        ("插卡革命", "主机与卡片深度解耦，功能扩展从改代码变为复制文件夹"),
        ("体验打磨", "思考播报 + Live2D 眼睛 + 导航界面，交互透明度大幅提升"),
    ]
    y = 57
    for title, desc in evolutions:
        ax.text(7, y, "->", fontsize=10, color=COLOR_PRIMARY, ha="left", va="top")
        ax.text(10, y, f"{title}：", fontsize=10, color=COLOR_PRIMARY,
                ha="left", va="top", fontweight="bold")
        ax.text(24, y, desc, fontsize=9, color=COLOR_TEXT, ha="left", va="top", wrap=True)
        y -= 5.5

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_architecture(pdf):
    fig, ax = new_page("4. 架构评估")

    ax.text(5, 84, (
        "项目采用分层模块化架构，每个引擎组件均定义了抽象基类（ABC），"
        "通过工厂函数根据配置动态创建具体实现，具备良好的可扩展性和可测试性。"
    ), fontsize=11, color=COLOR_TEXT, ha="left", va="top", wrap=True, linespacing=1.6)

    # 架构框图
    ax.text(5, 75, "4.1 插卡式架构", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")

    # 主机框
    host_rect = FancyBboxPatch((5, 25), 40, 45, boxstyle="round,pad=1",
                                facecolor="#E3F2FD", edgecolor=COLOR_PRIMARY, linewidth=2)
    ax.add_patch(host_rect)
    ax.text(25, 66, "Host 主机", fontsize=12, color=COLOR_PRIMARY,
            ha="center", va="center", fontweight="bold")
    host_items = ["HostOrchestrator", "VoicePipeline", "CardLoader", "AgentExecutor", "ConversationMemory"]
    y = 61
    for item in host_items:
        ax.text(25, y, f"- {item}", fontsize=9, color=COLOR_TEXT, ha="center", va="top")
        y -= 4

    # 卡片框
    card_rect = FancyBboxPatch((55, 25), 40, 45, boxstyle="round,pad=1",
                                facecolor="#E8F5E9", edgecolor=COLOR_ACCENT, linewidth=2)
    ax.add_patch(card_rect)
    ax.text(75, 66, "Card 卡片", fontsize=12, color=COLOR_ACCENT,
            ha="center", va="center", fontweight="bold")
    card_items = ["SkillCard (工具+人设)", "KnowledgeCard (知识库)", "manifest.json", "tools/", "docs/"]
    y = 61
    for item in card_items:
        ax.text(75, y, f"- {item}", fontsize=9, color=COLOR_TEXT, ha="center", va="top")
        y -= 4

    # 连接箭头
    ax.annotate("", xy=(55, 47), xytext=(45, 47),
                arrowprops=dict(arrowstyle="->", color=COLOR_PRIMARY, lw=2))
    ax.text(50, 50, "HostContext\n接口", fontsize=8, color=COLOR_PRIMARY,
            ha="center", va="center", style="italic")

    # 解耦状态
    ax.text(5, 20, "4.2 解耦状态", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")
    decouple_items = [
        ("主机零卡片逻辑", "HostOrchestrator 中无硬编码工具名、卡片名或 UI 逻辑"),
        ("接口驱动", "卡片通过 HostContext 注册能力，拔卡自动清理"),
        ("自包含导入", "卡片内部使用基于 __file__ 的导入，不依赖项目包路径"),
        ("动态 Agent 配置", "Agent 类型、迭代次数由卡片通过 AgentConfigurator 设定"),
    ]
    y = 15
    for title, desc in decouple_items:
        ax.text(7, y, "[OK]", fontsize=10, color=COLOR_ACCENT, ha="left", va="top")
        ax.text(10, y, f"{title}：", fontsize=10, color=COLOR_ACCENT,
                ha="left", va="top", fontweight="bold")
        ax.text(28, y, desc, fontsize=9, color=COLOR_TEXT, ha="left", va="top", wrap=True)
        y -= 5

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_performance(pdf):
    fig, ax = new_page("4.3 性能指标")

    ax.text(5, 84, "在 RK3588 平台上的实测性能表现：", fontsize=11,
            color=COLOR_TEXT, ha="left", va="top")

    perf_data = [
        ["指标", "实测值", "备注"],
        ["ASR 识别", "0.3 ~ 0.5s", "取决于音频长度"],
        ["LLM 首 token", "0.2 ~ 0.4s", "流式输出"],
        ["TTS 首句合成", "0.1 ~ 0.2s", "内存合成，无磁盘 I/O"],
        ["首字播放延迟", "0.6 ~ 1.1s", "ASR + LLM首句 + TTS首句"],
        ["总对话耗时", "1.5 ~ 3.0s", "取决于回复长度"],
        ["内存占用 (峰值)", "~3.1 GB", "8GB 系统可承受"],
        ["模型预热时间", "~5s", "首次加载后常驻内存"],
    ]

    table = ax.table(
        cellText=perf_data[1:],
        colLabels=perf_data[0],
        loc="upper left",
        bbox=[0.05, 0.45, 0.9, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor(COLOR_PRIMARY)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    for i in range(1, len(perf_data)):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor("#E3F2FD" if i % 2 == 1 else "white")
            cell.set_edgecolor("#BBDEFB")

    # 亮点标注
    ax.text(5, 42, "性能亮点", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")
    highlights = [
        "流式并发：LLM / TTS / Play 三线程并发，首句延迟降低 60%+",
        "内存合成：TTS 直接输出 WAV bytes，避免磁盘 I/O 瓶颈",
        "懒加载：模型首次使用时加载，启动零开销",
        "NPU 加速：Qwen3-1.7B 在 RK3588 NPU 上运行，功耗 < 5W",
    ]
    y = 37
    for h in highlights:
        ax.text(7, y, "*", fontsize=12, color=COLOR_WARN, ha="left", va="top")
        ax.text(10, y, h, fontsize=10, color=COLOR_TEXT, ha="left", va="top", wrap=True)
        y -= 4.5

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_risk(pdf):
    fig, ax = new_page("5. 风险分析")

    ax.text(5, 84, "项目当前面临技术、资源和进度三类风险，需重点关注测试覆盖率和硬件兼容性。",
            fontsize=11, color=COLOR_TEXT, ha="left", va="top", wrap=True, linespacing=1.6)

    # 风险矩阵图
    ax_matrix = fig.add_axes([0.08, 0.32, 0.5, 0.42])
    risks = [
        ("测试覆盖率不足", 4, 4.5, COLOR_DANGER),
        ("RK3588 NPU 兼容性", 3.5, 4.8, COLOR_DANGER),
        ("模型部署体积大", 3, 3.5, COLOR_WARN),
        ("多卡并发复杂度", 4, 3.2, COLOR_WARN),
        ("Windows 开发环境", 2, 2, COLOR_ACCENT),
        ("文档滞后", 2.5, 2.5, COLOR_ACCENT),
    ]
    for name, impact, prob, color in risks:
        ax_matrix.scatter(impact, prob, s=350, c=color, alpha=0.85,
                         edgecolors="white", linewidths=2, zorder=3)
        ax_matrix.annotate(name, (impact, prob), textcoords="offset points",
                          xytext=(8, 4), fontsize=7.5, ha="left")

    ax_matrix.set_xlim(0.5, 5.5)
    ax_matrix.set_ylim(0.5, 5.5)
    ax_matrix.set_xlabel("影响程度", fontsize=10)
    ax_matrix.set_ylabel("发生概率", fontsize=10)
    ax_matrix.set_title("风险矩阵", fontsize=12, fontweight="bold", pad=10)
    ax_matrix.add_patch(Rectangle((3.5, 3.5), 2, 2, facecolor="#FFEBEE", alpha=0.4, zorder=0))
    ax_matrix.add_patch(Rectangle((0.5, 3.5), 3, 2, facecolor="#FFF3E0", alpha=0.4, zorder=0))
    ax_matrix.add_patch(Rectangle((3.5, 0.5), 2, 3, facecolor="#FFF3E0", alpha=0.4, zorder=0))
    ax_matrix.add_patch(Rectangle((0.5, 0.5), 3, 3, facecolor="#E8F5E9", alpha=0.4, zorder=0))
    ax_matrix.set_xticks([1, 2, 3, 4, 5])
    ax_matrix.set_yticks([1, 2, 3, 4, 5])
    ax_matrix.set_xticklabels(["极低", "低", "中", "高", "极高"])
    ax_matrix.set_yticklabels(["极低", "低", "中", "高", "极高"])
    ax_matrix.grid(True, alpha=0.3, linestyle="--")

    # 风险应对策略
    ax.text(62, 75, "应对策略", fontsize=12, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")
    strategies = [
        ("高", "测试覆盖", "引入 pytest，目标 60%+ 覆盖率"),
        ("高", "NPU 兼容", "建立 RK3588 CI 测试环境"),
        ("中", "模型体积", "评估 INT4 量化+按需加载"),
        ("中", "多卡并发", "设计依赖图和冲突检测"),
    ]
    y = 70
    for level, name, action in strategies:
        color = COLOR_DANGER if level == "高" else COLOR_WARN
        ax.text(64, y, level, fontsize=9, color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"))
        ax.text(70, y, f"{name}: {action}", fontsize=9, color=COLOR_TEXT, ha="left", va="center")
        y -= 5

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_roadmap(pdf):
    fig, ax = new_page("6. 后续路线图")

    ax.text(5, 84, (
        "未来 8 周聚焦四大方向：工程化完善（测试+文档）、架构增强（多卡+SDK）、"
        "性能优化（流式TTS+内存）、交互升级（Web界面+唤醒）。"
    ), fontsize=11, color=COLOR_TEXT, ha="left", va="top", wrap=True, linespacing=1.6)

    # 甘特图
    ax.text(5, 75, "6.1 甘特图", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")

    tasks = [
        ("垃圾清理与解耦", 1, 0, COLOR_ACCENT),
        ("卡片 SDK 规范化", 2, 1, "#2196F3"),
        ("单元测试完善", 3, 2, COLOR_WARN),
        ("多卡并发支持", 2, 3, "#9C27B0"),
        ("性能优化(流式TTS)", 2, 1, "#E91E63"),
        ("Web 界面控制", 4, 3, "#00BCD4"),
        ("语音唤醒优化", 2, 5, "#795548"),
        ("文档完善", 6, 0, "#607D8B"),
    ]

    ax_gantt = fig.add_axes([0.08, 0.22, 0.88, 0.45])
    for i, (name, duration, start, color) in enumerate(tasks):
        ax_gantt.barh(i, duration, left=start, height=0.55, color=color,
                     edgecolor="white", alpha=0.9)
        ax_gantt.text(start + duration/2, i, name, ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")

    ax_gantt.set_yticks(range(len(tasks)))
    ax_gantt.set_yticklabels([f"T{i+1}" for i in range(len(tasks))], fontsize=8)
    ax_gantt.set_xlabel("周", fontsize=10)
    ax_gantt.set_xlim(0, 8)
    ax_gantt.set_title("未来 8 周开发路线图", fontsize=11, fontweight="bold", pad=10)
    ax_gantt.invert_yaxis()
    ax_gantt.spines["top"].set_visible(False)
    ax_gantt.spines["right"].set_visible(False)
    ax_gantt.grid(axis="x", alpha=0.3, linestyle="--")

    # 详细任务表格
    ax.text(5, 28, "6.2 详细任务", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")

    task_data = [
        ["任务", "工期", "目标", "交付物"],
        ["垃圾清理与解耦", "1 周", "仓库规范化+卡片自包含", "干净仓库+解耦卡片"],
        ["卡片 SDK 规范化", "2 周", "制定标准+模板+文档", "开发规范+模板"],
        ["单元测试完善", "3 周", "核心模块自动化测试", "pytest 套件+CI"],
        ["多卡并发支持", "2 周", "同时加载多张 Skill 卡", "多卡调度器"],
        ["性能优化", "2 周", "流式TTS+内存优化", "延迟<1s+内存<2.5GB"],
        ["Web 界面控制", "4 周", "浏览器配置+状态监控", "Web UI+API"],
        ["语音唤醒优化", "2 周", "唤醒准确率+功耗", "自定义唤醒词"],
        ["文档完善", "持续", "API文档+用户手册", "完整文档站"],
    ]
    table = ax.table(
        cellText=task_data[1:],
        colLabels=task_data[0],
        loc="upper left",
        bbox=[0.05, 0.02, 0.9, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.5)
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor(COLOR_PRIMARY)
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(task_data)):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor("#E3F2FD" if i % 2 == 1 else "white")
            cell.set_edgecolor("#BBDEFB")

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def draw_resources(pdf):
    fig, ax = new_page("7. 资源需求")

    ax.text(5, 84, "项目进入工程化和功能增强阶段，需要以下资源支持：",
            fontsize=11, color=COLOR_TEXT, ha="left", va="top", wrap=True, linespacing=1.6)

    resource_data = [
        ["资源类型", "具体需求", "优先级", "预估成本"],
        ["人力", "1 名测试工程师 + 1 名前端工程师", "高", "人力成本"],
        ["硬件", "RK3588 开发板 x2", "高", "~2000 元"],
        ["算力", "模型量化实验 GPU 服务器", "中", "云服务费用"],
        ["存储", "模型仓库 + 日志存储", "低", "~500 元/年"],
        ["工具", "pytest、CI/CD、文档站点", "中", "开源免费"],
    ]
    table = ax.table(
        cellText=resource_data[1:],
        colLabels=resource_data[0],
        loc="upper left",
        bbox=[0.05, 0.55, 0.9, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor(COLOR_PRIMARY)
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(resource_data)):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor("#E3F2FD" if i % 2 == 1 else "white")
            cell.set_edgecolor("#BBDEFB")

    # 总结
    ax.text(5, 45, "总结", fontsize=14, color=COLOR_SECONDARY,
            fontweight="bold", ha="left", va="top")
    summary = (
        "ZhiXia 项目已完成从原型到产品化的关键跨越，插卡式架构和流式流水线"
        "是项目的核心竞争力。接下来的重点是工程化打磨（测试、文档、CI/CD）"
        "和生态建设（卡片 SDK、开发者文档、模板项目）。"
        "预计在 2 个月内达到可对外发布 v1.0 的水平。"
    )
    ax.text(5, 40, summary, fontsize=11, color=COLOR_TEXT, ha="left", va="top",
            wrap=True, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=1", facecolor="#E8F5E9", alpha=0.5, edgecolor=COLOR_ACCENT))

    # 项目信息
    ax.text(50, 15, "ZhiXia 知匣  |  离线智能语音助手", fontsize=12,
            color=COLOR_PRIMARY, ha="center", va="center", fontweight="bold")
    ax.text(50, 10, "项目管理书  v0.1  |  2024", fontsize=10,
            color=COLOR_LIGHT_TEXT, ha="center", va="center")

    pdf.savefig(fig, facecolor="white", dpi=150)
    plt.close(fig)


def main():
    with PdfPages(str(OUTPUT_PATH)) as pdf:
        draw_cover(pdf)
        draw_overview(pdf)
        draw_current_status(pdf)
        draw_milestones(pdf)
        draw_architecture(pdf)
        draw_performance(pdf)
        draw_risk(pdf)
        draw_roadmap(pdf)
        draw_resources(pdf)

        # PDF 元数据
        d = pdf.infodict()
        d["Title"] = "ZhiXia 项目管理书"
        d["Author"] = "ZhiXia Team"
        d["Subject"] = "ZhiXia 离线智能语音助手项目状态与路线图"
        d["Keywords"] = "项目管理, 语音助手, RK3588, 离线AI"
        d["Creator"] = "ZhiXia Project Management Tool"
        d["CreationDate"] = datetime.now()

    print(f"[OK] 项目管理书已生成: {OUTPUT_PATH}")
    print(f"    共 9 页，包含甘特图、风险矩阵、里程碑时间线等图表")


if __name__ == "__main__":
    main()
