"""
Streamlit UI for Multi-Agent Regulator-Manufacturer Simulation
With math iteration + animation
Demo for SHMDIRI talk 2026-05-28
"""

import os
import time

import pandas as pd
import streamlit as st

from agent_sim import (
    LLMBackend, REGULATOR, MANUFACTURERS, SCENARIO,
    parse_risk_score,
)

st.set_page_config(
    page_title="Multi-Agent 监管动态模拟器",
    page_icon="🤝",
    layout="wide",
)

# ============== Global Style — Enterprise Dashboard ==============
st.markdown(
    """<style>
    /* ——— Typography & background ——— */
    html, body, [class*="css"] {
        font-family: -apple-system, "SF Pro Display", "Inter", "PingFang SC",
                     "Helvetica Neue", sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(59,130,246,0.04) 0%, transparent 40%),
            radial-gradient(circle at 80% 0%, rgba(168,85,247,0.03) 0%, transparent 40%),
            linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 600px);
    }
    .main .block-container { padding-top: 2.2rem; max-width: 1320px; }

    /* ——— Primary button: gradient + glow ——— */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 60%, #6366F1 100%) !important;
        border: 0 !important;
        box-shadow: 0 8px 24px -8px rgba(59,130,246,0.55),
                    0 0 0 1px rgba(255,255,255,0.06) inset !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        font-size: 16px !important;
        padding: 16px 0 !important;
        border-radius: 12px !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 32px -8px rgba(59,130,246,0.65) !important;
    }

    /* ——— st.metric → KPI tiles ——— */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04),
                    0 8px 24px -12px rgba(15,23,42,0.08);
    }
    [data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 11px !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: #64748B !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { font-weight: 700 !important; }

    /* ——— Expander polish ——— */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background: #FAFBFC !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.03) !important;
    }
    [data-testid="stExpander"] summary { font-weight: 600 !important; }

    /* ——— Charts background ——— */
    [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04),
                    0 8px 24px -12px rgba(15,23,42,0.06);
    }

    /* ——— Alerts ——— */
    .stAlert {
        border-radius: 12px !important;
        border-left-width: 4px !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04) !important;
    }

    /* ——— Sidebar: dark gradient ——— */
    [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: #FFFFFF !important; }
    [data-testid="stSidebar"] hr { border-color: #334155 !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label { color: #CBD5E1 !important; }

    /* ——— Custom dashboard classes ——— */
    .dash-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04),
                    0 12px 32px -16px rgba(15,23,42,0.10);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .dash-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 1px 3px rgba(15,23,42,0.04),
                    0 18px 40px -16px rgba(15,23,42,0.14);
    }
    .live-chip {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        color: #065F46; padding: 4px 12px; border-radius: 999px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.10em;
        border: 1px solid rgba(5,150,105,0.20);
    }
    .live-chip .dot {
        width: 6px; height: 6px; border-radius: 50%; background: #10B981;
        box-shadow: 0 0 0 3px rgba(16,185,129,0.20);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 3px rgba(16,185,129,0.20); }
        50%      { box-shadow: 0 0 0 6px rgba(16,185,129,0.10); }
    }
    .kpi-tile {
        flex: 1;
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04),
                    0 10px 28px -14px rgba(15,23,42,0.10);
        position: relative;
        overflow: hidden;
    }
    .kpi-tile::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
        background: linear-gradient(180deg, var(--accent, #3B82F6) 0%, transparent 100%);
    }
    .kpi-tile .label {
        font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
        color: #64748B; font-weight: 700;
    }
    .kpi-tile .value {
        font-size: 34px; font-weight: 800; color: #0F172A;
        letter-spacing: -0.02em; line-height: 1.1; margin-top: 6px;
    }
    .kpi-tile .hint {
        font-size: 11px; color: #94A3B8; margin-top: 4px;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ============== Sidebar ==============
with st.sidebar:
    st.title("⚙️ 模拟设置")
    st.markdown("**论文**: arXiv:2411.15356\n\n**作者**: Han, Guo (2024) · 10 引")
    st.markdown("---")
    model = st.selectbox("LLM 模型", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)
    anim_speed = st.slider("动画速度 (秒/帧)", 0.3, 2.0, 0.8, 0.1)
    show_math = st.checkbox("显示数学方程", value=True)
    st.markdown("---")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    if has_key:
        st.success("✅ 真实 LLM 模式")
    else:
        st.warning("⚠️ Mock 模式（确定性输出）")
    st.markdown("---")
    st.markdown("### 📋 场景")
    st.markdown(f"**{SCENARIO['title']}**")

# ============== Header ==============
st.markdown(
    """<div style="border-bottom:1px solid #E2E8F0; padding-bottom:24px; margin-bottom:24px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="font-size:15px; letter-spacing:0.14em; color:#64748B; text-transform:uppercase;
                        font-weight:700;">Regulator–Manufacturer AI Agents</div>
            <div class="live-chip"><span class="dot"></span>LIVE · IDLE</div>
        </div>
        <div style="font-size:52px; font-weight:800; line-height:1.10; margin-top:12px;
                    background:linear-gradient(135deg, #0F172A 0%, #1E40AF 60%, #6366F1 100%);
                    -webkit-background-clip:text; background-clip:text;
                    -webkit-text-fill-color:transparent; letter-spacing:-0.02em;">
            Multi-Agent 监管动态模拟器
        </div>
        <div style="font-size:21px; color:#334155; margin-top:14px; font-weight:500;">
            4 agent · 3 轮博弈 · 数学反馈方程驱动 · 量化合规漂移
        </div>
        <div style="font-size:14px; color:#94A3B8; margin-top:10px; font-style:italic;">
            Han Y, Guo Z. Regulator-Manufacturer AI Agents Modeling. arXiv:2411.15356, 2024.
        </div>
        <div style="font-size:13px; color:#475569; margin-top:14px;
                    background:linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%);
                    padding:10px 14px; border-radius:8px;
                    border-left:3px solid #6366F1;">
            <b>本次场景</b>　基于 FDA 2024-12 PCCP final guidance · NMPA YY/T 1833.5-2024
            《人工智能医疗器械质量要求和评价 第 5 部分：预训练模型》。
            <span style="color:#94A3B8;">演示用 hypothetical 征求意见稿。</span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ============== KPI Strip ==============
st.markdown(
    """<div style="display:flex; gap:14px; margin: 6px 0 28px 0;">
        <div class="kpi-tile" style="--accent:#3B82F6;">
            <div class="label">AGENTS</div>
            <div class="value">4</div>
            <div class="hint">1 regulator + 3 manufacturers</div>
        </div>
        <div class="kpi-tile" style="--accent:#8B5CF6;">
            <div class="label">ROUNDS</div>
            <div class="value">3</div>
            <div class="hint">issue → respond → adjust</div>
        </div>
        <div class="kpi-tile" style="--accent:#10B981;">
            <div class="label">EQUATIONS</div>
            <div class="value">4</div>
            <div class="hint">tanh-coupled feedback</div>
        </div>
        <div class="kpi-tile" style="--accent:#F59E0B;">
            <div class="label">STATE VARS</div>
            <div class="value">G·C·M·F</div>
            <div class="hint">guidance · compliance · market · feedback</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# Process flow diagram — dashboard tile with stronger depth
st.markdown(
    """<div class="dash-card" style="display:flex; justify-content:center; align-items:center; gap:28px;
                   padding:30px 24px; margin:10px 0 32px 0; flex-wrap:wrap;
                   background:linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);">
        <div style="text-align:center; min-width:108px;">
            <div style="font-size:44px; line-height:1; margin-bottom:10px;
                        filter:drop-shadow(0 4px 8px rgba(15,23,42,0.10));">⚖️</div>
            <div style="font-size:11px; color:#3B82F6; letter-spacing:0.14em; font-weight:800;">REGULATOR</div>
            <div style="font-size:20px; font-weight:700; color:#0F172A; margin-top:6px;">监管者</div>
        </div>
        <div style="color:#CBD5E1; font-size:30px; font-weight:300;">→</div>
        <div style="text-align:center; min-width:140px;">
            <div style="font-size:44px; line-height:1; margin-bottom:10px;
                        filter:drop-shadow(0 4px 8px rgba(15,23,42,0.10));">🏭</div>
            <div style="font-size:11px; color:#8B5CF6; letter-spacing:0.14em; font-weight:800;">MANUFACTURERS ×3</div>
            <div style="font-size:20px; font-weight:700; color:#0F172A; margin-top:6px;">生产者</div>
        </div>
        <div style="color:#CBD5E1; font-size:30px; font-weight:300;">→</div>
        <div style="text-align:center; min-width:108px;">
            <div style="font-size:44px; line-height:1; margin-bottom:10px;
                        filter:drop-shadow(0 4px 8px rgba(15,23,42,0.10));">📊</div>
            <div style="font-size:11px; color:#10B981; letter-spacing:0.14em; font-weight:800;">MARKET</div>
            <div style="font-size:20px; font-weight:700; color:#0F172A; margin-top:6px;">市场</div>
        </div>
        <div style="color:#CBD5E1; font-size:30px; font-weight:300;">↺</div>
        <div style="text-align:center; min-width:108px;">
            <div style="font-size:44px; line-height:1; margin-bottom:10px;
                        filter:drop-shadow(0 4px 8px rgba(15,23,42,0.10));">🔄</div>
            <div style="font-size:11px; color:#F59E0B; letter-spacing:0.14em; font-weight:800;">FEEDBACK</div>
            <div style="font-size:20px; font-weight:700; color:#0F172A; margin-top:6px;">反馈</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# Math equations overlay
if show_math:
    with st.expander("数学反馈方程（Section 3.2）", expanded=True):
        st.markdown(
            """- ⚖️ **G_i** — 监管者发布指南的严格度（Guidance issuance）
- 🏭 **C_m / G_m** — 厂商合规努力（Compliance effort）
- 📊 **M_m** — 市场适应（Market adaptation）
- 🔄 **F_m** — 厂商反馈回监管者（Feedback）"""
        )
        st.latex(r"""
        \begin{aligned}
        G_i(t+1) &= G_i(t) + \alpha \cdot \sum_m F_m(t)  && \text{Regulator update} \\
        G_m(t)   &= \tanh(\beta \cdot G_i(t) - \gamma \cdot \text{risk}_m) && \text{Compliance effort} \\
        M_m(t)   &= 1 - G_m(t) && \text{Market adaptation} \\
        F_m(t)   &= G_m(t) - M_m(t) && \text{Feedback to regulator}
        \end{aligned}
        """)
        st.caption("Transcendental equation — tanh 引入非线性饱和，避免线性平衡假设。")

# ============== Agent Cards ==============
st.markdown(
    """<div style="font-size:14px; letter-spacing:0.16em; color:#475569; font-weight:700;
                   text-transform:uppercase; margin:10px 0 14px 0;">Agents</div>""",
    unsafe_allow_html=True,
)
cols = st.columns(4)
agents_for_display = [REGULATOR] + MANUFACTURERS
# Role: regulator vs manufacturer · single icon per card + restrained accent
role_meta = [
    ("⚖️", "REGULATOR", "监管者", "#3B82F6"),        # NMPA · blue
    ("🏭", "MANUFACTURER A", "影像巨头", "#8B5CF6"),  # purple
    ("🏭", "MANUFACTURER B", "套利者",   "#F59E0B"),  # amber
    ("🏭", "MANUFACTURER C", "独角兽",   "#EC4899"),  # pink
]
for col, agent, (icon, role_en, role_cn, accent) in zip(cols, agents_for_display, role_meta):
    with col:
        st.markdown(
            f"""<div class="dash-card" style="padding:20px 22px; min-height:210px;
                          background:linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
                          position:relative; overflow:hidden;">
            <div style="position:absolute; left:0; top:0; bottom:0; width:4px;
                        background:linear-gradient(180deg, {accent} 0%, {accent}55 100%);"></div>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
                <div style="font-size:32px; line-height:1;
                            filter:drop-shadow(0 4px 8px rgba(15,23,42,0.10));">{icon}</div>
                <div>
                    <div style="font-size:10px; letter-spacing:0.14em; color:{accent};
                                font-weight:800;">{role_en}</div>
                    <div style="font-size:13px; color:#64748B; font-weight:600;
                                margin-top:2px;">{role_cn}</div>
                </div>
            </div>
            <div style="font-size:19px; font-weight:700; color:#0F172A; line-height:1.3;
                        margin-bottom:10px;">{agent.name}</div>
            <div style="font-size:14px; color:#64748B; line-height:1.55;">{agent.profile[:75]}...</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)


def compute_state(round_num, regulator_strictness, mfr_risks):
    """Compute the math state vector for the round.

    Implements simplified version of:
      G_i(t) = guidance issuance strictness (regulator side)
      G_m(t) = compliance effort per manufacturer
      M_m(t) = market adaption (1 - G_m, normalized)
      F_m(t) = G_m - M_m (feedback flowing to regulator)
    """
    import math
    alpha, beta, gamma = 0.3, 0.4, 0.1
    state = {"round": round_num, "G_i": regulator_strictness}
    feedbacks = []
    for i, risk in enumerate(mfr_risks):
        risk_norm = risk / 10.0 if risk is not None else 0.5
        g_m = math.tanh(beta * (regulator_strictness / 10.0) - gamma * risk_norm) + 0.5
        m_m = 1.0 - g_m
        f_m = g_m - m_m
        state[f"G_m{i+1}"] = round(g_m, 3)
        state[f"M_m{i+1}"] = round(m_m, 3)
        state[f"F_m{i+1}"] = round(f_m, 3)
        feedbacks.append(f_m)
    state["F_total"] = round(sum(feedbacks), 3)
    return state


if st.button("▶️ 启动模拟", type="primary", use_container_width=True):
    llm = LLMBackend(model=model)
    state_history = []

    # ============== Round 0: initial state ==============
    state_history.append({"round": 0, "G_i": 5.0, **{f"G_m{i+1}": 0.5 for i in range(3)},
                          **{f"M_m{i+1}": 0.5 for i in range(3)}, **{f"F_m{i+1}": 0.0 for i in range(3)}, "F_total": 0.0})

    # ============== Live state placeholders ==============
    metrics_placeholder = st.empty()
    chart_col, agent_col = st.columns([1, 1])
    chart_placeholder = chart_col.empty()
    text_placeholder = agent_col.empty()

    def render_state(history):
        df = pd.DataFrame(history)
        df_plot = df[["round", "G_i", "G_m1", "G_m2", "G_m3"]].set_index("round")
        with chart_placeholder.container():
            st.markdown("**🧮 状态向量演化（数学反馈方程实时计算）**")
            st.line_chart(df_plot, height=280, color=["#1E40AF", "#059669", "#D97706", "#DC2626"])
            st.caption("G_i = 监管者严格度 · G_m = 厂商合规努力 · 每轮按超越方程更新")

    render_state(state_history)

    # ============== Round 1: Regulator issues ==============
    text_placeholder.info("⚖️ **Round 1 · 指南发布** — NMPA 审评机构 发布征求意见稿...")
    time.sleep(anim_speed)
    reg_text = llm.chat(REGULATOR.profile, SCENARIO["regulator_prompt"])
    REGULATOR.last_action = {"text": reg_text}
    text_placeholder.info(f"⚖️ **Round 1 · 指南发布** · NMPA 发布征求意见稿\n\n{reg_text}")

    state_history.append(compute_state(1, regulator_strictness=7.5, mfr_risks=[5, 5, 5]))
    render_state(state_history)
    metrics_placeholder.markdown(
        f"""<div style="display:flex; gap: 12px;">
        <div style="background:#EFF6FF; padding:8px 12px; border-radius:6px;"><b>G_i</b>: 5.0 → 7.5 ⬆️</div>
        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">Round 1/3</div>
        </div>""", unsafe_allow_html=True
    )
    time.sleep(anim_speed)

    # ============== Round 2: Manufacturers respond ==============
    mfr_responses = []
    mfr_risks = []
    for i, m in enumerate(MANUFACTURERS):
        text_placeholder.info(f"🏭 **Round 2 · 厂商响应** — {m.name} 思考中...")
        prompt = (
            f"刚才 NMPA 发布的指南内容：\n---\n{reg_text}\n---\n\n"
            + SCENARIO["manufacturer_prompt"].format(name=m.name)
        )
        mfr_text = llm.chat(m.profile, prompt)
        risk = parse_risk_score(mfr_text) or 5
        m.last_action = {"text": mfr_text, "risk": risk}
        mfr_responses.append((m, mfr_text, risk))
        mfr_risks.append(risk)
        text_placeholder.info(
            f"🏭 **Round 2 · 厂商响应** — {m.name} · 风险评分 {risk}/10\n\n{mfr_text}"
        )
        state_history.append(compute_state(1 + (i + 1) * 0.3, regulator_strictness=7.5, mfr_risks=mfr_risks + [5] * (3 - len(mfr_risks))))
        render_state(state_history)
        time.sleep(anim_speed)

    # Display all 3 manufacturer cards
    with text_placeholder.container():
        st.markdown("🏭 **Round 2 · 厂商响应** — 三家厂商响应完成")
        for m, txt, risk in mfr_responses:
            risk_dot = "●" if risk <= 3 else ("●" if risk <= 6 else "●")
            risk_color = "#059669" if risk <= 3 else ("#D97706" if risk <= 6 else "#DC2626")
            st.markdown(
                f"<div style='margin:6px 0;'><b>{m.name}</b> · "
                f"<span style='color:{risk_color};'>{risk_dot}</span> {risk}/10</div>",
                unsafe_allow_html=True,
            )
            with st.expander(f"展开 {m.name} 详细响应"):
                st.write(txt)

    state_history.append(compute_state(2, regulator_strictness=7.5, mfr_risks=mfr_risks))
    render_state(state_history)
    metrics_placeholder.markdown(
        f"""<div style="display:flex; gap: 12px;">
        <div style="background:#EFF6FF; padding:8px 12px; border-radius:6px;"><b>G_i</b>: 7.5 (持平)</div>
        <div style="background:#FEF3C7; padding:8px 12px; border-radius:6px;"><b>厂商风险</b>: A={mfr_risks[0]} B={mfr_risks[1]} C={mfr_risks[2]}</div>
        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">Round 2/3</div>
        </div>""", unsafe_allow_html=True
    )
    time.sleep(anim_speed)

    # ============== Round 3: Regulator adjusts ==============
    text_placeholder.info("🔄 **Round 3 · 反馈调整** — NMPA 综合反馈，回应...")
    fb_summary = "\n\n".join(
        f"{m.name} (风险评分 {m.last_action.get('risk','N/A')}):\n{m.last_action['text']}"
        for m in MANUFACTURERS
    )
    prompt = (
        f"三家厂商的反馈如下：\n---\n{fb_summary}\n---\n\n"
        + SCENARIO["regulator_response_prompt"]
    )
    reg_response = llm.chat(REGULATOR.profile, prompt)
    success_prob = parse_risk_score(reg_response) or 6
    text_placeholder.info(f"🔄 **Round 3 · 反馈调整** · NMPA 综合反馈\n\n{reg_response}")

    # Regulator strictness adjusts down based on average risk
    new_strictness = max(5.0, 7.5 - 0.3 * sum(mfr_risks) / len(mfr_risks))
    state_history.append(compute_state(3, regulator_strictness=new_strictness, mfr_risks=mfr_risks))
    render_state(state_history)
    time.sleep(anim_speed)

    # ============== Final Summary ==============
    st.markdown("---")
    st.subheader("📊 模拟收敛结果")

    final_state = state_history[-1]
    summary_cols = st.columns(5)
    summary_cols[0].metric("G_i 最终值", f"{final_state['G_i']:.2f}", f"{final_state['G_i']-5.0:+.2f}")
    summary_cols[1].metric("厂商 A 风险", f"{mfr_risks[0]}/10")
    summary_cols[2].metric("厂商 B 风险", f"{mfr_risks[1]}/10")
    summary_cols[3].metric("厂商 C 风险", f"{mfr_risks[2]}/10")
    summary_cols[4].metric("指南落地概率", f"{success_prob}/10")

    # ============== 剧情解读 · 针对本次具体数字 ==============
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """<div style="border-top:2px solid #0F172A; padding-top:18px;">
            <div style="font-size:13px; letter-spacing:0.16em; color:#475569; font-weight:700;
                        text-transform:uppercase; margin-bottom:6px;">Interpretation</div>
            <div style="font-size:26px; font-weight:700; color:#0F172A; line-height:1.25;
                        margin-bottom:14px;">这次模拟告诉我们什么？</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Per-manufacturer story interpretation
    def mfr_story(idx, risk):
        names = ["影像巨头", "跨域套利者", "AI 原生独角兽"]
        roles = ["A · 合规至上", "B · 监管套利", "C · Move-Fast"]
        if idx == 0:  # A
            if risk <= 3:
                msg = "几乎全盘接受。这套规则对它没有边际成本——反而是它的<b>护城河</b>。"
            elif risk <= 6:
                msg = "有顾虑但不抗拒，PCCP-CN 颗粒度对其内部流程是个挑战。"
            else:
                msg = "罕见地强烈反应——意味着草案颗粒度超出了它一贯能消化的范围。"
        elif idx == 1:  # B
            if risk <= 3:
                msg = "意外配合，可能预示监管套利者也认可该规则。"
            elif risk <= 6:
                msg = "一脚海外。威胁把新品发布转到 HSA / MOHAP，<b>监管套利信号</b>显现。"
            else:
                msg = "强烈反弹，可能直接退出国内注册——值得监管关注。"
        else:  # C
            if risk <= 3:
                msg = "意外低风险，AI 独角兽群体已接受连续学习的合规约束。"
            elif risk <= 6:
                msg = "中度反弹，正在评估 sandbox 通道是否能消化合规成本。"
            else:
                msg = "牵头联名反对。存在<b>产业出海</b>实质风险——草案需要 sandbox 通道兜底。"
        return names[idx], roles[idx], msg

    interp_cols = st.columns(3)
    accents = ["#8B5CF6", "#F59E0B", "#EC4899"]  # match the manufacturer card colors
    for col, idx, accent in zip(interp_cols, range(3), accents):
        name, role, msg = mfr_story(idx, mfr_risks[idx])
        risk_color = "#059669" if mfr_risks[idx] <= 3 else ("#D97706" if mfr_risks[idx] <= 6 else "#DC2626")
        with col:
            st.markdown(
                f"""<div class="dash-card" style="padding:18px 22px; min-height:190px;
                              background:linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
                              position:relative; overflow:hidden;">
                <div style="position:absolute; left:0; top:0; bottom:0; width:4px;
                            background:linear-gradient(180deg, {accent} 0%, {accent}55 100%);"></div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;
                            margin-bottom:10px;">
                    <div style="font-size:11px; letter-spacing:0.14em; color:{accent};
                                font-weight:800;">{role}</div>
                    <div style="font-size:28px; font-weight:800; color:{risk_color};
                                letter-spacing:-0.02em;">{mfr_risks[idx]}<span style="font-size:14px; color:#94A3B8; font-weight:600;">/10</span></div>
                </div>
                <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:10px;">
                    {name}</div>
                <div style="font-size:14px; color:#334155; line-height:1.6;">{msg}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # System-level interpretation
    delta_gi = final_state['G_i'] - 7.5
    f_total = final_state['F_total']
    direction = "<b style='color:#059669;'>放宽</b>" if delta_gi < 0 else ("<b style='color:#DC2626;'>收紧</b>" if delta_gi > 0 else "<b>持平</b>")
    feedback_msg = (
        "合规努力 <b>压住</b> 市场抵抗，系统向合规收敛"
        if f_total > 0.3 else (
        "市场抵抗 <b>压过</b> 合规努力，系统出现漂移"
        if f_total < -0.3 else
        "合规与抵抗 <b>大致平衡</b>，系统处于临界态"
        )
    )
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div style="background:#FAFAFB; border:1px solid #E2E8F0; padding:18px 22px;
                       border-radius:4px;">
            <div style="font-size:11px; letter-spacing:0.14em; color:#475569; font-weight:700;
                        text-transform:uppercase; margin-bottom:10px;">System-Level Read</div>
            <div style="font-size:16px; color:#0F172A; line-height:1.7;">
                监管者把严格度从 <b>7.5</b> 调整到 <b>{final_state['G_i']:.2f}</b>
                （Δ = {delta_gi:+.2f}），方向 {direction}。
                系统总反馈 <b>F<sub>total</sub> = {f_total:+.2f}</b>，{feedback_msg}。
                综合预测的指南落地概率为 <b>{success_prob}/10</b>。
            </div>
            <div style="font-size:14px; color:#475569; line-height:1.6; margin-top:12px;
                        padding-top:12px; border-top:1px dashed #E2E8F0;">
                <b>一句话剧情：</b>同一份草案，对 A 是<b>护城河</b>、对 B 是<b>海外路径</b>、
                对 C 是<b>生死线</b>——这就是合规漂移的可视化。
                把这 3 轮循环换不同 agent profile 跑 N 次，就能在<b>正式发布前</b>
                预测哪些条款会引发反弹、哪个市场段最易出海、增设 sandbox 通道能否提高落地概率。
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ============== Implications · 模拟寓意 ==============
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """<div style="border-top:2px solid #0F172A; padding-top:18px; margin-top:8px;">
            <div style="font-size:13px; letter-spacing:0.16em; color:#475569; font-weight:700;
                        text-transform:uppercase; margin-bottom:8px;">Implications</div>
            <div style="font-size:24px; font-weight:700; color:#0F172A; line-height:1.25;
                        margin-bottom:14px;">模拟跑完之后，意味着什么？</div>
        </div>""",
        unsafe_allow_html=True,
    )

    impl_cols = st.columns(3)
    implications = [
        ("⚖️", "FOR REGULATORS", "对监管侧",
         "**指南发布前的沙盘推演。** 同一份草案，跑 50 次不同 agent 组合，"
         "可以提前看出哪些条款会引发合规漂移、哪个市场段反弹最大，"
         "为正式发布前的版本修订提供量化依据。"),
        ("🏭", "FOR MANUFACTURERS", "对厂商侧",
         "**新规对自家产品线的影响预演。** 把企业自身 profile 灌进 agent，"
         "预测在不同竞争对手响应下，自己应该选保守、跟随、还是激进策略，"
         "把「等监管落地再反应」换成「提前 3-6 个月布局」。"),
        ("🔄", "FOR THE FIELD", "对学科价值",
         "**从规则文本到行为预测的桥梁。** 传统监管研究停在条文解读，"
         "这套框架把「指南 → 厂商行为 → 市场反馈」的闭环写成可计算模型，"
         "支持假设检验、政策试错、审评员训练等下游应用。"),
    ]
    impl_accents = ["#3B82F6", "#10B981", "#6366F1"]
    for col, (icon, en, cn, body), accent in zip(impl_cols, implications, impl_accents):
        with col:
            st.markdown(
                f"""<div class="dash-card" style="padding:18px 22px; min-height:210px;
                              background:linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
                              position:relative; overflow:hidden;">
                <div style="position:absolute; left:0; top:0; bottom:0; width:4px;
                            background:linear-gradient(180deg, {accent} 0%, {accent}55 100%);"></div>
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                    <div style="font-size:28px; line-height:1;
                                filter:drop-shadow(0 4px 8px rgba(15,23,42,0.10));">{icon}</div>
                    <div style="font-size:10px; letter-spacing:0.14em; color:{accent};
                                font-weight:800;">{en}</div>
                </div>
                <div style="font-size:17px; font-weight:700; color:#0F172A; margin-bottom:10px;">
                    {cn}</div>
                <div style="font-size:14px; color:#334155; line-height:1.6;">{body}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.caption(
        "论文应用场景（arXiv:2411.15356）— Pre-publication guideline simulation · "
        "Reviewer training · Post-market behavior monitoring。"
    )

    # Final state table
    with st.expander("📋 完整状态向量历史"):
        st.dataframe(pd.DataFrame(state_history), use_container_width=True)

else:
    st.info("👆 点击「启动模拟」按钮开始。完整模拟约 8-12 秒（带动画）。")
