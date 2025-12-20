# app.py 整合完整版（含数据同意界面 + 深度分析解读）
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 云端无头模式，防止GUI报错
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
import json, pathlib, pandas as pd

# ---------- 0. 云端 Secrets 读取 ----------
def get_gspread_client():
    try:
        creds_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        return gspread.service_account_from_dict(creds_info, scopes)
    except Exception as e:
        st.error(f"❌ 无法连接到 Google Sheets: {e}")
        st.info("💡 请确保已在 Streamlit Secrets 中配置了正确的密钥。")
        return None

SHEET_ID = "1bLDL8ALzc11oU1Ox0Xv0SN9fi3aIRrmcfn4ogUtVPxY"

# ---------- 1. Matplotlib 通用字体 ----------
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ---------- 2. 核心数学模型 ----------
def generate_confession_times(mode, n=50):
    i_series = np.array(range(1, n + 1))
    if mode == "mo_ceng":
        return np.array([1 + 1/i for i in i_series])
    elif mode == "sao_dong":
        return np.array([1 - 1/i for i in i_series])
    else:
        return np.sort(np.random.uniform(0, 10, n))

def is_brave(times):
    if len(times) < 5:
        return False
    diff = np.abs(np.diff(times[-5:]))
    return np.all(diff < 1e-3)

def success_rate(t, A, t0, sigma):
    sigma = max(sigma, 1e-5)
    return A * np.exp(-((t - t0)**2) / (2*sigma**2))

def stability_analysis(t, A_val, t0, sigma, delta=0.01):
    # 计算当前点、左偏移点、右偏移点的成功率
    p_current = success_rate(t, A_val, t0, sigma)
    right_limit = success_rate(t + delta, A_val, t0, sigma)
    left_limit  = success_rate(t - delta, A_val, t0, sigma)
    
    # 1. 极端错误处理
    if np.isnan(p_current):
        return "数据异常 💀"

    # 2. 基础底气判定：如果 A_val (由IPC决定) 太低，基础不牢
    # A_val 范围通常在 0.55 (低IPC) 到 1.0 (高IPC) 之间
    if A_val < 0.65:
        return "现状堪忧 🌪️ (基础薄弱，建议先培养感情)"

    # 3. 趋势判定：当前成功率是在上升还是下降
    is_dropping = right_limit < left_limit  # 过了巅峰期，正在走下坡路

    # 4. 阶梯式状态判定
    # 状态 A：巅峰极高且就在当下
    if p_current > 0.8:
        return "稳操胜券 💍"
    
    # 状态 B：成功率尚可
    if p_current > 0.5:
        if is_dropping:
            return "速战速决 🏃 (成功率开始下滑，抓紧最后时机)"
        else:
            return "安排上了 🎁 (正处于上升期/巅峰期)"
            
    # 状态 C：成功率较低
    if p_current > 0.3:
        if is_dropping:
            return "错失良机 🍂 (最佳时刻已过，建议重新铺垫)"
        else:
            return "尚在发展 🌱 (好感度积累中，表白还需等待)"

    # 5. 默认兜底：成功率极低
    return "静观其变 🍵 (目前胜算较低，不宜贸然出击)"
def determine_mode(delay_choice, change_choice):
    if delay_choice == 1 and change_choice == 1:
        return "mo_ceng"
    elif delay_choice == 2 or change_choice == 2:
        return "sao_dong"
    else:
        return "random"

# ---------- 3. 评分与英文分类 ----------
def calculate_score(raw_scores):
    total_score = sum(raw_scores)
    final_score = 1 + ((total_score - 3) / (15 - 3)) * (10 - 1)
    return np.clip(round(final_score), 1, 10)

def classify_love_type_en(I, P, C, threshold=7):
    is_i = I >= threshold
    is_p = P >= threshold
    is_c = C >= threshold
    if is_i and is_p and is_c:
        return "Consummate Love", "Ideal state: Intimacy, Passion, and Commitment coexist."
    elif is_i and is_c:
        return "Companionate Love", "Deep affection and commitment, but passion may have faded."
    elif is_p and is_c:
        return "Fatuous Love", "Commitment based on passion without deep intimacy."
    elif is_i and is_p:
        return "Romantic Love", "Emotional and physical bond, but lacks long-term commitment."
    elif is_i:
        return "Liking", "Pure intimacy and friendship without intense passion."
    elif is_p:
        return "Infatuation", "Pure passion, often 'love at first sight'."
    elif is_c:
        return "Empty Love", "Commitment remains, but emotional spark is gone."
    else:
        return "Non-love", "Lacks all elements. Casual daily interaction."

# ---------- 4. 可视化函数 ----------
@st.cache_data
def plot_love_triangle(I, P, C):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    labels = ['Intimacy (I)', 'Passion (P)', 'Commitment (C)']
    values = np.array([I, P, C])
    values = np.concatenate((values, [I]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    plot_color = 'mediumvioletred'
    fill_color = 'lightpink'
    ax.plot(angles, values, 'o-', linewidth=3, color=plot_color,
            markerfacecolor=plot_color, markersize=8, label="Relationship Status")
    ax.fill(angles, values, color=fill_color, alpha=0.6)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels,
                      fontsize=11, color='darkslategray')
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 11, 2))
    ax.tick_params(axis='y', colors='gray', labelsize=10)
    ax.spines['polar'].set_visible(False)
    ax.grid(color='lightgray', linestyle='--')
    love_type_en, desc_en = classify_love_type_en(I, P, C)
    ax.text(0, 0, f"Type: {love_type_en}\n\n{desc_en}",
            ha='center', va='center', fontsize=10, color=plot_color, wrap=True,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle="round,pad=0.7"))
    ax.set_title("💞 Sternberg's Triangular Theory of Love",
                 va='bottom', fontsize=15, pad=20, color='darkslategray')
    return fig

@st.cache_data
def plot_success_curve(A, t_peak, sigma, current_time):
    t_start = max(0, min(t_peak, current_time) - 2 * sigma)
    t_end   = max(10, max(t_peak, current_time) + 2 * sigma)
    t       = np.linspace(t_start, t_end, 300)
    p       = success_rate(t, A, t_peak, sigma)
    p       = np.clip(p, 0, 1)
    predicted_rate = success_rate(current_time, A, t_peak, sigma)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.fill_between(t, 0, p, color='skyblue', alpha=0.2, label="Success Zone")
    ax.plot(t, p, color='steelblue', linewidth=3, label="Success Rate p(t)")

    ax.axvline(current_time, color='darkorange', linestyle='-', linewidth=2,
               label=f"Predicted Action (T={current_time:.2f}w)")
    ax.scatter(current_time, predicted_rate, s=150, color='darkorange',
               zorder=5, marker='o', edgecolor='white', linewidth=2)

    ax.axvline(t_peak, color='crimson', linestyle='--', linewidth=1.5,
               label=f"Ideal Peak (Tpeak={t_peak:.2f}w)")
    ax.axhline(A, color='forestgreen', linestyle=':',
               label=f"Max Rate (A={A:.2f})", linewidth=1.5)

    ax.annotate(f"Rate: {predicted_rate:.2f}",
                xy=(current_time, predicted_rate),
                xytext=(current_time + 0.5 * sigma, predicted_rate - 0.1),
                arrowprops=dict(facecolor='darkorange', shrink=0.05,
                                width=1, headwidth=8, headlength=8, alpha=0.7),
                fontsize=11, color='darkorange')

    ax.set_xlabel("Time t (Weeks)", fontsize=12)
    ax.set_ylabel("Probability p(t)", fontsize=12)
    ax.set_title("📈 Confession Timing & Success Rate Analysis",
                 fontsize=15, pad=15)
    ax.legend(fontsize=9, loc='upper right')

    return fig

# ---------- 5. 主分析函数 (含人格判断与详细解读) ----------
def run_analysis(data):
    # 基础数据提取
    q1_delay = data['q1_delay']
    q2_change = data['q2_change']
    raw_i = [data[f'i{i}'] for i in range(1, 4)]
    raw_p = [data[f'p{i}'] for i in range(1, 4)]
    raw_c = [data[f'c{i}'] for i in range(1, 4)]
    t0_ideal = data['t0_weeks']
    # 新增字段
    is_westlake = data['is_westlake']
    will_confess = data['will_confess']

    # 模型计算
    mode = determine_mode(q1_delay, q2_change)
    I = calculate_score(raw_i)
    P = calculate_score(raw_p)
    C = calculate_score(raw_c)

    A = 0.5 + ((I + P + C) / 30.0) * 0.5
    sigma = 0.5 + (C / 10.0) * 1.5

    I_norm = I / 10.0
    C_norm = C / 10.0
    alpha = 1.0 - ((I_norm + C_norm) / 2.0) * 0.5
    t_peak = t0_ideal * alpha
    t_peak = np.clip(t_peak, 0.01, None)

    times = generate_confession_times(mode)
    mean_times_last = np.mean(times[-10:])

    if mode == "random":
        current_time_mapped = t_peak + (mean_times_last - np.mean(times)) * (sigma / 4)
    else:
        current_time_mapped = t_peak + (mean_times_last - 1) * (sigma / 2)

    current_time_mapped = np.clip(current_time_mapped, 0.01, t_peak + sigma * 3)

    status = stability_analysis(current_time_mapped, A, t_peak, sigma)
    predicted_rate = success_rate(current_time_mapped, A, t_peak, sigma)

    # --- 写入 Google Sheets 逻辑 ---
    gc = get_gspread_client()
    if gc:
        try:
            sheet = gc.open_by_key(SHEET_ID).sheet1
            
            # 统一强制转换为原生 Python 类型
            row = [
                str(pd.Timestamp('now')), 
                int(q1_delay), 
                int(q2_change),
                *[int(x) for x in raw_i],
                *[int(x) for x in raw_p],
                *[int(x) for x in raw_c],
                float(t0_ideal),
                int(I), 
                int(P), 
                int(C), 
                round(float(t_peak), 2), 
                round(float(current_time_mapped), 2),
                round(float(predicted_rate), 2), 
                str(status),
                str(is_westlake),  # 第15列
                str(will_confess)  # 第16列
            ]
            sheet.append_row(row)
            st.success("✅ 数据已同步至云端表格")
        except Exception as e:
            st.warning(f"⚠️ 未能写入表格：{e}")

    # --- 前端展示部分 (大幅增强) ---
    st.markdown("## ✅ **恋爱分析报告**")
    st.markdown(f"### 当前恋爱状态判定：**{status}**")
    
    # 1. 人格类型分析
    mode_map = {
        "mo_ceng": "🐢 磨蹭型 (Hesitant) - 倾向于等待完美时机，但也可能错失良机。",
        "sao_dong": "🐇 骚动型 (Restless) - 行动果断，内心躁动，倾向于快速推进。",
        "random": "🎲 随缘型 (Spontaneous) - 行为难以预测，跟随感觉走。"
    }
    user_personality = mode_map.get(mode, "未知类型")
    st.info(f"🎭 **您的行动人格分析：{user_personality}**")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 关系基础分析 (IPC)")
        st.metric(label="亲密 (I) 评分", value=f"{I}/10", help="情感的温暖与亲近程度")
        st.metric(label="激情 (P) 评分", value=f"{P}/10", help="浪漫、身体吸引与冲动")
        st.metric(label="承诺 (C) 评分", value=f"{C}/10", help="维持关系的决定与责任感")

    with col2:
        st.subheader("🧭 时机分析 (T)")
        st.metric(label="🌟 理论最佳时刻 T_peak", value=f"{t_peak:.2f} 周后", help="模型计算出的成功率最高点")
        st.metric(label="🚀 预测行动时刻 T", value=f"{current_time_mapped:.2f} 周后",
                  delta=f"{current_time_mapped - t_peak:.2f} 周偏差", help="结合您的人格计算出的实际行动时间")
        st.metric(label="🎯 预测成功率 p(T)", value=f"{(predicted_rate*100):.1f}%")

    # --- 图表 1: 斯滕伯格三角 ---
    st.markdown("---")
    st.subheader("1️⃣ 爱之三角图 (Triangular Analysis)")
    st.pyplot(plot_love_triangle(I, P, C))
    
    # 图表解读 1
    st.markdown("""
    #### 💡 三角图解读：
    * **亲密 (I)**、**激情 (P)**、**承诺 (C)** 构成了三角形的三个顶点。
    * **均衡性**：三角形越接近正三角形，关系越平衡。
    * **面积**：三角形面积越大，代表爱的总量越丰富。
    """)
    if I < 4 and P < 4 and C < 4:
        st.warning("⚠️ **分析**：目前三项指标均较低，建议在行动前先增加日常互动，培养基础感情。")
    elif I >= 7 and P >= 7 and C >= 7:
        st.success("🎉 **分析**：恭喜！你们处于极其理想的『完美之爱』状态，基础非常牢固。")
    else:
        max_attr = max(I, P, C)
        if max_attr == I:
            st.info("ℹ️ **分析**：你们的关系以**亲密感**为主导，像知心好友般舒适，但可能需要更多激情的火花。")
        elif max_attr == P:
            st.info("ℹ️ **分析**：**激情**是你们关系的主要驱动力，吸引力很强，但需注意培养长期的稳定性。")
        elif max_attr == C:
            st.info("ℹ️ **分析**：**承诺**是当前的强项，关系很稳定，但可能稍显平淡，建议增加一些浪漫活动。")

    # --- 图表 2: 成功率曲线 ---
    st.subheader("2️⃣ 表白成功率曲线 (Success Probability Curve)")
    st.pyplot(plot_success_curve(A, t_peak, sigma, current_time_mapped))
    
    # 图表解读 2
    st.markdown("""
    #### 💡 曲线图解读：
    * **蓝色曲线**：代表随时间推移，表白成功率的变化趋势。
    * **红色虚线 (Ideal Peak)**：理论上的最高成功率时刻。
    * **橙色实线 (Predicted Action)**：系统预测你会采取行动的时刻。
    * **橙色点位置**：如果你在预测时间点行动，对应的成功率高度。
    """)
    
    # 时机建议逻辑
    delta_t = current_time_mapped - t_peak
    st.write(f"**数据明细**：理想时刻 `{t_peak:.2f}周` vs 实际行动 `{current_time_mapped:.2f}周`")
    
    if abs(delta_t) < 0.5:
        st.success("✅ **时机评价：精准！** 您的行动节奏与最佳时机高度重合，这是最好的信号。")
    elif delta_t < -0.5:
        st.warning("⚡ **时机评价：操之过急**。您可能比最佳时机行动得更早。虽然热情可嘉，但略显冒进，建议稍微沉住气，多做铺垫。")
    else:
        st.warning("🐢 **时机评价：稍显拖沓**。您可能在最佳时机之后才行动。犹豫可能会让热情冷却，建议加快节奏！")
    
    # --- 最终寄语 ---
    st.markdown("---")
    if will_confess == "是":
        st.success("### 🚀 系统最终建议：停止迭代幻想，开启一场真实的对话！")
    else:
        st.info("### 🍃 系统最终建议：花若盛开，蝴蝶自来。相信那个人在未来等你。")

# ---------- 6. Streamlit UI ----------
# ---------- 6. Streamlit UI (修复重复 Key 报错版) ----------
def main():
    st.set_page_config(page_title="恋爱分析系统", page_icon="💌", layout="centered")
    st.title("💌 恋爱告急·表白分析系统")

    # 初始化状态变量
    if 'data_consent' not in st.session_state:
        st.session_state['data_consent'] = False
    if 'final_confirmed' not in st.session_state:
        st.session_state['final_confirmed'] = False
    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None

    # 1. 数据授权阶段
    if not st.session_state['data_consent']:
        st.info("### 📝 数据授权告知")
        st.markdown("""
        欢迎使用本分析系统。在开始前，请阅读以下说明：
        1. **匿名收集**：系统会匿名收集数据以优化模型。
        2. **隐私保护**：不收集个人身份信息。
        3. **同步机制**：点击同意后数据同步至云端。
        """)
        if st.button("✅ 我同意并开始分析", use_container_width=True):
            st.session_state['data_consent'] = True
            st.rerun()
        return 

    # 2. 问卷与确认阶段
    if not st.session_state['final_confirmed']:
        
        # A. 填写表单：仅在没有暂存数据时显示
        if st.session_state['analysis_data'] is None:
            st.markdown("请完成以下问卷，系统将通过**斯滕伯格爱情理论**计算您的最佳表白时机。")
            
            with st.form("love_analysis_form"):
                st.subheader("0. 🏫 基本身份与意愿")
                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    is_westlake = st.radio("你是否为西湖大学学生？", options=["是", "否"], horizontal=True)
                with col_q2:
                    will_confess = st.radio("你是否有表白意愿？", options=["是", "否"], horizontal=True)
                
                st.markdown("---")
                st.subheader("1. 📝 行为倾向问卷")
                q1_delay = st.radio("Q1. 设想表白后，你更倾向于：", options=[1, 2],
                                    format_func=lambda x: "推迟/犹豫 (1)" if x == 1 else "果断行动 (2)")
                q2_change = st.radio("Q2. 你的表白计划是：", options=[1, 2],
                                    format_func=lambda x: "稳扎稳打 (1)" if x == 1 else "灵活变通 (2)")

                st.subheader("2. 💖 关系评估问卷 (1-5分)")
                ipc_scores = {}
                st.markdown("##### [亲密 Intimacy]")
                ipc_scores['i1'] = st.slider("Q3. 我可以向对方分享我最深处的恐惧和秘密。", 1, 5, 3)
                ipc_scores['i2'] = st.slider("Q4. 遇到困难时，对方是我的第一选择。", 1, 5, 3)
                ipc_scores['i3'] = st.slider("Q5. 我们在一起时，经常能感受到『心有灵犀』的默契。", 1, 5, 3)

                st.markdown("##### [激情 Passion]")
                ipc_scores['p1'] = st.slider("Q6. 想到或看到对方时，我会有心跳加速和兴奋的感觉。", 1, 5, 3)
                ipc_scores['p2'] = st.slider("Q7. 我会努力制造浪漫和惊喜来保持新鲜感。", 1, 5, 3)
                ipc_scores['p3'] = st.slider("Q8. 我主动或期望与对方有身体接触或亲密行为。", 1, 5, 3)

                st.markdown("##### [承诺 Commitment]")
                ipc_scores['c1'] = st.slider("Q9. 我对这段关系有明确的长期规划（例如：超过一年）。", 1, 5, 3)
                ipc_scores['c2'] = st.slider("Q10. 即使我们意见不合，我也会坚持这段关系，而不是轻易放弃。", 1, 5, 3)
                ipc_scores['c3'] = st.slider("Q11. 我认为对方是值得我投入时间和精力的『唯一』选择。", 1, 5, 3)

                st.subheader("3. 🧭 关键时刻 T₀ 引导")
                t0_weeks = st.number_input(f"请输入距离该理想事件还有多少周？", min_value=0.1, value=1.0, step=0.1)
                
                submitted = st.form_submit_button("🚀 获取我的恋爱分析报告")
                
                if submitted:
                    st.session_state['analysis_data'] = {
                        'q1_delay': q1_delay, 'q2_change': q2_change,
                        'is_westlake': is_westlake, 'will_confess': will_confess,
                        **ipc_scores, 't0_weeks': t0_weeks
                    }
                    st.rerun()

        # B. 确认真实性：在表单提交后，生成报告前显示
        else:
            st.warning("### 🧐 真实性确认 / Final Verification")
            st.info("💡 **“以上问卷所填写的每一项数据，都是我内心最真实的想法。”**")
            
            c_left, c_right = st.columns(2)
            with c_left:
                if st.button("✨ 是的，这是真实想法", use_container_width=True):
                    st.session_state['final_confirmed'] = True
                    st.rerun()
            with c_right:
                if st.button("⬅️ 返回修改数据", use_container_width=True):
                    st.session_state['analysis_data'] = None
                    st.rerun()

    # 3. 报告展示阶段
    else:
        run_analysis(st.session_state['analysis_data'])
        
        st.markdown("---")
        if st.button("🔄 重新进行测试", use_container_width=True):
            st.session_state['analysis_data'] = None
            st.session_state['final_confirmed'] = False
            st.rerun()

if __name__ == '__main__':
    main()

