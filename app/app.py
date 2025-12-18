import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 设置 Matplotlib 支持中文和简洁风格
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. 核心数学模型函数 (保持不变) ---

def generate_confession_times(mode, n=50):
    i_series = np.array(range(1, n + 1))
    if mode == "mo_ceng":      
        return np.array([1 + 1/i for i in i_series])
    elif mode == "sao_dong":   
        return np.array([1 - 1/i for i in i_series])
    else:
        return np.sort(np.random.uniform(0, 10, n))

def is_brave(times):
    if len(times) < 5: return False
    diff = np.abs(np.diff(times[-5:]))
    return np.all(diff < 1e-3)

def success_rate(t, A, t0, sigma):
    if sigma <= 0: sigma = 1e-5
    return A * np.exp(-((t - t0)**2) / (2*sigma**2))

def stability_analysis(t, A_val, t0, sigma, delta=0.01):
    right_limit = success_rate(t + delta, A_val, t0, sigma)
    left_limit = success_rate(t - delta, A_val, t0, sigma)

    if np.isnan(left_limit) or np.isnan(right_limit):
        return "骚操作把自己骚死了 💀"

    is_limit_equal = abs(left_limit - right_limit) < 1e-2

    if is_limit_equal:
        if abs(left_limit - success_rate(t, A_val, t0, sigma)) < 1e-2:
            return "尚在发展 🌱"
        else:
            return "随缘 🍃"
    else:
        return "安排上了 🎁"

def determine_mode(delay_choice, change_choice):
    if delay_choice == 1 and change_choice == 1:
        return "mo_ceng"
    elif delay_choice == 2 or change_choice == 2:
        return "sao_dong"
    else:
        return "random"

# --- 2. 辅助函数：评分与分类 ---

def calculate_score(raw_scores):
    # Streamlit 传入的已经是数值，不再需要复杂的字符串解析
    total_score = sum(raw_scores)
    
    # 映射公式: 总分 3-15 -> 1-10 (保持原逻辑)
    final_score = 1 + ((total_score - 3) / (15 - 3)) * (10 - 1)
    return np.clip(round(final_score), 1, 10) 

def classify_love_type(I, P, C, threshold=7):
    is_i = I >= threshold
    is_p = P >= threshold
    is_c = C >= threshold

    if is_i and is_p and is_c:
        return "圆满的爱 (Consummate Love)", "圆满的爱：这是理想状态，三要素俱全。"
    elif is_i and is_c:
        return "伴侣之爱 (Companionate Love)", "伴侣之爱：稳定深情，但激情可能淡化。"
    elif is_p and is_c:
        return "愚昧的爱 (Fatuous Love)", "愚昧的爱：闪电式结合，缺乏深刻了解的亲密。"
    elif is_i and is_p:
        return "浪漫的爱 (Romantic Love)", "浪漫的爱：深情和激情并存，但缺乏长期承诺。"
    elif is_i:
        return "喜欢 (Liking)", "喜欢：只包含亲密，是真正的友谊。"
    elif is_p:
        return "迷恋 (Infatuation)", "迷恋：只包含激情，一见钟情或单相思。"
    elif is_c:
        return "空洞的爱 (Empty Love)", "空洞的爱：只包含承诺，缺乏情感和吸引力。"
    else:
        return "非爱 (Non-love)", "非爱：三要素均不满足，需从零开始。"

# --- 3. 可视化函数 (适应 Streamlit) ---

@st.cache_data
def plot_love_triangle(I, P, C):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    
    labels = ['亲密 (I)', '激情 (P)', '承诺 (C)']
    values = np.array([I, P, C])
    values = np.concatenate((values, [I]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    plot_color = 'mediumvioletred' 
    fill_color = 'lightpink'
    
    ax.plot(angles, values, 'o-', linewidth=3, color=plot_color, markerfacecolor=plot_color, markersize=8, label="当前关系状态")
    ax.fill(angles, values, color=fill_color, alpha=0.6)

    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontsize=12, color='darkslategray')
    ax.set_ylim(0, 10) 
    ax.set_yticks(np.arange(0, 11, 2)) 
    ax.tick_params(axis='y', colors='gray', labelsize=10)
    ax.spines['polar'].set_visible(False) 
    ax.grid(color='lightgray', linestyle='--')

    love_type, description = classify_love_type(I, P, C)
    ax.text(0, 0, f"类型: {love_type}\n\n{description}", 
            ha='center', va='center', fontsize=11, color=plot_color, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle="round,pad=0.7"))

    ax.set_title("💞 斯滕伯格的爱之三角：关系类型分析", va='bottom', fontsize=16, pad=15, color='darkslategray')
    return fig

@st.cache_data
def plot_success_curve(A, t_peak, sigma, current_time):
    t_start = max(0, min(t_peak, current_time) - 2 * sigma)
    t_end = max(10, max(t_peak, current_time) + 2 * sigma)
    t = np.linspace(t_start, t_end, 300) 
    p = success_rate(t, A, t_peak, sigma)
    p = np.clip(p, 0, 1)
    predicted_rate = success_rate(current_time, A, t_peak, sigma)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.fill_between(t, 0, p, color='skyblue', alpha=0.2, label="成功率区域")
    ax.plot(t, p, color='steelblue', linewidth=3, label="表白成功率 p(t)")
    
    ax.axvline(current_time, color='darkorange', linestyle='-', linewidth=2, label=f"预测行动 T={current_time:.2f} 周")
    ax.scatter(current_time, predicted_rate, s=150, color='darkorange', zorder=5, marker='o', edgecolor='white', linewidth=2)
    
    ax.axvline(t_peak, color='crimson', linestyle='--', linewidth=1.5, label=f"实际最佳 Tpeak={t_peak:.2f} 周")
    ax.axhline(A, color='forestgreen', linestyle=':', label=f"最大成功率 A={A:.2f}", linewidth=1.5)

    ax.annotate(f"预测成功率: {predicted_rate:.2f}", 
                 xy=(current_time, predicted_rate), 
                 xytext=(current_time + 0.5 * sigma, predicted_rate - 0.1),
                 arrowprops=dict(facecolor='darkorange', shrink=0.05, width=1, headwidth=8, headlength=8, alpha=0.7),
                 fontsize=11, color='darkorange')

    ax.set_xlabel("时间 t（周）", fontsize=13)
    ax.set_ylabel("成功率 p(t)", fontsize=13)
    ax.set_title("📈 恋爱时机分析：表白成功率曲线", fontsize=16, pad=15)
    ax.legend(fontsize=10)
    
    return fig

# --- 4. Streamlit 主程序 ---

def run_analysis(data):
    # 提取数据
    q1_delay = data['q1_delay']
    q2_change = data['q2_change']
    raw_i = [data[f'i{i}'] for i in range(1, 4)]
    raw_p = [data[f'p{i}'] for i in range(1, 4)]
    raw_c = [data[f'c{i}'] for i in range(1, 4)]
    t0_ideal = data['t0_weeks']
    
    # 1. 行为模式
    mode = determine_mode(q1_delay, q2_change)
    
    # 2. IPC 评分
    I = calculate_score(raw_i)
    P = calculate_score(raw_p)
    C = calculate_score(raw_c)

    # 3. 计算 A, sigma, t_peak
    A = 0.5 + ((I + P + C) / 30.0) * 0.5 
    sigma = 0.5 + (C / 10.0) * 1.5       
    
    I_norm = I / 10.0
    C_norm = C / 10.0
    alpha = 1.0 - ((I_norm + C_norm) / 2.0) * 0.5
    
    t_peak = t0_ideal * alpha
    t_peak = np.clip(t_peak, 0.01, None) 

    # 4. 计算预测时刻 t
    times = generate_confession_times(mode)
    brave = is_brave(times)
    mean_times_last = np.mean(times[-10:])
    
    if mode == "random":
        current_time_mapped = t_peak + (mean_times_last - np.mean(times)) * (sigma / 4)
    else:
        current_time_mapped = t_peak + (mean_times_last - 1) * (sigma / 2)
    
    current_time_mapped = np.clip(current_time_mapped, 0.01, t_peak + sigma * 3)
    
    # 5. 分析状态
    status = stability_analysis(current_time_mapped, A, t_peak, sigma)
    predicted_rate = success_rate(current_time_mapped, A, t_peak, sigma)
    love_type, _ = classify_love_type(I, P, C)

    # --- 结果展示 ---
    st.markdown("## ✅ **恋爱分析报告**")
    st.markdown(f"### 当前恋爱状态判定：**{status}**")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 关系基础分析 (IPC)")
        st.metric(label="**亲密 (I) 评分**", value=f"{I}/10")
        st.metric(label="**激情 (P) 评分**", value=f"{P}/10")
        st.metric(label="**承诺 (C) 评分**", value=f"{C}/10")
        st.markdown(f"**恋爱类型：** *{love_type}*")
        st.markdown(f"**最大成功率 (A)：** {A:.2f}")

    with col2:
        st.subheader("🧭 时机分析 (T)")
        st.metric(label="**理想锚定时刻 T₀**", value=f"{t0_ideal:.2f} 周后")
        st.metric(label="**🌟 实际最佳时刻 Tpeak**", value=f"{t_peak:.2f} 周后")
        st.metric(label="**预测的行动时刻 T**", value=f"{current_time_mapped:.2f} 周后", delta=f"偏离最佳 {current_time_mapped - t_peak:.2f} 周")
        st.metric(label="**预测成功率 p(T)**", value=f"{predicted_rate:.2f}")
        st.markdown(f"**倾向模式：** {mode}")
        st.markdown(f"**是否勇敢表白：** {'✅ 是' if brave else '❌ 否'}")

    st.markdown("---")
    st.subheader("💞 爱之三角图")
    st.pyplot(plot_love_triangle(I, P, C))

    st.subheader("📈 表白成功率曲线")
    st.pyplot(plot_success_curve(A, t_peak, sigma, current_time_mapped))
    
    st.markdown("---")


def main():
    st.title("💌 恋爱告急·表白分析系统")
    st.markdown("请完成以下问卷，系统将结合您的恋爱关系和行为模式，计算您的最佳表白时机。")

    # 使用 Streamlit 状态管理来保存问卷数据
    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None

    with st.form("love_analysis_form"):
        # --- 1. 行为倾向问卷 ---
        st.subheader("1. 📝 行为倾向问卷")
        q1_delay = st.radio(
            "Q1. 设想表白后，你更倾向于：",
            options=[1, 2],
            format_func=lambda x: "推迟 (1)" if x == 1 else "提前 (2)",
            index=0,
            key='q1_delay'
        )
        q2_change = st.radio(
            "Q2. 你的表白计划是：",
            options=[1, 2],
            format_func=lambda x: "不轻易改变 (1)" if x == 1 else "反复修改 (2)",
            index=0,
            key='q2_change'
        )

        # --- 2. 关系评估问卷 (IPC) ---
        st.subheader("2. 💖 关系评估问卷 (1-5分，5为完全符合)")
        
        ipc_scores = {}
        
        st.markdown("##### [亲密 Intimacy]")
        ipc_scores['i1'] = st.slider("Q3. 我可以向对方分享我最深处的恐惧和秘密。", 1, 5, 3, key='i1')
        ipc_scores['i2'] = st.slider("Q4. 遇到困难时，对方是我的第一选择。", 1, 5, 3, key='i2')
        ipc_scores['i3'] = st.slider("Q5. 我们在一起时，经常能感受到『心有灵犀』的默契。", 1, 5, 3, key='i3')
        
        st.markdown("##### [激情 Passion]")
        ipc_scores['p1'] = st.slider("Q6. 想到或看到对方时，我会有心跳加速和兴奋的感觉。", 1, 5, 3, key='p1')
        ipc_scores['p2'] = st.slider("Q7. 我会努力制造浪漫和惊喜来保持新鲜感。", 1, 5, 3, key='p2')
        ipc_scores['p3'] = st.slider("Q8. 我主动或期望与对方有身体接触或亲密行为。", 1, 5, 3, key='p3')

        st.markdown("##### [承诺 Commitment]")
        ipc_scores['c1'] = st.slider("Q9. 我对这段关系有明确的长期规划（例如：超过一年）。", 1, 5, 3, key='c1')
        ipc_scores['c2'] = st.slider("Q10. 即使我们意见不合，我也会坚持这段关系，而不是轻易放弃。", 1, 5, 3, key='c2')
        ipc_scores['c3'] = st.slider("Q11. 我认为对方是值得我投入时间和精力的『唯一』选择。", 1, 5, 3, key='c3')

        # --- 3. 关键时刻 T₀ 引导 ---
        st.subheader("3. 🧭 关键时刻 T₀ 引导")
        t0_type = st.selectbox(
            "请选择你理想的『关键事件』类型：",
            options=["纪念日/里程碑", "个人事件/节日", "情感高峰期"],
            key='t0_type'
        )
        t0_weeks = st.number_input(
            f"请输入距离该『{t0_type}』事件还有多少**周**？ (例如: 3.5)",
            min_value=0.1,
            value=4.0,
            step=0.1,
            key='t0_weeks'
        )
        
        submitted = st.form_submit_button("🚀 获取我的恋爱分析报告")

    if submitted:
        # 收集所有数据并存入 session_state
        analysis_data = {
            'q1_delay': q1_delay,
            'q2_change': q2_change,
            'i1': ipc_scores['i1'], 'i2': ipc_scores['i2'], 'i3': ipc_scores['i3'],
            'p1': ipc_scores['p1'], 'p2': ipc_scores['p2'], 'p3': ipc_scores['p3'],
            'c1': ipc_scores['c1'], 'c2': ipc_scores['c2'], 'c3': ipc_scores['c3'],
            't0_weeks': t0_weeks,
            't0_type': t0_type
        }
        st.session_state['analysis_data'] = analysis_data
        
    if st.session_state['analysis_data']:
        run_analysis(st.session_state['analysis_data'])

if __name__ == '__main__':
    main()