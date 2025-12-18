# app.py  (完整可运行版，已集成 Google-Sheets 自动收集)
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
import json
import toml   # pip install toml

# ---------- 0. Google Sheets 配置 ----------
# ① 把 TOML 文件里 [[gc]] 的内容原样粘到下面三引号之间
GOOGLE_SHEETS_CREDENTIALS = """
{
  "type": "service_account",
  "project_id": "love-emergency",
  "private_key_id": "e58ff450cdd028ba5745ea62d93fda233ff203aa",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC8+E88zDIHbO8s\niWBwAMrCtJjxonHUNfHuP3INYxZPuRz2SS97629ey7K4mCPBB6Cdo/Z5BINYZq56\nwN+F/uL5WFcEJasxNBktwopqHREO6DoQXR+OocWUGoqp1atdrDP33Nf2AKZXpB2N\nTavW7I2/OPKvKoYIkI2QZoRJHTYlz9i+OeHDukl8e/1HgDtw5B9FwA4t4oiySuYA\n9Z51+8fpgY2ybuGbefZtI8eN++FUch6U8sPdykA0Dpr95Kb20QCplTFc19lM1L0h\nfqB+H+7QJHlsKQx1une6sJt0XyvV7VLPe4I9gfvw8d2lFtBTUlW2J2Q+325Z9AUa\npL6hmDtFAgMBAAECggEAHiD0+TuRSm/K3m30y0bFDTAgJn6A6ZXEQfkppQrEVuer\nv3TBtl8+PX36u8W+BZvCtv+aX4chremJLhcsTD/sTlcQYJ/k4I5u6UXYLbz+qELM\nZymBy4rtZoSo0RU5IlE/Y+h5IkbOPrDy6UGWAUlr/C4HO3hrMFhjyb8enk2jAXoQ\n8GUj/oypq1Rttd4MOD7zKm7TM4e3lkl0vkUrq165l6ssIzePAbqdN+Fd162SHLEC\nUc/TTR6WnBgsXqacpXnyTkFedEbbW3tNLmd2TiadutCCZTuk/vQgzHBBnkRnW7Nk\nOLM0Mw13TRFli9eFxFIMKHOsLjBXfojVFEAuuirngQKBgQDdZhkdJepYQuX07Y41\n/sqzOUb7NRDcpwtIsq2o0AVKrIEnZ1jd8vYIx/2xCIMCDszxQ2p2LtzDV1y4548y\nX1Ev0ylT0y2al3Ua3g+F8fFG2p7nBy21izR647Uty3C4yQeapz0hnrFXzuKnxNII\nDPHOM129KfC8qsl1ks7hIeRy4QKBgQDagMV/BQn6MS4gmVtxQ0aXAyOBNolSIQck\nL0pvf3Bzbbzbo3b6yECHvNm0M7SoPR3D+RvKVyWl9M7hNaUrxpkhKwDOxeDW/e0h\nG1xeTKfigN8jdwmLO7zZX9yt9Z2iHLr+TF7kLcfJujYme/bVQ4Pe/OyS+H/pvkAO\nfMFJKuR45QKBgQC/SeeI96lyeNqWtGma3XnlQCfEBCV9gBaPyVGh+ZmY21L76J8v\npSxOifz3aJNIw+Du04C4e+TiIilK2UcwDorm91tNwbg1SYc0n4hqApCk119T3S/x\nG0VMqFFyL8RE4+xeAwEeey5e37GVosiVjBmgP2FOf14wpJ9Lpnx4p//qAQKBgDp4\n72EYdh0QACoVIBVlTYSoAF5Zu9HQqNqUFTVVQ0CAg2O7kOF3qV0pupCwrY3AHTEO\nftNdEuQgaSR3eKYIVX48xdCPv6WI+mY7rjJGDT9eAVi6SEGMUPNS5flfmzmAusHG\nqjYh0i83t7oAvoM/uBB6WszR11kz4mx+EjOEWPPJAoGBAI1H4H/o86K0Msv3EuB0\n+fA7/XhBNMrC0FS4bEMkjN/hv/CbFyQdWHUaeuA8nqIM1BJPJIhXJFqV5JLK1/IJ\nG0Y8DW7i02lRNqy2GVAGll4S5yApGvi/mzA0FEvPXiKNzNwHeqYchbBofhsIN2jr\njWAn8GNYVXVVgQoZIiZp7hXN\n-----END PRIVATE KEY-----\n",
  "client_email": "wlt-607@love-emergency.iam.gserviceaccount.com",
  "client_id": "112209856028945305453",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/wlt-607%40love-emergency.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
"""
# ② 打开目标表格，把 URL 里 /d/ 与 /edit 之间的那段 ID 粘过来
SHEET_ID = "spreadsheets/d/1bLDL8ALzc11oU1Ox0Xv0SN9fi3aIRrmcfn4ogUtVPxY"

# 建立 gspread 连接（全局只连一次）
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
gc = gspread.service_account_from_dict(json.loads(GOOGLE_SHEETS_CREDENTIALS), scopes)
sheet = gc.open_by_key(SHEET_ID).sheet1   # 默认用第 1 个工作表

# ---------- 1. Matplotlib 配置 ----------
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ---------- 2. 核心数学模型函数（与之前一致，省略重复注释） ----------
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
    if sigma <= 0:
        sigma = 1e-5
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

# ---------- 4. 可视化函数（与之前一致，省略） ----------
@st.cache_data
def plot_love_triangle(I, P, C):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    labels = ['Intimacy (I)', 'Passion (P)', 'Commitment (C)']
    values = np.array([I, P, C])
    values = np.concatenate((values, [I]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    plot_color, fill_color = 'mediumvioletred', 'lightpink'
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
    t_end = max(10, max(t_peak, current_time) + 2 * sigma)
    t = np.linspace(t_start, t_end, 300)
    p = success_rate(t, A, t_peak, sigma)
    p = np.clip(p, 0, 1)
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

# ---------- 5. 分析主函数 ----------
def run_analysis(data):
    q1_delay = data['q1_delay']
    q2_change = data['q2_change']
    raw_i = [data[f'i{i}'] for i in range(1, 4)]
    raw_p = [data[f'p{i}'] for i in range(1, 4)]
    raw_c = [data[f'c{i}'] for i in range(1, 4)]
    t0_ideal = data['t0_weeks']

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
    brave = is_brave(times)
    mean_times_last = np.mean(times[-10:])
    if mode == "random":
        current_time_mapped = t_peak + (mean_times_last - np.mean(times)) * (sigma / 4)
    else:
        current_time_mapped = t_peak + (mean_times_last - 1) * (sigma / 2)
    current_time_mapped = np.clip(current_time_mapped, 0.01, t_peak + sigma * 3)

    status = stability_analysis(current_time_mapped, A, t_peak, sigma)
    predicted_rate = success_rate(current_time_mapped, A, t_peak, sigma)

    # ---- 6. 追加到 Google Sheets ----
    # 把要保存的字段 flatten 成一行
    row = [
        str(st.session_state.get('submit_time', '')),  # 提交时间
        q1_delay,
        q2_change,
        *raw_i,   # i1 i2 i3
        *raw_p,   # p1 p2 p3
        *raw_c,   # c1 c2 c3
        t0_ideal,
        I, P, C,
        round(t_peak, 2),
        round(current_time_mapped, 2),
        round(predicted_rate, 2),
        status
    ]
    sheet.append_row(row)

    # ---------- 7. 前端展示 ----------
    st.markdown("## ✅ **恋爱分析报告**")
    st.markdown(f"### 当前恋爱状态判定：**{status}**")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 关系基础分析 (IPC)")
        st.metric(label="亲密 (I) 评分", value=f"{I}/10")
        st.metric(label="激情 (P) 评分", value=f"{P}/10")
        st.metric(label="承诺 (C) 评分", value=f"{C}/10")
    with col2:
        st.subheader("🧭 时机分析 (T)")
        st.metric(label="🌟 实际最佳时刻 Tpeak", value=f"{t_peak:.2f} 周后")
        st.metric(label="预测的行动时刻 T", value=f"{current_time_mapped:.2f} 周后",
                  delta=f"{current_time_mapped - t_peak:.2f} 偏差")
        st.metric(label="预测成功率 p(T)", value=f"{predicted_rate:.2f}")
    st.markdown("---")
    st.subheader("💞 爱之三角图 (Triangular Analysis)")
    st.pyplot(plot_love_triangle(I, P, C))
    st.subheader("📈 表白成功率曲线 (Success Probability Curve)")
    st.pyplot(plot_success_curve(A, t_peak, sigma, current_time_mapped))

# ---------- 8. Streamlit UI ----------
def main():
    st.title("💌 恋爱告急·表白分析系统")
    st.markdown("请完成以下问卷，系统将通过**斯滕伯格爱情理论**计算您的最佳表白时机。")
    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None

    with st.form("love_analysis_form"):
        st.subheader("1. 📝 行为倾向问卷")
        q1_delay = st.radio("Q1. 设想表白后，你更倾向于：", options=[1, 2],
                            format_func=lambda x: "推迟/犹豫 (1)" if x == 1 else "果断行动 (2)")
        q2_change = st.radio("Q2. 你的表白计划是：", options=[1, 2],
                            format_func=lambda x: "稳扎稳打 (1)" if x == 1 else "灵活变通 (2)")
        st.subheader("2. 💖 关系评估问卷 (1-5分)")
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
        st.subheader("3. 🧭 关键时刻 T₀ 引导")
        t0_weeks = st.number_input("距离下一个重要节点（如节日、纪念日）还有几周？", min_value=0.1, value=4.0)
        submitted = st.form_submit_button("🚀 获取我的恋爱分析报告")

    if submitted:
        st.session_state['submit_time'] = str(pd.Timestamp('now'))  # 记录提交时间
        analysis_data = {
            'q1_delay': q1_delay,
            'q2_change': q2_change,
            **ipc_scores,
            't0_weeks': t0_weeks
        }
        st.session_state['analysis_data'] = analysis_data

    if st.session_state['analysis_data']:
        run_analysis(st.session_state['analysis_data'])

if __name__ == '__main__':
    import pandas as pd  # 仅为了生成提交时间戳
    main()



