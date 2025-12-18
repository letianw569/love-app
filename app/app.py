# app.py  无emoji方格完整版
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
import json, pathlib, pandas as pd

# ---------- 0. 云端Secrets读取 ----------
def get_gspread_client():
    try:
        creds_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        return gspread.service_account_from_dict(creds_info, scopes)
    except Exception as e:
        st.error(f"无法连接到Google Sheets: {e}")
        st.info("请确保已在Streamlit Secrets中配置正确密钥。")
        return None

SHEET_ID = "1qRsD5Z2LxM0QYrVKL8g_6ZxyAj5VQYDXxR2oVwKoB7I"

# ---------- 1. Matplotlib字体 ----------
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ---------- 2. 核心模型 ----------
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
    right_limit = success_rate(t + delta, A_val, t0, sigma)
    left_limit  = success_rate(t - delta, A_val, t0, sigma)
    if np.isnan(left_limit) or np.isnan(right_limit):
        return "异常状态"
    is_limit_equal = abs(left_limit - right_limit) < 1e-2
    if is_limit_equal:
        return "尚在发展" if abs(left_limit - success_rate(t, A_val, t0, sigma)) < 1e-2 else "随缘"
    return "安排上了"

def determine_mode(delay_choice, change_choice):
    if delay_choice == 1 and change_choice == 1:
        return "mo_ceng"
    elif delay_choice == 2 or change_choice == 2:
        return "sao_dong"
    else:
        return "random"

def calculate_score(raw_scores):
    total = sum(raw_scores)
    final = 1 + ((total - 3) / (15 - 3)) * (10 - 1)
    return np.clip(round(final), 1, 10)

def classify_love_type_en(I, P, C, threshold=7):
    is_i, is_p, is_c = I >= threshold, P >= threshold, C >= threshold
    if is_i and is_p and is_c:
        return "Consummate Love", "完美爱情：亲密、激情与承诺并存。"
    if is_i and is_c:
        return "Companionate Love", "伴侣之爱：深厚的友谊与承诺，但缺乏激情。"
    if is_p and is_c:
        return "Fatuous Love", "愚蠢之爱：仅靠激情支撑的承诺。"
    if is_i and is_p:
        return "Romantic Love", "浪漫之爱：情感与身体的联结，缺乏长期规划。"
    if is_i:
        return "Liking", "喜爱：纯粹的友谊。"
    if is_p:
        return "Infatuation", "迷恋：迷恋对方的外在或某种特质。"
    if is_c:
        return "Empty Love", "空洞之爱：徒留名义上的承诺。"
    return "Non-love", "无爱：日常的普通社交。"

# ---------- 3. 可视化（无emoji） ----------
@st.cache_data
def plot_love_triangle(I, P, C):
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    labels = ['亲密 (I)', '激情 (P)', '承诺 (C)']
    values = np.array([I, P, C])
    values = np.concatenate((values, [I]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # 每轴彩色渐变 + 数值标签
    axis_colors = ['#4B92DB', '#FF6B6B', '#4ECB71']
    for ang, val, color in zip(angles[:-1], values[:-1], axis_colors):
        ax.bar(ang, val, width=2*np.pi/3, color=color, alpha=0.65, edgecolor=color, lw=2)
        ax.text(ang, val+0.3, f'{val}', color=color, fontsize=12, ha='center', weight='bold')

    ax.plot(angles, values, 'o-', color='darkslategray', lw=3, markersize=9)
    ax.fill(angles, values, alpha=0.15, color='gray')
    ax.set_thetagrids(angles[:-1]*180/np.pi, labels, fontsize=13)
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 11, 2))
    ax.tick_params(axis='y', colors='gray', labelsize=10)
    ax.spines['polar'].set_visible(False)
    ax.grid(color='lightgray', linestyle='--', alpha=0.8)

    love_type, desc = classify_love_type_en(I, P, C)
    ax.set_title(f'Sternberg 爱情三角理论\n{love_type}', pad=25, fontsize=14, color='darkslategray')
    return fig


@st.cache_data
def plot_success_curve(A, t_peak, sigma, current_time):
    t_start = max(0, min(t_peak, current_time) - 2 * sigma)
    t_end   = max(15, max(t_peak, current_time) + 2 * sigma)
    t       = np.linspace(t_start, t_end, 400)
    p       = success_rate(t, A, t_peak, sigma)
    p       = np.clip(p, 0, 1)
    predicted_rate = success_rate(current_time, A, t_peak, sigma)

    fig, ax = plt.subplots(figsize=(9, 5))

    # 1. 成功概率段高亮
    ax.fill_between(t, 0, p, color='skyblue', alpha=0.25, label='成功概率区间')
    # 2. 主线
    ax.plot(t, p, color='steelblue', linewidth=3, label='成功率曲线 p(t)')

    # 3. 三线标注
    ax.axvline(current_time, color='darkorange', ls='-', lw=2.5,
               label=f'预测行动点  T={current_time:.2f}周')
    ax.scatter(current_time, predicted_rate, s=160, color='darkorange',
               zorder=6, marker='o', edgecolors='white', linewidths=2)

    ax.axvline(t_peak, color='crimson', ls='--', lw=2,
               label=f'理论最佳点  Tpeak={t_peak:.2f}周')
    ax.axhline(A, color='forestgreen', ls=':', lw=2,
               label=f'峰值成功率  A={A:.2f}')

    # 4. 箭头注解
    ax.annotate(f'当前成功率\n{predicted_rate:.2f}',
                xy=(current_time, predicted_rate),
                xytext=(current_time + 0.8 * sigma, predicted_rate + 0.15),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5),
                fontsize=11, color='darkorange', ha='center')

    # 5. 中文轴标签 & 标题
    ax.set_xlabel('时间 (周)', fontsize=13)
    ax.set_ylabel('成功概率', fontsize=13)
    ax.set_title('表白时机成功率分析', fontsize=15, pad=15)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(color='lightgray', linestyle='--', alpha=0.6)

    return fig


# ---------- 4. 分析 & 写入 ----------
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
    t_peak = np.clip(t0_ideal * (1.0 - ((I/10.0 + C/10.0)/2.0)*0.5), 0.1, None)

    times = generate_confession_times(mode)
    mean_times_last = np.mean(times[-10:])
    current_time_mapped = np.clip(t_peak + (mean_times_last - 1) * (sigma / 2), 0.1, 15)

    status = stability_analysis(current_time_mapped, A, t_peak, sigma)
    predicted_rate = success_rate(current_time_mapped, A, t_peak, sigma)

    # 写入Google Sheets
    gc = get_gspread_client()
    if gc:
        try:
            sheet = gc.open_by_key(SHEET_ID).sheet1
            row = [str(pd.Timestamp('now')), q1_delay, q2_change,
                   *raw_i, *raw_p, *raw_c, t0_ideal,
                   I, P, C, round(t_peak, 2), round(current_time_mapped, 2),
                   round(predicted_rate, 2), status]
            sheet.append_row(row)
            st.success("数据已同步至云端表格")
        except Exception as e:
            st.warning(f"未能写入表格：{e}")

    # 前端展示
    st.markdown("## 恋爱分析报告")
    st.markdown(f"**当前状态：{status}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("亲密 I", I)
    col2.metric("激情 P", P)
    col3.metric("承诺 C", C)
    st.pyplot(plot_love_triangle(I, P, C))
    st.pyplot(plot_success_curve(A, t_peak, sigma, current_time_mapped))


# ---------- 5. 主界面 ----------
def main():
    st.title('恋爱告急 · 表白分析系统')
    st.markdown('请完成以下问卷，系统将通过斯滕伯格爱情理论计算您的最佳表白时机。')

    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None

    with st.form('love_form'):
        st.subheader('1. 行为倾向问卷')
        q1 = st.radio('Q1. 设想表白后，你更倾向于：', [1, 2],
                     format_func=lambda x: '推迟/犹豫 (1)' if x == 1 else '果断行动 (2)')
        q2 = st.radio('Q2. 你的表白计划是：', [1, 2],
                     format_func=lambda x: '稳扎稳打 (1)' if x == 1 else '灵活变通 (2)')

        st.subheader('2. 关系评估问卷 (1-5分)')
        scores = {}
        for cat, label in [('i', '亲密'), ('p', '激情'), ('c', '承诺')]:
            for i in range(1, 4):
                scores[f'{cat}{i}'] = st.slider(f'{label}指标 {i}', 1, 5, 3, key=f'{cat}{i}')

        st.subheader('3. 关键时刻 T₀ 引导')
        t0_type = st.selectbox('请选择你理想的『关键事件』类型：',
                              ['纪念日/里程碑', '个人事件/节日', '情感高峰期'], key='t0_type')
        t0_weeks = st.number_input(f'距离该『{t0_type}』事件还有多少周？', 0.1, 20.0, 4.0, step=0.1, key='t0_weeks')

        submitted = st.form_submit_button('生成报告')

    if submitted:
        st.session_state['analysis_data'] = {**scores, 'q1_delay': q1, 'q2_change': q2, 't0_weeks': t0_weeks}

    if st.session_state['analysis_data']:
        run_analysis(st.session_state['analysis_data'])


if __name__ == '__main__':
    main()
