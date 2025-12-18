import streamlit as st
import numpy as np
import matplotlib
# 强制使用非交互式后端，防止在服务器运行报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
import json
import pandas as pd 

# ---------- 0. Google Sheets 配置 (安全读取) ----------
def get_gspread_client():
    try:
        # 这里的 "gcp_service_account" 对应第二步中 Secrets 的命名
        creds_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        gc = gspread.service_account_from_dict(creds_info, scopes)
        return gc
    except Exception as e:
        st.error(f"❌ 无法连接到 Google Sheets: {e}")
        st.info("💡 请确保已在 Streamlit Secrets 中配置了正确的密钥。")
        return None

SHEET_ID = "1qRsD5Z2LxM0QYrVKL8g_6ZxyAj5VQYDXxR2oVwKoB7I"

# ---------- 1. Matplotlib 中文与样式配置 ----------
# 注意：云端可能没有 Arial 字体，这里保留通用设置
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ---------- 2. 核心数学模型函数 ----------
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
    sigma = max(sigma, 1e-5)
    return A * np.exp(-((t - t0)**2) / (2*sigma**2))

def stability_analysis(t, A_val, t0, sigma, delta=0.01):
    right_limit = success_rate(t + delta, A_val, t0, sigma)
    left_limit = success_rate(t - delta, A_val, t0, sigma)
    if np.isnan(left_limit) or np.isnan(right_limit):
        return "异常状态 💀"
    is_limit_equal = abs(left_limit - right_limit) < 1e-2
    if is_limit_equal:
        return "尚在发展 🌱" if abs(left_limit - success_rate(t, A_val, t0, sigma)) < 1e-2 else "随缘 🍃"
    return "安排上了 🎁"

def determine_mode(delay_choice, change_choice):
    if delay_choice == 1 and change_choice == 1: return "mo_ceng"
    if delay_choice == 2 or change_choice == 2: return "sao_dong"
    return "random"

def calculate_score(raw_scores):
    total = sum(raw_scores)
    final = 1 + ((total - 3) / (15 - 3)) * (10 - 1)
    return np.clip(round(final), 1, 10)

def classify_love_type_en(I, P, C, threshold=7):
    is_i, is_p, is_c = I >= threshold, P >= threshold, C >= threshold
    if is_i and is_p and is_c: return "Consummate Love", "完美爱情：亲密、激情与承诺并存。"
    if is_i and is_c: return "Companionate Love", "伴侣之爱：深厚的友谊与承诺，但缺乏激情。"
    if is_p and is_c: return "Fatuous Love", "愚蠢之爱：仅靠激情支撑的承诺。"
    if is_i and is_p: return "Romantic Love", "浪漫之爱：情感与身体的联结，缺乏长期规划。"
    if is_i: return "Liking", "喜爱：纯粹的友谊。"
    if is_p: return "Infatuation", "迷恋：迷恋对方的外在或某种特质。"
    if is_c: return "Empty Love", "空洞之爱：徒留名义上的承诺。"
    return "Non-love", "无爱：日常的普通社交。"

# ---------- 3. 可视化函数 ----------
def plot_love_triangle(I, P, C):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    labels = ['Intimacy (I)', 'Passion (P)', 'Commitment (C)']
    values = np.array([I, P, C])
    values = np.concatenate((values, [I]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    ax.plot(angles, values, 'o-', linewidth=3, color='mediumvioletred')
    ax.fill(angles, values, color='lightpink', alpha=0.6)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_ylim(0, 10)
    
    love_type, desc = classify_love_type_en(I, P, C)
    ax.set_title(f"类型: {love_type}\n{desc}", pad=20)
    return fig

def plot_success_curve(A, t_peak, sigma, current_time):
    t = np.linspace(0, 15, 300)
    p = success_rate(t, A, t_peak, sigma)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, p, label="成功率曲线")
    ax.axvline(current_time, color='orange', label=f"预测时机: {current_time:.2f}w")
    ax.fill_between(t, 0, p, alpha=0.1)
    ax.set_xlabel("时间 (周)")
    ax.set_ylabel("成功概率")
    ax.legend()
    return fig

# ---------- 4. 分析逻辑 ----------
def run_analysis(data):
    # 计算 IPC
    I = calculate_score([data[f'i{i}'] for i in range(1, 4)])
    P = calculate_score([data[f'p{i}'] for i in range(1, 4)])
    C = calculate_score([data[f'c{i}'] for i in range(1, 4)])
    
    # 模型推导
    A = 0.5 + ((I + P + C) / 30.0) * 0.5
    sigma = 0.5 + (C / 10.0) * 1.5
    t_peak = np.clip(data['t0_weeks'] * (1.0 - ((I/10.0 + C/10.0)/2.0)*0.5), 0.1, None)
    
    mode = determine_mode(data['q1_delay'], data['q2_change'])
    times = generate_confession_times(mode)
    current_time = np.clip(t_peak + (np.mean(times[-10:]) - 1) * (sigma / 2), 0.1, 15)
    
    status = stability_analysis(current_time, A, t_peak, sigma)
    
    # 存入 Google Sheets
    gc = get_gspread_client()
    if gc:
        try:
            sheet = gc.open_by_key(SHEET_ID).sheet1
            sheet.append_row([str(pd.Timestamp.now()), I, P, C, round(current_time, 2), status])
            st.success("✅ 数据已同步至云端表格")
        except Exception as e:
            st.warning(f"无法保存到表格: {e}")

    # UI 显示
    st.divider()
    st.header(f"诊断结论：{status}")
    c1, c2, c3 = st.columns(3)
    c1.metric("亲密 I", I)
    c2.metric("激情 P", P)
    c3.metric("承诺 C", C)
    
    st.pyplot(plot_love_triangle(I, P, C))
    st.pyplot(plot_success_curve(A, t_peak, sigma, current_time))

# ---------- 5. 主程序 ----------
def main():
    st.title("💌 恋爱告急·表白分析系统")
    
    with st.form("main_form"):
        q1 = st.radio("Q1. 你的行为倾向：", [1, 2], format_func=lambda x: "推迟 (1)" if x==1 else "果断 (2)")
        q2 = st.radio("Q2. 计划变动：", [1, 2], format_func=lambda x: "稳健 (1)" if x==1 else "灵活 (2)")
        
        st.write("--- 关系评估 (1-5分) ---")
        scores = {}
        for cat, label in [('i', '亲密'), ('p', '激情'), ('c', '承诺')]:
            for i in range(1, 4):
                scores[f'{cat}{i}'] = st.slider(f"{label}指标 {i}", 1, 5, 3)
        
        t0 = st.number_input("距离下次节日/纪念日还有几周？", 0.1, 20.0, 4.0)
        submitted = st.form_submit_button("开始量子波动分析 ✨")

    if submitted:
        data = {**scores, 'q1_delay': q1, 'q2_change': q2, 't0_weeks': t0}
        run_analysis(data)

if __name__ == "__main__":
    main()
