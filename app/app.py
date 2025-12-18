# ========== ① 爱之三角图（极坐标）==========
@st.cache_data
def plot_love_triangle(I, P, C):
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    labels = ['亲密 (I)', '激情 (P)', '承诺 (C)']
    values = np.array([I, P, C])
    values = np.concatenate((values, [I]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # 每轴彩色渐变 + 阴影
    axis_colors = ['#4B92DB', '#FF6B6B', '#4ECB71']
    for ang, val, color in zip(angles[:-1], values[:-1], axis_colors):
        ax.bar(ang, val, width=2*np.pi/3, color=color, alpha=0.65, edgecolor=color, lw=2)
        # 数值标签
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
    ax.set_title(f"💞 {love_type}\n{desc}", pad=25, fontsize=14, color='darkslategray')
    return fig


# ========== ② 成功率曲线（直角坐标）==========
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
    ax.set_title('📈 表白时机成功率分析', fontsize=15, pad=15)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(color='lightgray', linestyle='--', alpha=0.6)

    return fig
