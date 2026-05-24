"""
可视化模块
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})


COLORS = {
    'primary': '#2E86AB',   # 主色：深蓝
    'secondary': '#A23B72', # 次色：玫红
    'accent': '#F18F01',    # 强调色：橙黄
    'success': '#73AB84',   # 成功色：苔绿
    'warning': '#F18F01',   # 警告色：橙黄
    'info': '#3E92CC',      # 信息色：浅蓝
    'neutral': '#6C6E70',   # 中性色：灰
    'grid': '#D3D3D3',      # 网格线：浅灰
    'background': '#F8F9FA' # 背景色：浅灰白
}

MACARON_COLORS = {
    '整装': '#A8E6CF',   # 薄荷绿
    '车程': '#FFD3B6',   # 杏仁色
    '游览': '#C7CEEA',   # 淡薰衣草
    '就餐': '#FF8B94',   # 珊瑚粉
}


from data import ATTRACTIONS, CONSTRAINTS, FIXED_TIME
from problem1 import calculate_comprehensive_priority
from problem2 import design_baseline_itinerary


def plot_radar_chart():
    """
    景点四维特征雷达图（前5名景点）
    维度：喜好度、游览耗时、通勤距离、拥堵敏感度
    """
    priority_full = calculate_comprehensive_priority(ATTRACTIONS)

    # 只取前5名
    top5 = priority_full[:5]
    df = pd.DataFrame(top5)
    df = df.set_index('id')

    # 修改为四维度
    dimensions = ['preference_score', 'visit_score', 'commute_score', 'congestion_score']
    dim_labels = ['喜好度', '游览耗时', '通勤距离', '拥堵敏感度']

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))

    radar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
                    COLORS['success'], COLORS['info']]

    for i, (idx, row) in enumerate(df.iterrows()):
        values = [row[d] for d in dimensions]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=f"{idx}({row['name']})",
                color=radar_colors[i % len(radar_colors)])
        ax.fill(angles, values, alpha=0.1, color=radar_colors[i % len(radar_colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('景点四维特征雷达图（Top5）', fontsize=13, fontweight='bold', pad=20)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig('景点四维雷达图.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_linkage_heatmap():
    """
    景点联动评分热力图
    """
    from problem1 import calculate_linkage_score

    attraction_ids = list(ATTRACTIONS.keys())
    n = len(attraction_ids)

    linkage_matrix = np.zeros((n, n))
    for i, a1 in enumerate(attraction_ids):
        for j, a2 in enumerate(attraction_ids):
            if i == j:
                linkage_matrix[i, j] = np.nan
            else:
                linkage_matrix[i, j] = calculate_linkage_score(a1, a2, ATTRACTIONS)

    fig, ax = plt.subplots(figsize=(9, 7))


    im = ax.imshow(linkage_matrix, cmap='Blues', vmin=0, vmax=1, alpha=0.8)

    for i in range(n):
        for j in range(n):
            if i != j and linkage_matrix[i, j] > 0:

                text_color = 'white' if linkage_matrix[i, j] > 0.6 else 'black'
                ax.text(j, i, f'{linkage_matrix[i, j]:.2f}',
                        ha="center", va="center", color=text_color, fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(attraction_ids, fontsize=10)
    ax.set_yticklabels(attraction_ids, fontsize=10)

    ax.set_xlabel('景点编号', fontsize=11)
    ax.set_ylabel('景点编号', fontsize=11)
    ax.set_title('景点联动评分热力图', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('联动评分', fontsize=10)

    plt.tight_layout()
    plt.savefig('景点联动热力图.png', dpi=300, bbox_inches='tight')
    plt.show()



def plot_gantt_chart(output):
    """
    5天行程甘特图（优化版：景点名称与日期坐标一起显示）
    """
    morning_prep = FIXED_TIME['morning_prep']
    meal_time = FIXED_TIME['meal']

    fig, ax = plt.subplots(figsize=(13, 5.5))

    y_pos = 0
    y_ticks = []
    y_labels = []

    gantt_colors = MACARON_COLORS

    for plan in output.daily_plans:
        current_time = plan.start_time

        # 整装
        ax.barh(y_pos, morning_prep, left=current_time, facecolor=gantt_colors['整装'],
                edgecolor='white', linewidth=0.8)
        current_time += morning_prep

        # 车程
        drive_go = plan.drive_segments[0]
        ax.barh(y_pos, drive_go, left=current_time, facecolor=gantt_colors['车程'],
                edgecolor='white', linewidth=0.8)
        current_time += drive_go

        # 游览
        attr_id, visit_time = plan.attractions[0]
        ax.barh(y_pos, visit_time, left=current_time, facecolor=gantt_colors['游览'],
                edgecolor='white', linewidth=0.8)
        # 标注第一个景点名称（在游览条内部左侧）
        attr = ATTRACTIONS[attr_id]
        ax.text(current_time + 0.05, y_pos, attr['name'],
                fontsize=8, va='center', ha='left', color='white', fontweight='bold')
        current_time += visit_time

        # 午餐
        if 11.5 <= current_time <= 13.5:
            ax.barh(y_pos, meal_time, left=current_time, facecolor=gantt_colors['就餐'],
                    edgecolor='white', linewidth=0.8)
            current_time += meal_time

        # 第二个景点
        if len(plan.attractions) >= 2:
            drive_2 = plan.drive_segments[1]
            ax.barh(y_pos, drive_2, left=current_time, facecolor=gantt_colors['车程'],
                    edgecolor='white', linewidth=0.8)
            current_time += drive_2

            attr2_id, visit2_time = plan.attractions[1]
            ax.barh(y_pos, visit2_time, left=current_time, facecolor=gantt_colors['游览'],
                    edgecolor='white', linewidth=0.8)
            # 标注第二个景点名称（在游览条内部左侧）
            attr2 = ATTRACTIONS[attr2_id]
            ax.text(current_time + 0.05, y_pos, attr2['name'],
                    fontsize=8, va='center', ha='left', color='white', fontweight='bold')
            current_time += visit2_time

        # 返回车程
        drive_back = plan.drive_segments[-1]
        ax.barh(y_pos, drive_back, left=current_time, facecolor=gantt_colors['车程'],
                edgecolor='white', linewidth=0.8)

        y_ticks.append(y_pos)
        y_labels.append(f'第{plan.day}天')
        y_pos += 1

    ax.invert_yaxis()

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('时间（小时）', fontsize=11)
    ax.set_title('5天行程甘特图', fontsize=13, fontweight='bold')
    ax.set_xlim(6, 16)
    ax.axvline(x=21, color=COLORS['warning'], linestyle='--', linewidth=1.5, label='21:00截止线')

    legend_patches = [mpatches.Patch(color=gantt_colors[l], label=l)
                      for l in ['整装', '车程', '游览', '就餐']]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('行程甘特图.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_daily_load(output):
    """
    每日负荷对比（游览+车程+整装+就餐）
    """
    days = []
    visit_loads = []
    drive_loads = []
    prep_loads = []
    meal_loads = []

    for plan in output.daily_plans:
        days.append(f'第{plan.day}天')
        total_visit = sum(duration for _, duration in plan.attractions)
        total_drive = sum(plan.drive_segments)

        # 整装时间
        prep = FIXED_TIME['morning_prep']

        # 就餐时间（根据行程判断）
        current_time = plan.start_time + prep
        current_time += plan.drive_segments[0]
        current_time += plan.attractions[0][1]
        meal = 0
        if 11.5 <= current_time <= 13.5:
            meal += FIXED_TIME['meal']
        if len(plan.attractions) >= 2:
            current_time += plan.drive_segments[1]
            current_time += plan.attractions[1][1]
            if 17.5 <= current_time <= 19.5:
                meal += FIXED_TIME['meal']

        visit_loads.append(total_visit)
        drive_loads.append(total_drive)
        prep_loads.append(prep)
        meal_loads.append(meal)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(days))
    width = 0.6

    bottom1 = np.array(prep_loads)
    bottom2 = bottom1 + np.array(drive_loads)
    bottom3 = bottom2 + np.array(visit_loads)

    bars1 = ax.bar(x, prep_loads, width, label='整装', color=MACARON_COLORS['整装'],
                   edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, drive_loads, width, bottom=bottom1, label='车程',
                   color=MACARON_COLORS['车程'], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x, visit_loads, width, bottom=bottom2, label='游览',
                   color=MACARON_COLORS['游览'], edgecolor='white', linewidth=1)
    bars4 = ax.bar(x, meal_loads, width, bottom=bottom3, label='就餐',
                   color=MACARON_COLORS['就餐'], edgecolor='white', linewidth=1)

    total_loads = bottom3 + np.array(meal_loads)
    ax.plot(x, total_loads, 'o-', color=COLORS['secondary'], linewidth=2, markersize=8,
            label='总负荷', zorder=5)

    for i, v in enumerate(total_loads):
        ax.text(i, v + 0.15, f'{v:.1f}h', ha='center', va='bottom', fontsize=9,
                color=COLORS['secondary'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(days, fontsize=10)
    ax.set_ylabel('时长（小时）', fontsize=11)
    ax.set_title('每日行程负荷对比', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS['grid'])
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig('每日负荷对比.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence_curve(baseline, max_sims=20000, step=1000):
    """蒙特卡洛收敛性曲线"""
    from problem3b import evaluate_stability

    sim_counts = list(range(step, max_sims + step, step))
    reliabilities = []

    for n in sim_counts:
        result = evaluate_stability(baseline, n_simulations=n)
        reliabilities.append(result['R_total'])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sim_counts, reliabilities, 'o-', linewidth=2, markersize=5, color=COLORS['primary'])
    ax.axhline(y=reliabilities[-1], color=COLORS['warning'], linestyle='--', alpha=0.7,
               label=f'收敛值: {reliabilities[-1]:.4f}')

    # 添加波动带
    rolling_std = pd.Series(reliabilities).rolling(window=5).std()
    ax.fill_between(sim_counts,
                    [r - s for r, s in zip(reliabilities, rolling_std)],
                    [r + s for r, s in zip(reliabilities, rolling_std)],
                    alpha=0.2, color=COLORS['primary'])

    ax.set_xlabel('模拟次数', fontsize=11)
    ax.set_ylabel('整体可靠度', fontsize=11)
    ax.set_title('蒙特卡洛模拟收敛性分析', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', color=COLORS['grid'])
    ax.set_facecolor('#FAFAFA')

    plt.savefig('收敛性曲线.png', dpi=300, bbox_inches='tight')
    plt.show()



def generate_all_charts():

    print("【问题一】生成图表...")
    plot_radar_chart()
    plot_linkage_heatmap()

    print("【问题二】生成图表...")
    output, _, _, _ = design_baseline_itinerary()
    plot_gantt_chart(output)
    plot_daily_load(output)

    print("【问题三】生成图表...")
    plot_convergence_curve(output, max_sims=20000, step=1000)


if __name__ == '__main__':
    generate_all_charts()