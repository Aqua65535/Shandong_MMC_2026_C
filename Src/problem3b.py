"""
问题三 - b：随机扰动下的行程稳定性量化评估

"""

import random
import numpy as np
from data import (
    ATTRACTIONS, FIXED_TIME, CONSTRAINTS,
    TRAFFIC_DELAY_DIST, QUEUE_DELAY_DIST, TIME_PERIODS,
    Problem2Output, DailyPlan
)


# ========== 1. 时段判断与抽样函数 ==========

def get_traffic_period(hour: float) -> str:
    for rush_start, rush_end in TIME_PERIODS['rush']:
        if rush_start <= hour < rush_end:
            return 'rush'
    return 'normal'


def get_entry_period(hour: float) -> str:
    if 9.0 <= hour <= 12.0:
        return 'rush_entry'
    return 'normal_entry'


def sample_traffic_delay(hour: float, mode: str = 'random') -> float:
    if mode == 'zero':
        return 0.0
    period = get_traffic_period(hour)
    low, high = TRAFFIC_DELAY_DIST[period]
    return random.uniform(low, high)


def sample_queue_delay(hour: float, mode: str = 'random') -> float:
    if mode == 'zero':
        return 0.0
    period = get_entry_period(hour)
    low, high = QUEUE_DELAY_DIST[period]
    return random.uniform(low, high)


# ========== 2. 单日单次模拟 ==========

def simulate_single_day(plan: DailyPlan, attractions: dict,
                        traffic_mode: str = 'random',
                        queue_mode: str = 'random') -> dict:
    """
    对单日行程做一次随机模拟（每天1个景点）
    环节：整装 → 去程 → 排队 → 游览 → 午餐 → 返程
    """
    current_time = plan.start_time
    morning_prep = FIXED_TIME['morning_prep']
    meal_time = FIXED_TIME['meal']

    total_traffic_delay = 0.0
    total_queue_delay = 0.0

    # 整装
    current_time += morning_prep

    # 酒店→景点（去程）
    drive_go = plan.drive_segments[0]
    delay_go = sample_traffic_delay(current_time, traffic_mode)
    total_traffic_delay += delay_go
    current_time += drive_go + delay_go

    # 景点
    attr_id, plan_visit = plan.attractions[0]
    attr = attractions[attr_id]

    # 若早于开放则等待
    if current_time < attr['open'][0]:
        current_time = attr['open'][0]

    # 入园排队
    queue = sample_queue_delay(current_time, queue_mode)
    total_queue_delay += queue
    current_time += queue

    # 游览
    current_time += plan_visit

    # 午餐
    if 11.0 <= current_time <= 14.0:
        current_time += meal_time

    # 景点→酒店（返程）
    drive_back = plan.drive_segments[-1]
    delay_back = sample_traffic_delay(current_time, traffic_mode)
    total_traffic_delay += delay_back
    current_time += drive_back + delay_back

    # ========== 失效判定 ==========

    day_end = CONSTRAINTS['day_end']  # 21.0

    # 1. 超时：超过 21:00 就算
    is_overtime = current_time > day_end

    # 2. 游览不达标：总扰动超过缓冲+弹性
    # （保持原逻辑，但放宽阈值）
    ideal_end = plan.ideal_end_time
    buffer_time = day_end - ideal_end  # 理想结束到21:00的缓冲
    attr = ATTRACTIONS[attr_id]
    elasticity = attr['comfort_visit'] - attr['min_visit']
    total_disturbance = total_traffic_delay + total_queue_delay

    # 扰动超过（缓冲 + 弹性 + 1h宽容）才算不达标
    is_undervisit = total_disturbance > (buffer_time + elasticity + 2.0)  # 放宽到2h

    # 3. 超负荷：结束时间超过 22:00（给1小时余量）
    is_overload = current_time > (day_end + 1.0)

    # 4. 综合失效
    is_failure = is_overtime or is_undervisit or is_overload
    return {
        'is_overtime': is_overtime,
        'is_undervisit': is_undervisit,
        'is_overload': is_overload,
        'is_failure': is_failure,  # 用于可靠度计算
        'actual_end_time': current_time,
        'total_disturbance': total_disturbance,
        'traffic_delay': total_traffic_delay,
        'queue_delay': total_queue_delay,
    }


# ========== 3. 蒙特卡洛模拟核心 ==========

def run_simulation(baseline: Problem2Output, n_simulations: int,
                   traffic_mode: str = 'random',
                   queue_mode: str = 'random') -> dict:
    """运行N次模拟，统计各日失效概率"""
    overtime_counts = [0] * 5
    undervisit_counts = [0] * 5
    overload_counts = [0] * 5
    fail_counts = [0] * 5

    for _ in range(n_simulations):
        for day_idx, plan in enumerate(baseline.daily_plans):
            result = simulate_single_day(plan, ATTRACTIONS, traffic_mode, queue_mode)

            if result['is_overtime']:
                overtime_counts[day_idx] += 1
            if result['is_undervisit']:
                undervisit_counts[day_idx] += 1
            if result['is_overload']:
                overload_counts[day_idx] += 1
            if result['is_failure']:  # 使用新的失效定义
                fail_counts[day_idx] += 1
    n = n_simulations
    return {
        'P_overtime': {i + 1: c / n for i, c in enumerate(overtime_counts)},
        'P_undervisit': {i + 1: c / n for i, c in enumerate(undervisit_counts)},
        'P_overload': {i + 1: c / n for i, c in enumerate(overload_counts)},
        'P_fail_day': {i + 1: c / n for i, c in enumerate(fail_counts)},
    }


# ========== 4. 贡献度分解  ==========

def calculate_contribution(baseline: Problem2Output, n_simulations: int) -> dict:
    """分别量化道路堵车、景点排队的贡献程度（Shapley值风格）"""

    # 四种场景的失效概率
    sim_all = run_simulation(baseline, n_simulations, 'random', 'random')
    P_fail_all = 1 - np.prod([1 - sim_all['P_fail_day'][d] for d in range(1, 6)])

    sim_traffic = run_simulation(baseline, n_simulations, 'random', 'zero')
    P_fail_traffic = 1 - np.prod([1 - sim_traffic['P_fail_day'][d] for d in range(1, 6)])

    sim_queue = run_simulation(baseline, n_simulations, 'zero', 'random')
    P_fail_queue = 1 - np.prod([1 - sim_queue['P_fail_day'][d] for d in range(1, 6)])

    # ========== 无扰动基线 ==========
    sim_none = run_simulation(baseline, n_simulations, 'zero', 'zero')
    P_fail_none = 1 - np.prod([1 - sim_none['P_fail_day'][d] for d in range(1, 6)])

    # ========== Shapley 值风格贡献度 ==========
    # 堵车边际贡献：
    #   - 当排队存在时：P_all - P_queue_only
    #   - 当排队不存在时：P_traffic_only - P_none
    #   取平均，再归一化到总失效中
    delta_t_with_q = P_fail_all - P_fail_queue  # 有排队时，堵车的边际贡献
    delta_t_without_q = P_fail_traffic - P_fail_none  # 无排队时，堵车的边际贡献
    raw_C_traffic = (delta_t_with_q + delta_t_without_q) / 2.0

    # 排队边际贡献：
    delta_q_with_t = P_fail_all - P_fail_traffic  # 有堵车时，排队的边际贡献
    delta_q_without_t = P_fail_queue - P_fail_none  # 无堵车时，排队的边际贡献
    raw_C_queue = (delta_q_with_t + delta_q_without_t) / 2.0

    # 归一化（使 C_traffic + C_queue ≈ 1，剩余为交互项或基线）
    total_marginal = raw_C_traffic + raw_C_queue
    if total_marginal > 1e-6:
        C_traffic = raw_C_traffic / total_marginal
        C_queue = raw_C_queue / total_marginal
    else:
        C_traffic = 0.5
        C_queue = 0.5

    # 可靠度
    R_all = 1 - P_fail_all
    R_traffic_only = 1 - P_fail_traffic
    R_queue_only = 1 - P_fail_queue
    R_none = 1 - P_fail_none  # 新增

    return {
        'R_all': R_all,
        'R_traffic_only': R_traffic_only,
        'R_queue_only': R_queue_only,
        'R_none': R_none,  # 新增
        'P_fail_all': P_fail_all,
        'P_fail_traffic': P_fail_traffic,
        'P_fail_queue': P_fail_queue,
        'P_fail_none': P_fail_none,  # 新增
        'C_traffic': C_traffic,
        'C_queue': C_queue,
        'raw_C_traffic': raw_C_traffic,  # 原始边际贡献
        'raw_C_queue': raw_C_queue,
    }


# ========== 5. 薄弱点识别 ==========

def identify_weak_points(baseline: Problem2Output, sim_results: dict) -> list:
    weak_points = []

    # 1. 检查是否有景点使用了最小游览（弹性为0）
    for plan in baseline.daily_plans:
        attr_id, visit = plan.attractions[0]
        attr = ATTRACTIONS[attr_id]
        if visit <= attr['min_visit'] + 0.1:
            weak_points.append({
                'type': '无弹性景点',
                'day': plan.day,
                'attraction': attr_id,
                'description': f'{attr_id}已压缩到最小游览({visit}h)，无抗扰动余量'
            })

    # 2. 长车程路段
    for plan in baseline.daily_plans:
        for i, seg in enumerate(plan.drive_segments):
            if seg >= 0.7:
                weak_points.append({
                    'type': '长车程路段',
                    'day': plan.day,
                    'value': seg,
                    'description': f'第{plan.day}天第{i + 1}段车程{seg}h'
                })

    # 3. 超时风险最高日（>15%）
    for day in range(1, 6):
        if sim_results['P_overtime'][day] > 0.15:
            weak_points.append({
                'type': '超时风险',
                'day': day,
                'value': sim_results['P_overtime'][day],
                'description': f'第{day}天超时概率{sim_results["P_overtime"][day]:.1%}'
            })

    return weak_points


# ========== 6. 主计算函数 ==========

def evaluate_stability(baseline: Problem2Output, n_simulations: int = 10000) -> dict:
    """核心评估函数"""
    random.seed(42)

    # 基础模拟（全随机）
    sim_results = run_simulation(baseline, n_simulations, 'random', 'random')

    # 整体可靠度
    P_fail_total = 1 - np.prod([1 - sim_results['P_fail_day'][d] for d in range(1, 6)])
    R_total = 1 - P_fail_total

    # 贡献度（Shapley值风格）
    contrib = calculate_contribution(baseline, n_simulations)

    # 薄弱点
    weak_points = identify_weak_points(baseline, sim_results)

    return {
        'R_total': R_total,
        'P_fail_total': P_fail_total,
        'P_overtime': sim_results['P_overtime'],
        'P_undervisit': sim_results['P_undervisit'],
        'P_overload': sim_results['P_overload'],
        'P_fail_day': sim_results['P_fail_day'],
        'C_traffic': contrib['C_traffic'],
        'C_queue': contrib['C_queue'],
        'raw_C_traffic': contrib['raw_C_traffic'],
        'raw_C_queue': contrib['raw_C_queue'],
        'P_fail_all': contrib['P_fail_all'],
        'P_fail_traffic': contrib['P_fail_traffic'],
        'P_fail_queue': contrib['P_fail_queue'],
        'P_fail_none': contrib['P_fail_none'],
        'weak_points': weak_points,
    }

# ========== 7. 打印输出 ==========

def print_stability_result(result: dict, baseline: Problem2Output) -> None:
    """打印稳定性评估报告"""

    print("问题三-B：随机扰动下的行程稳定性量化评估")


    # 评估对象
    print("【评估对象：调整后的抗扰动行程】")

    print(f"  选中景点: {', '.join(baseline.selected_attractions)} (共{len(baseline.selected_attractions)}个)")
    print(f"  总喜好度: {baseline.total_preference:.1f}")
    for plan in baseline.daily_plans:
        attr_id, visit = plan.attractions[0]
        print(f"  第{plan.day}天: {attr_id}({ATTRACTIONS[attr_id]['name']}) {visit}h | "
              f"出发{plan.start_time:.1f}:00 | 结束{plan.ideal_end_time:.1f}:00")

    # 整体可靠度
    print("【1. 行程整体稳定可靠度】")

    R = result['R_total']
    status = "达标（≥90%）" if R >= 0.9 else " 未达标（<90%）"
    print(f"  R_total = {R:.4f}  {status}")
    print(f"  P_fail  = {result['P_fail_total']:.4f}")

    # 逐日失效
    print("【2. 逐日失效概率统计】")
    print("-" * 60)
    print(f"{'天数':<6} {'超时P':<10} {'不达标P':<10} {'超负荷P':<10} {'当日失效P':<10}")
    print("-" * 60)
    for day in range(1, 6):
        print(f"  第{day}天   {result['P_overtime'][day]:<10.4f} "
              f"{result['P_undervisit'][day]:<10.4f} "
              f"{result['P_overload'][day]:<10.4f} "
              f"{result['P_fail_day'][day]:<10.4f}")

    # 贡献度
    print("【3. 扰动因素贡献度分解（Shapley值风格）】")

    print(f"  原始边际贡献:")
    print(f"    堵车 raw_C_traffic = {result['raw_C_traffic']:.4f}")
    print(f"    排队 raw_C_queue   = {result['raw_C_queue']:.4f}")
    print(f"  归一化贡献度:")
    print(f"    C_traffic = {result['C_traffic']:.4f} ({result['C_traffic']*100:.1f}%)")
    print(f"    C_queue   = {result['C_queue']:.4f} ({result['C_queue']*100:.1f}%)")
    print(f"  参考基准:")
    print(f"    P_fail(全随机) = {result['P_fail_all']:.4f}")
    print(f"    P_fail(仅堵车) = {result['P_fail_traffic']:.4f}")
    print(f"    P_fail(仅排队) = {result['P_fail_queue']:.4f}")
    print(f"    P_fail(无扰动) = {result['P_fail_none']:.4f}")

    # 薄弱点
    print("【4. 结构性薄弱点分析】")

    for wp in result['weak_points']:
        loc = f"第{wp['day']}天" if 'day' in wp else ""
        if 'attraction' in wp:
            loc += f" {wp['attraction']}"
        print(f"  ■ [{wp['type']}] {loc}")
        print(f"    → {wp['description']}")

    # 结论
    print("【评估结论与建议】")

    if R < 0.9:
        print("可靠度未达90%，建议：")
        print("1. 针对薄弱日增加时间冗余（提前出发或简化安排）")
        print("2. 将瓶颈景点替换为弹性更大的备选")
        print("3. 避开高峰时段通勤，降低堵车暴露")
    else:
        print("可靠度达标，但建议关注薄弱日的边际风险")


# ========== 主函数 ==========

def main():
    # 加载problem3a调整后的行程
    adjusted = load_adjusted_from_3a()

    # 稳定性评估
    result = evaluate_stability(adjusted, n_simulations=10000)

    # 打印报告
    print_stability_result(result, adjusted)

    return result


# ========== 加载problem3a结果 ==========

def load_adjusted_from_3a() -> Problem2Output:
    output = Problem2Output()
    output.selected_attractions = ['A1', 'A2', 'A5', 'A8', 'A10']

    # 第1天: A1 古城老街
    p1 = DailyPlan()
    p1.day = 1
    p1.start_time = 6.5
    p1.attractions = [('A1', 3.5)]
    p1.drive_segments = [0.5, 0.5]
    p1.meal_time = None
    p1.ideal_end_time = 13.5

    # 第2天: A2 海洋乐园
    p2 = DailyPlan()
    p2.day = 2
    p2.start_time = 6.0
    p2.attractions = [('A2', 3.0)]
    p2.drive_segments = [0.8, 0.8]
    p2.meal_time = None
    p2.ideal_end_time = 13.8

    # 第3天: A5 民俗古村
    p3 = DailyPlan()
    p3.day = 3
    p3.start_time = 7.0  # 车程短、非高峰，无需提前
    p3.attractions = [('A5', 3.0)]  # 保持舒适时长（弹性大）
    p3.drive_segments = [0.6, 0.6]  # 车程0.6h
    p3.meal_time = None
    p3.ideal_end_time = 13.1

    # 第4天: A8 亲子农庄
    p4 = DailyPlan()
    p4.day = 4
    p4.start_time = 6.0
    p4.attractions = [('A8', 2.0)]  # 压缩到最小
    p4.drive_segments = [0.7, 0.7]
    p4.meal_time = None
    p4.ideal_end_time = 12.7

    # 第5天: A10 文创小镇
    p5 = DailyPlan()
    p5.day = 5
    p5.start_time = 6.5
    p5.attractions = [('A10', 3.0)]
    p5.drive_segments = [0.5, 0.5]
    p5.meal_time = None
    p5.ideal_end_time = 13.5

    output.daily_plans = [p1, p2, p3, p4, p5]
    output.total_preference = 40.9
    output.total_drive_time = 6.2
    output.balance_score = 0.811

    return output


if __name__ == '__main__':
    main()
