"""
问题三 - a：基准行程的计算
"""

from data import (
    ATTRACTIONS, FIXED_TIME,
    TIME_PERIODS, TRAFFIC_DELAY_DIST, QUEUE_DELAY_DIST,
    Problem2Output, DailyPlan
)


def get_traffic_period(hour: float) -> str:
    """判断行车时段"""
    for rush_start, rush_end in TIME_PERIODS['rush']:
        if rush_start <= hour < rush_end:
            return 'rush'
    return 'normal'


def get_entry_period(hour: float) -> str:
    """判断入园时段"""
    if 9.0 <= hour <= 12.0:
        return 'rush_entry'
    return 'normal_entry'


def expected_traffic_delay(hour: float) -> float:
    """期望堵车延时"""
    period = get_traffic_period(hour)
    low, high = TRAFFIC_DELAY_DIST[period]
    return (low + high) / 2


def expected_queue_delay(hour: float) -> float:
    """期望排队延时"""
    period = get_entry_period(hour)
    low, high = QUEUE_DELAY_DIST[period]
    return (low + high) / 2


def calculate_risk_exposure(plan: DailyPlan) -> dict:
    """
    计算单日行程的风险暴露度
    每天1个景点，风险主要来自：酒店往返车程 + 入园排队
    """
    current_time = plan.start_time + FIXED_TIME['morning_prep']

    # 酒店→景点（去程）
    drive_go = plan.drive_segments[0]
    go_period = get_traffic_period(current_time)
    go_risk = 2.5 if go_period == 'rush' else 0.75  # 期望堵车

    current_time += drive_go

    # 入园排队风险
    entry_period = get_entry_period(current_time)
    queue_risk = 1.75 if entry_period == 'rush_entry' else 0.5  # 期望排队

    # 景点→酒店（返程）
    # 返程时间取决于游览结束时间
    visit_end = current_time + plan.attractions[0][1]
    back_period = get_traffic_period(visit_end)
    back_risk = 2.5 if back_period == 'rush' else 0.75

    total_risk = go_risk + queue_risk + back_risk

    return {
        'go_risk': go_risk,
        'queue_risk': queue_risk,
        'back_risk': back_risk,
        'total_risk': total_risk,
        'entry_period': entry_period,
        'go_period': go_period,
    }


def adjust_daily_schedule(plan: DailyPlan) -> DailyPlan:
    morning_prep = FIXED_TIME['morning_prep']
    meal_time = FIXED_TIME['meal']

    attr_id, comfort_visit = plan.attractions[0]
    attr = ATTRACTIONS[attr_id]
    min_visit = attr['min_visit']

    new_start = plan.start_time
    use_visit = comfort_visit

    # ========== 新增：长车程景点强制早出发 ==========
    if plan.drive_segments[0] >= 0.7:
        new_start = min(new_start, 6.0)

    # 策略1：入园高峰风险 → 提前出发
    arrival_time_candidate = new_start + morning_prep + plan.drive_segments[0]
    if 9.0 <= arrival_time_candidate <= 12.0:
        target_arrival = 8.5
        needed_start = target_arrival - morning_prep - plan.drive_segments[0]
        new_start = max(5.5, needed_start)

    # 策略2：长车程景点强制压缩到最小游览
    if plan.drive_segments[0] >= 0.7 and comfort_visit > min_visit:
        use_visit = min_visit

    # 重新计算时间轴
    current_time = new_start + morning_prep
    current_time += plan.drive_segments[0]
    if current_time < attr['open'][0]:
        current_time = attr['open'][0]
    current_time += use_visit
    if 11.0 <= current_time <= 14.0:
        current_time += meal_time
    current_time += plan.drive_segments[-1]

    new_plan = DailyPlan()
    new_plan.day = plan.day
    new_plan.start_time = round(new_start, 1)
    new_plan.drive_segments = plan.drive_segments[:]
    new_plan.attractions = [(attr_id, use_visit)]
    new_plan.ideal_end_time = round(current_time, 1)
    new_plan.meal_time = None

    return new_plan


def build_adjusted_itinerary(baseline: Problem2Output) -> Problem2Output:
    """生成抗扰动行程"""
    adjusted_plans = []
    for plan in baseline.daily_plans:
        new_plan = adjust_daily_schedule(plan)
        adjusted_plans.append(new_plan)

    output = Problem2Output()
    output.selected_attractions = baseline.selected_attractions[:]
    output.daily_plans = adjusted_plans
    output.total_preference = baseline.total_preference
    output.total_drive_time = baseline.total_drive_time
    output.balance_score = baseline.balance_score

    return output


def print_adjustment(before: Problem2Output, after: Problem2Output) -> None:
    """打印调整对比"""

    print("问题三-A：带期望扰动的行程时间轴调整")


    print("【风险暴露与调整策略】")

    print(f"{'天数':<6} {'景点':<8} {'风险评分':<10} {'调整前结束':<12} {'调整后结束':<12} {'出发':<8} {'游览时长'}")


    for i in range(len(before.daily_plans)):
        b = before.daily_plans[i]
        a = after.daily_plans[i]
        risk = calculate_risk_exposure(b)
        attr_id = b.attractions[0][0]

        visit_change = ""
        if a.attractions[0][1] < b.attractions[0][1]:
            visit_change = f" 压缩{b.attractions[0][1]}→{a.attractions[0][1]}h"

        print(f"  第{b.day}天   {attr_id:<8} {risk['total_risk']:<10.1f} {b.ideal_end_time:<12.1f} "
              f"{a.ideal_end_time:<12.1f} {a.start_time:<8.1f} {a.attractions[0][1]}h{visit_change}")


def load_baseline_from_problem2() -> Problem2Output:
    """硬编码problem2输出（每天1个景点版）"""
    output = Problem2Output()
    output.selected_attractions = ['A1', 'A2', 'A5', 'A8', 'A10']  # ✅ 改成A5

    # 第1天: A1 古城老街
    p1 = DailyPlan()
    p1.day = 1
    p1.start_time = 7.0
    p1.attractions = [('A1', 3.5)]
    p1.drive_segments = [0.5, 0.5]
    p1.meal_time = None
    p1.ideal_end_time = 14.0

    # 第2天: A2 海洋乐园
    p2 = DailyPlan()
    p2.day = 2
    p2.start_time = 7.0
    p2.attractions = [('A2', 5.0)]
    p2.drive_segments = [0.8, 0.8]
    p2.meal_time = None
    p2.ideal_end_time = 15.1

    # 第3天: A5 民俗古村
    p3 = DailyPlan()
    p3.day = 3
    p3.start_time = 7.0
    p3.attractions = [('A5', 3.0)]  # 民俗古村 comfort_visit = 3.0
    p3.drive_segments = [0.6, 0.6]  # 车程 0.6h（根据data.py，hotel-A5 = 0.6）
    p3.meal_time = None
    p3.ideal_end_time = 13.1  # 根据 problem2 输出

    # 第4天: A8 亲子农庄
    p4 = DailyPlan()
    p4.day = 4
    p4.start_time = 7.0
    p4.attractions = [('A8', 3.0)]
    p4.drive_segments = [0.7, 0.7]
    p4.meal_time = None
    p4.ideal_end_time = 13.9

    # 第5天: A10 文创小镇
    p5 = DailyPlan()
    p5.day = 5
    p5.start_time = 7.0
    p5.attractions = [('A10', 3.0)]
    p5.drive_segments = [0.5, 0.5]
    p5.meal_time = None
    p5.ideal_end_time = 13.5

    output.daily_plans = [p1, p2, p3, p4, p5]
    output.total_preference = 40.9
    output.total_drive_time = 6.2
    output.balance_score = 0.811

    return output


def main():
    # 加载problem2基准行程
    baseline = load_baseline_from_problem2()

    # 生成抗扰动行程
    adjusted = build_adjusted_itinerary(baseline)

    # 打印对比
    print_adjustment(baseline, adjusted)

    return adjusted


if __name__ == '__main__':
    main()