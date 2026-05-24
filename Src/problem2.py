"""
问题二：无随机扰动下的多目标景点优选与基准行程设计

"""

import numpy as np
from itertools import combinations
from data import ATTRACTIONS, DRIVE_MATRIX, FIXED_TIME, CONSTRAINTS, Problem2Output, DailyPlan


# 1. 计算景点综合价值，调用10次（每个景点）
def calculate_attraction_value(attr_id, attractions):
    """
    计算景点综合价值得分（用于景点优选）
    维度：喜好度(0.5) + 游览舒适时长归一化(0.2) + 通勤便利性(0.3)
    不考虑堵车和排队
    """
    attr = attractions[attr_id]

    # 喜好度（原始分，最高9.2，最低6.8）
    preference = attr['preference']
    preference_normalized = (preference - 6.5) / (9.5 - 6.5)

    # 舒适游览时长归一化（越短越灵活）
    comfort = attr['comfort_visit']
    comfort_normalized = 1 - (comfort - 2.5) / (5.0 - 2.5)
    comfort_normalized = max(0, min(1, comfort_normalized))

    # 通勤便利性：到酒店距离 + 到其他景点的平均距离
    hotel_dist = DRIVE_MATRIX[('hotel', attr_id)]
    other_dists = []
    for other in attractions.keys():
        if other != attr_id:
            other_dists.append(DRIVE_MATRIX[(attr_id, other)])
    avg_other_dist = np.mean(other_dists) if other_dists else 0

    commute_score = 1 - (hotel_dist * 0.6 + avg_other_dist * 0.4) / 1.5
    commute_score = max(0, min(1, commute_score))

    # 综合得分
    total = (preference_normalized * 0.5 +
             comfort_normalized * 0.2 +
             commute_score * 0.3)

    return round(total, 3)


# =========================================
def estimate_combo_feasibility(combo, attractions):
    """
    估计组合的行程可行性
    基于景点开放时间、游览时长，能否在5天内合理安排
    """
    # 计算总所需游览时间（使用舒适时长）
    total_visit = sum(attractions[aid]['comfort_visit'] for aid in combo)

    # 每天最多2个景点，计算所需最少天数
    days_needed = (len(combo) + 1) // 2

    if days_needed > 5:
        return 0.0

    # 估算每天可用时间（7:00-21:00共14小时）
    daily_available = 14.0
    daily_fixed = FIXED_TIME['morning_prep'] + FIXED_TIME['meal']

    # 日均所需时间
    avg_daily_visit = total_visit / days_needed
    avg_daily_drive = 0.5

    total_daily_need = avg_daily_visit + avg_daily_drive + daily_fixed

    if total_daily_need <= daily_available:
        return 1.0
    else:
        return max(0, 1 - (total_daily_need - daily_available) / daily_available)

# 调用C(10,5)+C(10,6)+C(10,7)+C(10,8)次
def evaluate_attraction_combination(combo, attractions):
    """
    评价一个景点组合的综合质量
    返回：(综合得分, 指标字典)
    """
    # 1. 总喜好度（归一化）
    total_preference = sum(attractions[aid]['preference'] for aid in combo)
    max_possible_pref = sum(sorted([a['preference'] for a in attractions.values()], reverse=True)[:len(combo)])
    pref_score = total_preference / max_possible_pref if max_possible_pref > 0 else 0

    # 2. 组合内通勤效率（景点间平均车程）
    drive_times = []
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            drive_times.append(DRIVE_MATRIX[(combo[i], combo[j])])
    avg_combo_drive = np.mean(drive_times) if drive_times else 0
    drive_score = 1 - avg_combo_drive / 1.5
    drive_score = max(0, min(1, drive_score))

    # 3. 组合的行程可行性评估
    feasibility_score = estimate_combo_feasibility(combo, attractions)

    # 综合得分
    total_score = (pref_score * 0.4 +
                   drive_score * 0.3 +
                   feasibility_score * 0.3)

    metrics = {
        'total_preference': total_preference,
        'pref_score': pref_score,
        'avg_combo_drive': avg_combo_drive,
        'drive_score': drive_score,
        'feasibility_score': feasibility_score
    }

    return total_score, metrics


# 2. 优选景点组合，遍历所有组合，找出最优
def select_optimal_attractions(attractions, min_num=5, max_num=8):
    """
    从备选景点中优选一组合理景点组合
    使用多目标评价：总喜好度 + 组合内通勤效率
    """
    attr_ids = list(attractions.keys())
    n = len(attr_ids)

    best_combination = None
    best_score = -np.inf
    best_metrics = {}

    for k in range(min_num, min(max_num, n) + 1):
        for combo in combinations(attr_ids, k):
            combo_score, metrics = evaluate_attraction_combination(combo, attractions)

            if combo_score > best_score:
                best_score = combo_score
                best_combination = combo
                best_metrics = metrics

    return list(best_combination), best_metrics


# =========================================
# 调用C(n,2)次
def calculate_simple_linkage_score(attr1_id, attr2_id, attractions):
    """
    计算两个景点之间的基础联动评分（不考虑堵车排队）
    """
    attr1 = attractions[attr1_id]
    attr2 = attractions[attr2_id]

    # 地理邻近性
    drive_time = DRIVE_MATRIX[(attr1_id, attr2_id)]
    geo_score = 1 - min(1, drive_time / 1.2)

    # 时间兼容性：开放时间重叠
    open1_start, open1_end = attr1['open']
    open2_start, open2_end = attr2['open']
    overlap_start = max(open1_start, open2_start)
    overlap_end = min(open1_end, open2_end)
    overlap_hours = max(0, overlap_end - overlap_start)

    # 最小游览时间之和 + 车程
    min_total = attr1['min_visit'] + attr2['min_visit']
    need_time = min_total + drive_time + 0.5
    time_score = min(1.0, overlap_hours / need_time) if need_time > 0 else 0

    # 类型互补
    type_score = 1.0 if attr1['type'] != attr2['type'] else 0.5

    total = geo_score * 0.4 + time_score * 0.4 + type_score * 0.2
    return round(total, 3)

# 3. 按天分组
def group_attractions_by_day(selected_attractions, attractions):
    """
    将选中的景点分配到5天中（每天1-2个景点）
    确保每天至少1个景点
    使用贪心算法优化：优先将地理邻近的景点组合在同一天
    """
    n = len(selected_attractions)
    total_days = CONSTRAINTS['total_days']

    # 计算所有景点两两之间的联动评分
    linkage_scores = {}
    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = selected_attractions[i], selected_attractions[j]
            score = calculate_simple_linkage_score(a1, a2, attractions)
            linkage_scores[(a1, a2)] = score

    # 按联动评分降序排序
    sorted_pairs = sorted(linkage_scores.items(), key=lambda x: x[1], reverse=True)

    used = set()
    daily_groups = []

    # 先处理高联动评分的组合
    for (a1, a2), score in sorted_pairs:
        if a1 not in used and a2 not in used:
            attr1, attr2 = attractions[a1], attractions[a2]
            open1_start, open1_end = attr1['open']
            open2_start, open2_end = attr2['open']
            overlap_start = max(open1_start, open2_start)
            overlap_end = min(open1_end, open2_end)
            overlap_hours = max(0, overlap_end - overlap_start)

            need_time = attr1['min_visit'] + attr2['min_visit'] + DRIVE_MATRIX[(a1, a2)] + 0.5

            if overlap_hours >= need_time or score > 0.7:
                daily_groups.append([a1, a2])
                used.add(a1)
                used.add(a2)

    # 剩余未分配的景点单独成组
    remaining = [aid for aid in selected_attractions if aid not in used]

    for aid in remaining:
        daily_groups.append([aid])

    # 如果组数多于5天，合并一些组
    while len(daily_groups) > total_days:
        single_groups = [i for i, g in enumerate(daily_groups) if len(g) == 1]
        if len(single_groups) >= 2:
            idx1, idx2 = single_groups[0], single_groups[1]
            daily_groups[idx1].extend(daily_groups[idx2])
            daily_groups.pop(idx2)
        else:
            daily_groups[-2].extend(daily_groups[-1])
            daily_groups.pop()

    # 确保恰好5天
    while len(daily_groups) < total_days:
        if daily_groups and len(daily_groups[-1]) >= 2:
            last_group = daily_groups.pop()
            daily_groups.append([last_group[0]])
            daily_groups.append([last_group[1]])
        else:
            daily_groups.append([])

    # 填补空缺日
    for day_idx, group in enumerate(daily_groups):
        if not group and day_idx < len(selected_attractions):
            for other_idx, other_group in enumerate(daily_groups):
                if len(other_group) >= 2:
                    daily_groups[day_idx].append(other_group.pop())
                    break

    return daily_groups[:total_days]


# =========================================
# 4. 编排详细时序
def arrange_daily_schedule(day_groups, attractions):
    """
    编排每天的详细时序行程
    返回DailyPlan列表
    """
    daily_plans = []
    day_start = CONSTRAINTS['day_start']
    day_end = CONSTRAINTS['day_end']
    morning_prep = FIXED_TIME['morning_prep']
    meal_time = FIXED_TIME['meal']

    for day_idx, group in enumerate(day_groups):
        if not group:
            continue

        current_time = day_start + morning_prep

        drive_segments = []
        attractions_plan = []

        # 从酒店到第一个景点
        first_attr = group[0]
        drive_to_first = DRIVE_MATRIX[('hotel', first_attr)]
        drive_segments.append(drive_to_first)
        current_time += drive_to_first

        # 游览第一个景点
        attr1 = attractions[first_attr]
        visit1_time = attr1['comfort_visit']
        attractions_plan.append((first_attr, visit1_time))
        current_time += visit1_time

        # 就餐时间（午餐）
        if 11.5 <= current_time <= 13.5:
            current_time += meal_time

        # 如果有第二个景点
        if len(group) >= 2:
            second_attr = group[1]
            drive_to_second = DRIVE_MATRIX[(first_attr, second_attr)]
            drive_segments.append(drive_to_second)
            current_time += drive_to_second

            attr2 = attractions[second_attr]
            visit2_time = attr2['comfort_visit']
            attractions_plan.append((second_attr, visit2_time))
            current_time += visit2_time

            if 17.5 <= current_time <= 19.5:
                current_time += meal_time

        # 返回酒店
        last_attr = group[-1]
        drive_back = DRIVE_MATRIX[(last_attr, 'hotel')]
        drive_segments.append(drive_back)
        current_time += drive_back

        ideal_end_time = current_time

        # 调整出发时间使行程在21:00前结束
        if ideal_end_time > day_end:
            new_start = day_start - (ideal_end_time - day_end)
            new_start = max(7.0, new_start)
            current_time = new_start + morning_prep
            current_time += drive_to_first + visit1_time
            if len(group) >= 2:
                current_time += drive_to_second + visit2_time
            current_time += drive_back
            ideal_end_time = current_time

        daily_plan = DailyPlan()
        daily_plan.day = day_idx + 1
        daily_plan.start_time = day_start
        daily_plan.attractions = attractions_plan
        daily_plan.drive_segments = drive_segments
        daily_plan.meal_time = None
        daily_plan.ideal_end_time = ideal_end_time

        daily_plans.append(daily_plan)

    return daily_plans


# =========================================
# 5. 计算评估指标
def calculate_total_preference(selected_attractions, attractions):
    return sum(attractions[aid]['preference'] for aid in selected_attractions)


def calculate_total_drive_time(daily_plans):
    total = 0
    for plan in daily_plans:
        total += sum(plan.drive_segments)
    return total


def calculate_balance_score(daily_plans):
    if len(daily_plans) == 0:
        return 0

    daily_loads = []
    for plan in daily_plans:
        total_visit = sum(duration for _, duration in plan.attractions)
        total_drive = sum(plan.drive_segments)
        daily_loads.append(total_visit + total_drive)

    if len(daily_loads) < 5:
        daily_loads.extend([0] * (5 - len(daily_loads)))

    std_load = np.std(daily_loads)

    balance_score = max(0, 1 - std_load / 5.0)
    return round(balance_score, 3)


# =========================================
# 主函数部分1：计算
def design_baseline_itinerary():
    """
    设计基准行程方案
    输入：data.py原始参数
    处理：景点优选、行程编排
    输出：Problem2Output
    """
    # 步骤1：计算景点综合价值
    value_scores = {}
    for aid, attr in ATTRACTIONS.items():
        value_scores[aid] = calculate_attraction_value(aid, ATTRACTIONS)

    # 步骤2：优选景点组合
    selected, metrics = select_optimal_attractions(
        ATTRACTIONS,
        min_num=CONSTRAINTS['min_attractions'],
        max_num=CONSTRAINTS['max_attractions']
    )

    # 步骤3：按天分组
    day_groups = group_attractions_by_day(selected, ATTRACTIONS)

    # 步骤4：编排详细时序
    daily_plans = arrange_daily_schedule(day_groups, ATTRACTIONS)

    # 步骤5：计算评估指标
    total_preference = calculate_total_preference(selected, ATTRACTIONS)
    total_drive_time = calculate_total_drive_time(daily_plans)
    balance_score = calculate_balance_score(daily_plans)

    # 构建输出对象
    output = Problem2Output()
    output.selected_attractions = selected
    output.daily_plans = daily_plans
    output.total_preference = total_preference
    output.total_drive_time = total_drive_time
    output.balance_score = balance_score

    return output, value_scores, metrics, day_groups


# 主函数部分2：打印
def print_analysis_result(output, value_scores, metrics, day_groups):
    """
    打印分析结果（用于展示）
    """
    print("问题二：无随机扰动下的多目标景点优选与基准行程设计")

    # 【1. 景点综合价值评估】
    print("【1. 景点综合价值评估】")

    print(f"{'景点':<6} {'名称':<12} {'类型':<12} {'喜好度':<6} {'价值得分':<10}")


    sorted_scores = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)
    for aid, score in sorted_scores:
        attr = ATTRACTIONS[aid]
        print(f"{aid:<6} {attr['name']:<12} {attr['type']:<12} {attr['preference']:<6} {score:<10.3f}")

    # 【2. 景点优选结果】
    print("【2. 景点优选结果】")

    print(f"选中景点 ({len(output.selected_attractions)}个):")
    for aid in output.selected_attractions:
        attr = ATTRACTIONS[aid]
        print(f"  • {aid} {attr['name']} ({attr['type']}) - 喜好度: {attr['preference']}")

    print(f"组合评估指标:")
    print(f"总喜好度: {metrics['total_preference']:.1f}")
    print(f"平均组合内车程: {metrics['avg_combo_drive']:.2f}h")
    print(f"喜好度得分: {metrics['pref_score']:.3f}")
    print(f"通勤效率得分: {metrics['drive_score']:.3f}")
    print(f"行程可行性得分: {metrics['feasibility_score']:.3f}")

    # 【3. 景点分组】
    print("【3. 景点分组（每天1-2个）】")

    for day_idx, group in enumerate(day_groups, 1):
        if group:
            names = [f"{aid}({ATTRACTIONS[aid]['name']})" for aid in group]
            print(f"  第{day_idx}天: {' → '.join(names)}")
        else:
            print(f"  第{day_idx}天: (休整)")

    # 【4. 行程评估指标】
    print("【4. 行程评估指标】")

    print(f"总喜好度: {output.total_preference:.1f}")
    print(f"5天总行车时长: {output.total_drive_time:.2f}h")
    print(f"松紧均衡性评分: {output.balance_score:.3f}")

    # 【5. 详细时间轴】
    def print_timeline(daily_plans, attractions):
        """按时间轴拆解基准行程全部环节"""

        print("【5. 基准行程详细时间轴】")
        print("=" * 70)

        day_start = CONSTRAINTS['day_start']
        morning_prep = FIXED_TIME['morning_prep']
        meal_time = FIXED_TIME['meal']

        for plan in daily_plans:

            print(f"第 {plan.day} 天")


            current_time = day_start

            print(
                f"  {current_time:.1f}:00 - {current_time + morning_prep:.1f}:00  [起床+早餐+整装] ({morning_prep}h)")
            current_time += morning_prep

            print(f"  {current_time:.1f}:00  [从酒店出发]")

            drive1 = plan.drive_segments[0]
            print(
                f"  {current_time:.1f}:00 - {current_time + drive1:.1f}:00  [驾车前往 {attractions[plan.attractions[0][0]]['name']}] ({drive1}h)")
            current_time += drive1

            attr1_id, visit1_time = plan.attractions[0]
            attr1 = attractions[attr1_id]
            print(
                f"  {current_time:.1f}:00 - {current_time + visit1_time:.1f}:00  [游览 {attr1['name']}] ({visit1_time}h)")
            print(f"          └─ 开放时间: {attr1['open'][0]:.1f}:00 - {attr1['open'][1]:.1f}:00")
            current_time += visit1_time

            if 11.5 <= current_time <= 13.5:
                print(f"  {current_time:.1f}:00 - {current_time + meal_time:.1f}:00  [午餐] ({meal_time}h)")
                current_time += meal_time

            if len(plan.attractions) >= 2:
                drive2 = plan.drive_segments[1]
                print(
                    f"  {current_time:.1f}:00 - {current_time + drive2:.1f}:00  [驾车前往 {attractions[plan.attractions[1][0]]['name']}] ({drive2}h)")
                current_time += drive2

                attr2_id, visit2_time = plan.attractions[1]
                attr2 = attractions[attr2_id]
                print(
                    f"  {current_time:.1f}:00 - {current_time + visit2_time:.1f}:00  [游览 {attr2['name']}] ({visit2_time}h)")
                print(f"          └─ 开放时间: {attr2['open'][0]:.1f}:00 - {attr2['open'][1]:.1f}:00")
                current_time += visit2_time

                if 17.5 <= current_time <= 19.5:
                    print(f"  {current_time:.1f}:00 - {current_time + meal_time:.1f}:00  [晚餐] ({meal_time}h)")
                    current_time += meal_time

            drive_back = plan.drive_segments[-1]
            print(f"  {current_time:.1f}:00 - {current_time + drive_back:.1f}:00  [驾车返回酒店] ({drive_back}h)")
            current_time += drive_back

            print(f"  [当日结束时间] {current_time:.1f}:00")
            print(f"  [当日总时长] {current_time - day_start:.1f} 小时")

    print_timeline(output.daily_plans, ATTRACTIONS)

    print("" + "=" * 60)
    print("【基准行程方案总结】")

    print(f"选中景点: {', '.join(output.selected_attractions)} (共{len(output.selected_attractions)}个)")
    print(f"总喜好度: {output.total_preference:.1f}")
    print(f"总行车时长: {output.total_drive_time:.2f}小时")
    print(f"均衡性评分: {output.balance_score:.3f}")


# 主函数
def main():
    output, value_scores, metrics, day_groups = design_baseline_itinerary()
    print_analysis_result(output, value_scores, metrics, day_groups)

    return output

# =========================================
if __name__ == '__main__':
    main()
