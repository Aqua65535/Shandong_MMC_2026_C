"""
问题一：景点特征分析与组合优先级研究

修改：将排队敏感和堵车敏感合并为"拥堵敏感度"维度
"""

import numpy as np
from data import ATTRACTIONS, DRIVE_MATRIX, TIME_PERIODS, Problem1Output

# ==================== 维度评分函数 ====================

def calculate_visit_time_score(attraction: dict) -> float:
    """游览耗时维度评分"""
    min_visit = attraction['min_visit']
    comfort_visit = attraction['comfort_visit']

    comfort_normalized = (comfort_visit - 2.5) / (5.0 - 2.5)
    comfort_normalized = max(0, min(1, comfort_normalized))

    elasticity = (comfort_visit - min_visit) / comfort_visit if comfort_visit > 0 else 0

    score = comfort_normalized * 0.6 + elasticity * 0.4
    return round(score, 3)


def calculate_commute_score(attraction_id: str) -> float:
    """通勤距离维度评分"""
    hotel_to_attr = DRIVE_MATRIX[('hotel', attraction_id)]

    other_distances = []
    for other in ATTRACTIONS.keys():
        if other != attraction_id:
            other_distances.append(DRIVE_MATRIX[(attraction_id, other)])
    avg_to_others = np.mean(other_distances) if other_distances else 0

    max_hotel_dist = 1.5
    max_avg_dist = 1.4

    hotel_score = 1 - (hotel_to_attr / max_hotel_dist)
    avg_score = 1 - (avg_to_others / max_avg_dist)

    score = hotel_score * 0.6 + avg_score * 0.4
    return round(max(0, min(1, score)), 3)


def calculate_jam_score(attraction: dict) -> float:
    """堵车敏感度评分（单独计算，用于合并）"""
    open_start, open_end = attraction['open']
    min_visit = attraction['min_visit']

    open_window = open_end - open_start
    if open_window <= 0:
        open_window = 24.0

    rush_periods = TIME_PERIODS['rush']
    rush_in_open = 0.0
    for rush_start, rush_end in rush_periods:
        overlap_start = max(open_start, rush_start)
        overlap_end = min(open_end, rush_end)
        if overlap_end > overlap_start:
            rush_in_open += (overlap_end - overlap_start)

    non_rush = open_window - rush_in_open
    rush_ratio = rush_in_open / open_window if open_window > 0 else 0

    need_time = min_visit + 0.5
    avoid_rush = min(1.0, non_rush / need_time) if need_time > 0 else 0

    sensitivity = rush_ratio * 0.5 + (1 - avoid_rush) * 0.5
    score = 1 - sensitivity
    return round(score, 3)


def calculate_queue_score(attraction: dict) -> float:
    """排队敏感度评分（单独计算，用于合并）"""
    open_start, open_end = attraction['open']
    min_visit = attraction['min_visit']

    rush_entry_start, rush_entry_end = 9.0, 12.0
    latest_entry = open_end - min_visit

    if open_end <= rush_entry_start or latest_entry <= rush_entry_start:
        return 1.0
    if open_start >= rush_entry_end:
        return 1.0

    feasible_start = open_start
    feasible_end = latest_entry

    if feasible_end <= feasible_start:
        return 0.5

    overlap_start = max(feasible_start, rush_entry_start)
    overlap_end = min(feasible_end, rush_entry_end)
    overlap_hours = max(0, overlap_end - overlap_start)

    total_feasible = feasible_end - feasible_start

    forced_rush_prob = overlap_hours / total_feasible if total_feasible > 0 else 1.0

    return round(1 - forced_rush_prob, 3)


def calculate_congestion_score(attraction: dict) -> float:
    """
    拥堵敏感度维度评分（合并堵车敏感 + 排队敏感）
    权重：堵车敏感 0.6，排队敏感 0.4
    """
    jam = calculate_jam_score(attraction)
    queue = calculate_queue_score(attraction)

    # 合并权重：堵车影响更大（路况不确定性高）
    score = jam * 0.6 + queue * 0.4
    return round(score, 3)


def calculate_preference_score(attraction: dict) -> float:
    """喜好度维度评分"""
    preference = attraction['preference']
    normalized = (preference - 6.5) / (9.5 - 6.5)
    return round(normalized, 3)


# ==================== 联动分析函数 ====================

def calculate_linkage_score(attr1_id: str, attr2_id: str, attractions: dict) -> float:
    """计算两个景点之间的联动评分"""
    attr1 = attractions[attr1_id]
    attr2 = attractions[attr2_id]

    drive_time = DRIVE_MATRIX[(attr1_id, attr2_id)]
    geo_score = 1 - (drive_time / 1.5)
    geo_score = max(0, min(1, geo_score))

    open1_start, open1_end = attr1['open']
    open2_start, open2_end = attr2['open']

    overlap_start = max(open1_start, open2_start)
    overlap_end = min(open1_end, open2_end)
    overlap_hours = max(0, overlap_end - overlap_start)

    min_total = attr1['min_visit'] + attr2['min_visit']
    window_len = overlap_hours
    need_time = min_total + drive_time + 0.5
    time_compatible = min(1.0, window_len / need_time)

    type_complement = 1.0 if attr1['type'] != attr2['type'] else 0.5

    hotel_to_1 = DRIVE_MATRIX[('hotel', attr1_id)]
    hotel_to_2 = DRIVE_MATRIX[('hotel', attr2_id)]
    commute_efficiency = 1 - abs(hotel_to_1 - hotel_to_2) / 1.5

    total_score = (geo_score * 0.4 +
                   time_compatible * 0.3 +
                   type_complement * 0.15 +
                   commute_efficiency * 0.15)

    return round(total_score, 3)


def find_linkage_pairs(attractions: dict, top_n: int = 15) -> list:
    """挖掘可联动的景点组合"""
    pairs = []
    attraction_ids = list(attractions.keys())

    for i in range(len(attraction_ids)):
        for j in range(i + 1, len(attraction_ids)):
            attr1_id = attraction_ids[i]
            attr2_id = attraction_ids[j]

            linkage_score = calculate_linkage_score(attr1_id, attr2_id, attractions)

            if linkage_score > 0.3:
                pairs.append((attr1_id, attr2_id, linkage_score))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


# ==================== 综合优先级计算（四维） ====================

def calculate_comprehensive_priority(attractions: dict) -> list:
    """
    计算综合优先级得分
    四维度加权：喜好度(0.40) + 游览耗时(0.25) + 通勤距离(0.20) + 拥堵敏感度(0.15)
    """
    results = []

    for attr_id, attr in attractions.items():
        # 计算各维度得分
        visit_score = calculate_visit_time_score(attr)      # 游览耗时
        commute_score = calculate_commute_score(attr_id)    # 通勤距离
        congestion_score = calculate_congestion_score(attr) # 拥堵敏感（合并后）
        preference_score = calculate_preference_score(attr) # 喜好度

        # 单独保留子维度（用于详细展示，但不参与综合评分）
        jam_score = calculate_jam_score(attr)               # 堵车敏感（记录用）
        queue_score = calculate_queue_score(attr)           # 排队敏感（记录用）

        # 四维加权综合得分（权重和为1）
        comprehensive = (preference_score * 0.40 +
                         visit_score * 0.25 +
                         commute_score * 0.20 +
                         congestion_score * 0.15)

        results.append({
            'id': attr_id,
            'name': attr['name'],
            'type': attr['type'],
            # 四维主维度
            'preference_score': preference_score,   # 喜好度
            'visit_score': visit_score,             # 游览耗时
            'commute_score': commute_score,         # 通勤距离
            'congestion_score': congestion_score,   # 拥堵敏感度（合并）
            # 保留的子维度（仅用于展示分析）
            'jam_score': jam_score,                 # 堵车敏感（子维度）
            'queue_score': queue_score,             # 排队敏感（子维度）
            # 综合得分
            'comprehensive_score': comprehensive
        })

    results.sort(key=lambda x: x['comprehensive_score'], reverse=True)
    return results


# ==================== 主函数 ====================

def analyze_attractions() -> Problem1Output:
    """分析景点特征，返回结果"""
    priority_results = calculate_comprehensive_priority(ATTRACTIONS)
    priority_scores = {r['id']: r['comprehensive_score'] for r in priority_results}

    linkage_pairs = find_linkage_pairs(ATTRACTIONS)
    candidate_pool = list(ATTRACTIONS.keys())

    output = Problem1Output()
    output.priority_scores = priority_scores
    output.linkage_pairs = linkage_pairs
    output.candidate_pool = candidate_pool

    return output


def print_analysis_result(result: Problem1Output) -> None:
    """打印分析结果"""
    print("问题一：景点特征分析与组合优先级研究（四维模型）")


    print("【1. 综合优先级排名（四维：喜好+游览+通勤+拥堵）】")

    print(f"{'排名':<4} {'景点编号':<8} {'景点名称':<12} {'类型':<12} {'综合得分':<10}")


    priority_full = calculate_comprehensive_priority(ATTRACTIONS)
    for rank, item in enumerate(priority_full, 1):
        print(f"{rank:<4} {item['id']:<8} {item['name']:<12} {item['type']:<12} {item['comprehensive_score']:<10.3f}")

    print("【2. 四维度得分详情】")

    print(f"{'景点':<8} {'喜好度':<10} {'游览耗时':<10} {'通勤距离':<10} {'拥堵敏感':<10} {'综合':<8}")

    for item in priority_full:
        print(f"{item['id']:<8} {item['preference_score']:<10.3f} {item['visit_score']:<10.3f} "
              f"{item['commute_score']:<10.3f} {item['congestion_score']:<10.3f} "
              f"{item['comprehensive_score']:<8.3f}")

    print("【3. 拥堵敏感度子维度拆解（堵车+排队）】")

    print(f"{'景点':<8} {'景点名称':<12} {'堵车敏感':<10} {'排队敏感':<10} {'拥堵综合':<10}")

    for item in priority_full:
        print(f"{item['id']:<8} {item['name']:<12} {item['jam_score']:<10.3f} "
              f"{item['queue_score']:<10.3f} {item['congestion_score']:<10.3f}")

    print("【4. 高优先级景点推荐（前5）】")

    high_priority = priority_full[:5]
    for item in high_priority:
        print(f"• {item['id']} {item['name']} ({item['type']}) - 综合得分: {item['comprehensive_score']:.3f}")
        print(f"  推荐理由: 喜好度{item['preference_score']:.2f}, 通勤{item['commute_score']:.2f}, "
              f"拥堵敏感{item['congestion_score']:.2f}")

    print("【5. 低优先级景点（后3）】")

    low_priority = priority_full[-3:]
    for item in low_priority:
        print(f"• {item['id']} {item['name']} ({item['type']}) - 综合得分: {item['comprehensive_score']:.3f}")

    print("【6. 可联动游玩组合推荐】")

    print(f"{'排名':<4} {'组合':<12} {'联动评分':<10} {'推荐理由'}")


    for rank, (a1, a2, score) in enumerate(result.linkage_pairs[:10], 1):
        name1 = ATTRACTIONS[a1]['name']
        name2 = ATTRACTIONS[a2]['name']
        drive = DRIVE_MATRIX[(a1, a2)]
        type1 = ATTRACTIONS[a1]['type']
        type2 = ATTRACTIONS[a2]['type']

        reason_parts = []
        if drive < 0.5:
            reason_parts.append(f"车程仅{drive}h")
        if type1 != type2:
            reason_parts.append("类型互补")
        if score > 0.90:
            reason_parts.append("高度适配")

        reason = "、".join(reason_parts) if reason_parts else "基础可联"

        print(f"{rank:<4} {a1}-{a2:<7} {score:<10.3f} {reason}")
        print(f"      ({name1} ↔ {name2})")

    print("【7. 备选景点池】")
    print(f"共 {len(result.candidate_pool)} 个景点: {', '.join(result.candidate_pool)}")


def main() -> Problem1Output:
    result = analyze_attractions()
    print_analysis_result(result)
    return result


if __name__ == '__main__':
    main()