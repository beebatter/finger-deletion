"""
细节点匹配模块
===============
基于细节点的空间拓扑关系进行指纹比对。
核心算法：
1. 全局对齐（旋转 + 平移估计）
2. 贪心匹配 + 一致性验证
3. 匹配分数计算
"""

import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MINUTIAE_MATCH
from src.minutiae_extractor import MinutiaePoint


class MinutiaeMatcher:
    """基于细节点的指纹匹配器"""

    def __init__(self, config=None):
        self.cfg = config or MINUTIAE_MATCH

    def match(self, minutiae_a: List[MinutiaePoint],
              minutiae_b: List[MinutiaePoint]) -> dict:
        """
        对两组细节点进行匹配。

        Returns:
            dict: {
                "score": float (0-1),
                "matched_pairs": int,
                "total_a": int,
                "total_b": int,
                "details": list of matched pairs
            }
        """
        if len(minutiae_a) < 3 or len(minutiae_b) < 3:
            return {
                "score": 0.0,
                "matched_pairs": 0,
                "total_a": len(minutiae_a),
                "total_b": len(minutiae_b),
                "details": [],
            }

        # 尝试多种对齐方案，取最优
        best_score = 0.0
        best_result = None

        # 使用每对可能的锚点进行对齐尝试
        # 为了效率，选取质量最高的若干点作为锚点
        anchors_a = self._select_anchors(minutiae_a, max_count=8)
        anchors_b = self._select_anchors(minutiae_b, max_count=8)

        for ia in anchors_a:
            for ib in anchors_b:
                # 只对相同类型的点尝试对齐
                if minutiae_a[ia].type != minutiae_b[ib].type:
                    continue

                # 估计变换参数
                dx, dy, dtheta = self._estimate_transform(
                    minutiae_a[ia], minutiae_b[ib]
                )

                # 用这组变换参数匹配所有点
                result = self._match_with_transform(
                    minutiae_a, minutiae_b, dx, dy, dtheta
                )

                if result["score"] > best_score:
                    best_score = result["score"]
                    best_result = result

        if best_result is None:
            return {
                "score": 0.0,
                "matched_pairs": 0,
                "total_a": len(minutiae_a),
                "total_b": len(minutiae_b),
                "details": [],
            }

        return best_result

    # ----------------------------------------------------------
    # 锚点选择
    # ----------------------------------------------------------
    def _select_anchors(self, minutiae: List[MinutiaePoint],
                        max_count: int = 8) -> List[int]:
        """选取质量最高的若干点作为对齐锚点"""
        indexed = [(i, m.quality) for i, m in enumerate(minutiae)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in indexed[:max_count]]

    # ----------------------------------------------------------
    # 变换估计
    # ----------------------------------------------------------
    def _estimate_transform(self, pt_a: MinutiaePoint,
                            pt_b: MinutiaePoint) -> Tuple[float, float, float]:
        """估计从B到A的刚性变换（平移 + 旋转）"""
        dtheta = pt_a.angle - pt_b.angle
        # 将角度归一化到 [-pi, pi]
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta < -np.pi:
            dtheta += 2 * np.pi

        # 旋转B的坐标后计算平移
        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)
        rotated_bx = pt_b.x * cos_t - pt_b.y * sin_t
        rotated_by = pt_b.x * sin_t + pt_b.y * cos_t

        dx = pt_a.x - rotated_bx
        dy = pt_a.y - rotated_by

        return dx, dy, dtheta

    # ----------------------------------------------------------
    # 带变换的匹配
    # ----------------------------------------------------------
    def _match_with_transform(self, minutiae_a: List[MinutiaePoint],
                               minutiae_b: List[MinutiaePoint],
                               dx: float, dy: float,
                               dtheta: float) -> dict:
        """将变换应用到 B 集合后，与 A 集合进行贪心匹配"""
        dist_thresh = self.cfg["distance_threshold"]
        angle_thresh = self.cfg["angle_threshold"]

        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)

        # 变换 B 的所有点
        transformed_b = []
        for pb in minutiae_b:
            new_x = pb.x * cos_t - pb.y * sin_t + dx
            new_y = pb.x * sin_t + pb.y * cos_t + dy
            new_angle = pb.angle + dtheta
            transformed_b.append((new_x, new_y, new_angle, pb.type))

        # 计算距离矩阵
        n_a = len(minutiae_a)
        n_b = len(transformed_b)
        cost_matrix = np.full((n_a, n_b), float("inf"))

        for i, pa in enumerate(minutiae_a):
            for j, (bx, by, bangle, btype) in enumerate(transformed_b):
                # 类型必须一致
                if pa.type != btype:
                    continue

                # 空间距离
                spatial_dist = np.sqrt((pa.x - bx) ** 2 + (pa.y - by) ** 2)
                if spatial_dist > dist_thresh:
                    continue

                # 角度差
                angle_diff = abs(pa.angle - bangle)
                while angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                if angle_diff > angle_thresh:
                    continue

                # 综合代价
                cost_matrix[i, j] = spatial_dist + 10 * angle_diff

        # 贪心匹配（按代价排序，优先匹配最佳对）
        matched_pairs = []
        used_a = set()
        used_b = set()

        # 获取所有有效候选
        candidates = []
        for i in range(n_a):
            for j in range(n_b):
                if cost_matrix[i, j] < float("inf"):
                    candidates.append((cost_matrix[i, j], i, j))

        candidates.sort()

        for cost, i, j in candidates:
            if i not in used_a and j not in used_b:
                matched_pairs.append({
                    "a_idx": i,
                    "b_idx": j,
                    "distance": cost,
                    "type": minutiae_a[i].type,
                })
                used_a.add(i)
                used_b.add(j)

        # 计算匹配分数
        n_matched = len(matched_pairs)
        if n_matched < self.cfg["min_matched_points"]:
            score = n_matched / max(self.cfg["min_matched_points"], 1) * 0.3
        else:
            # 基础分：匹配比例
            ratio = n_matched / max(min(n_a, n_b), 1)
            # 一致性分：平均代价取反归一化
            if matched_pairs:
                avg_cost = np.mean([p["distance"] for p in matched_pairs])
                consistency = max(0, 1 - avg_cost / (dist_thresh + 10 * angle_thresh))
            else:
                consistency = 0

            score = 0.6 * ratio + 0.4 * consistency
            score = min(score, 1.0)

        return {
            "score": score,
            "matched_pairs": n_matched,
            "total_a": n_a,
            "total_b": n_b,
            "details": matched_pairs,
        }


def match_minutiae(minutiae_a: List[MinutiaePoint],
                   minutiae_b: List[MinutiaePoint]) -> dict:
    """便捷函数：匹配两组细节点"""
    matcher = MinutiaeMatcher()
    return matcher.match(minutiae_a, minutiae_b)
