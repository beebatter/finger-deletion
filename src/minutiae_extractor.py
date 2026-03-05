"""
细节点提取模块
===============
从骨架化后的指纹图中提取端点（Ridge Ending）和分叉点（Bifurcation）。
包含多层伪细节点过滤机制：
1. 边界排除
2. 最小间距过滤
3. 局部对比度验证
4. ROI 掩码验证
5. 数量上限控制
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MINUTIAE


@dataclass
class MinutiaePoint:
    """表示一个细节点特征"""
    x: int           # 列坐标
    y: int           # 行坐标
    angle: float     # 方向（弧度）
    type: str        # "ending" 或 "bifurcation"
    quality: float   # 质量分（0-1）

    def to_dict(self):
        return {
            "x": self.x, "y": self.y,
            "angle": self.angle,
            "type": self.type,
            "quality": self.quality,
        }


class MinutiaeExtractor:
    """基于骨架的细节点提取器（含伪特征过滤）"""

    def __init__(self, config=None):
        self.cfg = config or MINUTIAE

    def extract(self, skeleton: np.ndarray, mask: np.ndarray,
                original_gray: np.ndarray = None) -> List[MinutiaePoint]:
        """
        从骨架图中提取细节点。

        Args:
            skeleton: 骨架化后的二值图 (0/255)
            mask: ROI 掩码 (0/255)
            original_gray: 原始灰度图（用于质量评估）

        Returns:
            过滤后的细节点列表
        """
        skel_bool = skeleton > 0
        h, w = skeleton.shape

        # 3x3 邻域交叉数（Crossing Number）检测
        raw_points = self._crossing_number_detect(skel_bool, h, w)

        # 多层过滤
        filtered = self._filter_border(raw_points, h, w)
        filtered = self._filter_by_mask(filtered, mask)
        filtered = self._filter_min_distance(filtered)

        if original_gray is not None:
            filtered = self._filter_by_contrast(filtered, original_gray)

        # 计算每个点的方向
        for pt in filtered:
            pt.angle = self._estimate_direction(skel_bool, pt.y, pt.x)

        # 质量评估
        if original_gray is not None:
            for pt in filtered:
                pt.quality = self._assess_quality(original_gray, pt.y, pt.x)
        else:
            for pt in filtered:
                pt.quality = 0.5  # 默认中等质量

        # 数量上限
        if len(filtered) > self.cfg["max_minutiae_count"]:
            filtered.sort(key=lambda p: p.quality, reverse=True)
            filtered = filtered[: self.cfg["max_minutiae_count"]]

        return filtered

    # ----------------------------------------------------------
    # 交叉数检测
    # ----------------------------------------------------------
    def _crossing_number_detect(self, skel: np.ndarray,
                                 h: int, w: int) -> List[MinutiaePoint]:
        """
        交叉数（Crossing Number, CN）算法。
        CN = 0.5 * |P1-P2| + |P2-P3| + ... + |P8-P1|
        CN=1 -> 端点, CN=3 -> 分叉点
        """
        points = []
        # 8邻域偏移（顺时针）
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, 1), (1, 1), (1, 0),
                   (1, -1), (0, -1)]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not skel[y, x]:
                    continue

                # 获取 8 邻域值
                neighbors = []
                for dy, dx in offsets:
                    neighbors.append(1 if skel[y + dy, x + dx] else 0)

                # 计算交叉数
                cn = 0
                for i in range(8):
                    cn += abs(neighbors[i] - neighbors[(i + 1) % 8])
                cn //= 2

                if cn == 1:
                    points.append(MinutiaePoint(
                        x=x, y=y, angle=0.0, type="ending", quality=0.0
                    ))
                elif cn == 3:
                    points.append(MinutiaePoint(
                        x=x, y=y, angle=0.0, type="bifurcation", quality=0.0
                    ))

        return points

    # ----------------------------------------------------------
    # 过滤层1：边界排除
    # ----------------------------------------------------------
    def _filter_border(self, points: List[MinutiaePoint],
                       h: int, w: int) -> List[MinutiaePoint]:
        """排除图像边缘附近的伪特征"""
        margin = self.cfg["border_margin"]
        return [p for p in points
                if margin < p.x < w - margin and margin < p.y < h - margin]

    # ----------------------------------------------------------
    # 过滤层2：ROI 掩码验证
    # ----------------------------------------------------------
    def _filter_by_mask(self, points: List[MinutiaePoint],
                        mask: np.ndarray) -> List[MinutiaePoint]:
        """排除 ROI 外部和掩码边缘的点"""
        margin = 5
        kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
        eroded_mask = cv2.erode(mask, kernel)
        return [p for p in points if eroded_mask[p.y, p.x] > 0]

    # ----------------------------------------------------------
    # 过滤层3：最小间距过滤（消除密集伪特征簇）
    # ----------------------------------------------------------
    def _filter_min_distance(self, points: List[MinutiaePoint]) -> List[MinutiaePoint]:
        """
        对于距离过近的细节点对，保留一个更可靠的。
        密集的伪特征通常来自局部噪声引起的毛刺。
        """
        min_dist = self.cfg["min_distance"]
        keep = [True] * len(points)

        for i in range(len(points)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(points)):
                if not keep[j]:
                    continue
                dist = np.sqrt((points[i].x - points[j].x) ** 2 +
                               (points[i].y - points[j].y) ** 2)
                if dist < min_dist:
                    # 如果两个点类型不同（一个端点一个分叉），可能是伪特征对
                    if points[i].type != points[j].type:
                        keep[i] = False
                        keep[j] = False
                    else:
                        # 保留后一个（质量未计算时随机保留一个）
                        keep[j] = False

        return [p for p, k in zip(points, keep) if k]

    # ----------------------------------------------------------
    # 过滤层4：局部对比度验证
    # ----------------------------------------------------------
    def _filter_by_contrast(self, points: List[MinutiaePoint],
                            gray: np.ndarray) -> List[MinutiaePoint]:
        """
        在原始灰度图上检查细节点周围的对比度。
        真正的脊线特征应该有明显的明暗交替。
        """
        min_contrast = self.cfg["min_local_contrast"]
        result = []
        for p in points:
            patch = gray[max(0, p.y - 8): p.y + 9, max(0, p.x - 8): p.x + 9]
            if patch.size > 0:
                contrast = np.max(patch.astype(float)) - np.min(patch.astype(float))
                if contrast >= min_contrast:
                    result.append(p)
        return result

    # ----------------------------------------------------------
    # 方向估计
    # ----------------------------------------------------------
    def _estimate_direction(self, skel: np.ndarray, y: int, x: int,
                            radius: int = 8) -> float:
        """
        沿骨架线跟踪一小段，估计细节点方向。
        """
        h, w = skel.shape
        # 简化：在半径内搜索骨架像素，计算加权平均方向
        dy_sum, dx_sum = 0.0, 0.0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    weight = 1.0 / (abs(dy) + abs(dx) + 1)
                    dy_sum += dy * weight
                    dx_sum += dx * weight

        return np.arctan2(dy_sum, dx_sum)

    # ----------------------------------------------------------
    # 质量评估
    # ----------------------------------------------------------
    def _assess_quality(self, gray: np.ndarray, y: int, x: int,
                        radius: int = 12) -> float:
        """
        基于局部图像质量评估细节点的可信度（0-1）。
        综合考虑对比度、清晰度。
        """
        patch = gray[max(0, y - radius): y + radius + 1,
                     max(0, x - radius): x + radius + 1]
        if patch.size < 4:
            return 0.0

        # 对比度分（归一化）
        contrast = (np.max(patch.astype(float)) - np.min(patch.astype(float))) / 255.0

        # 清晰度分（Laplacian 方差）
        laplacian = cv2.Laplacian(patch, cv2.CV_64F)
        sharpness = min(np.var(laplacian) / 500.0, 1.0)

        return 0.5 * contrast + 0.5 * sharpness

    # ----------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------
    @staticmethod
    def visualize(image: np.ndarray, minutiae: List[MinutiaePoint],
                  save_path: str = None) -> np.ndarray:
        """
        在图像上绘制细节点。
        红色圆圈 = 端点, 蓝色圆圈 = 分叉点
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for pt in minutiae:
            color = (0, 0, 255) if pt.type == "ending" else (255, 0, 0)
            radius = max(3, int(pt.quality * 6) + 2)
            cv2.circle(vis, (pt.x, pt.y), radius, color, 1)
            # 绘制方向线
            end_x = int(pt.x + 12 * np.cos(pt.angle))
            end_y = int(pt.y + 12 * np.sin(pt.angle))
            cv2.line(vis, (pt.x, pt.y), (end_x, end_y), color, 1)

        if save_path:
            cv2.imwrite(save_path, vis)

        return vis
