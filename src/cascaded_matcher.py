"""
级联匹配引擎
=============
将深度学习初筛和细节点精确匹配组合成"级联匹配"流水线。

流程：
┌─────────────────┐
│  指纹图像 A, B  │
└────────┬────────┘
         │
    ┌────▼────┐
    │ 第一阶段 │  深度学习全局特征相似度（快速初筛）
    │ 阈值较低 │  → 不通过则直接判定不匹配（节省时间）
    └────┬────┘
         │ 通过
    ┌────▼────┐
    │ 第二阶段 │  细节点提取 + 精确拓扑匹配
    │ 阈值较高 │  → 给出最终确认和精确分数
    └────┬────┘
         │
    ┌────▼────┐
    │ 分数融合 │  加权融合两阶段分数，输出最终相似度
    └─────────┘
"""

import time
import numpy as np
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CASCADE, get_fingerprint_path
from src.preprocessing import FingerprintPreprocessor
from src.minutiae_extractor import MinutiaeExtractor
from src.minutiae_matcher import MinutiaeMatcher
from src.deep_feature import get_feature_extractor


class CascadedMatcher:
    """级联匹配引擎"""

    def __init__(self, config=None, prefer_deep=True):
        self.cfg = config or CASCADE
        self.preprocessor = FingerprintPreprocessor()
        self.minutiae_extractor = MinutiaeExtractor()
        self.minutiae_matcher = MinutiaeMatcher()
        self.feature_extractor = get_feature_extractor(prefer_deep=prefer_deep)

        # 缓存：避免重复提取
        self._deep_cache = {}       # path -> feature_vector
        self._minutiae_cache = {}   # path -> minutiae_list
        self._skeleton_cache = {}   # path -> (skeleton, mask, gray)

    # ----------------------------------------------------------
    # 核心匹配方法
    # ----------------------------------------------------------
    def match(self, path_a: str, path_b: str,
              verbose: bool = False) -> dict:
        """
        级联匹配两枚指纹。

        Args:
            path_a: 指纹A的图像路径
            path_b: 指纹B的图像路径
            verbose: 是否输出详细过程

        Returns:
            dict: {
                "final_score": float (0-1),
                "deep_score": float,
                "minutiae_score": float,
                "stage1_passed": bool,
                "matched_minutiae": int,
                "total_minutiae_a": int,
                "total_minutiae_b": int,
                "time_ms": float,
            }
        """
        start_time = time.time()

        result = {
            "path_a": path_a,
            "path_b": path_b,
            "final_score": 0.0,
            "deep_score": 0.0,
            "minutiae_score": 0.0,
            "stage1_passed": False,
            "matched_minutiae": 0,
            "total_minutiae_a": 0,
            "total_minutiae_b": 0,
            "time_ms": 0.0,
        }

        # ==========================
        # 第一阶段：深度学习初筛
        # ==========================
        if verbose:
            print(f"  [阶段1] 深度学习特征比对...")

        feat_a = self._get_deep_feature(path_a)
        feat_b = self._get_deep_feature(path_b)

        deep_score = self.feature_extractor.cosine_similarity(feat_a, feat_b)
        result["deep_score"] = deep_score

        if verbose:
            print(f"    深度学习相似度: {deep_score:.4f} "
                  f"(阈值: {self.cfg['stage1_threshold']})")

        # 初筛判断
        if deep_score < self.cfg["stage1_threshold"]:
            # 不通过初筛，直接返回低分
            result["final_score"] = deep_score * 0.3  # 给一个很低的分
            result["stage1_passed"] = False
            result["time_ms"] = (time.time() - start_time) * 1000
            if verbose:
                print(f"    ✗ 初筛未通过，跳过细节点匹配")
            return result

        result["stage1_passed"] = True
        if verbose:
            print(f"    ✓ 初筛通过，进入细节点精确匹配")

        # ==========================
        # 第二阶段：细节点精确匹配
        # ==========================
        if verbose:
            print(f"  [阶段2] 细节点提取与匹配...")

        minutiae_a = self._get_minutiae(path_a)
        minutiae_b = self._get_minutiae(path_b)

        result["total_minutiae_a"] = len(minutiae_a)
        result["total_minutiae_b"] = len(minutiae_b)

        if verbose:
            print(f"    指纹A: {len(minutiae_a)} 个细节点")
            print(f"    指纹B: {len(minutiae_b)} 个细节点")

        minutiae_result = self.minutiae_matcher.match(minutiae_a, minutiae_b)
        minutiae_score = minutiae_result["score"]
        result["minutiae_score"] = minutiae_score
        result["matched_minutiae"] = minutiae_result["matched_pairs"]

        if verbose:
            print(f"    细节点匹配分: {minutiae_score:.4f} "
                  f"(匹配 {minutiae_result['matched_pairs']} 对)")

        # ==========================
        # 分数融合
        # ==========================
        final_score = (self.cfg["deep_weight"] * deep_score +
                       self.cfg["minutiae_weight"] * minutiae_score)
        result["final_score"] = min(final_score, 1.0)

        result["time_ms"] = (time.time() - start_time) * 1000

        if verbose:
            print(f"  [最终分数] {result['final_score']:.4f} "
                  f"(深度: {deep_score:.3f} × {self.cfg['deep_weight']} + "
                  f"细节点: {minutiae_score:.3f} × {self.cfg['minutiae_weight']})")

        return result

    # ----------------------------------------------------------
    # 仅深度学习匹配（快速模式）
    # ----------------------------------------------------------
    def match_deep_only(self, path_a: str, path_b: str) -> float:
        """仅使用深度学习特征计算相似度"""
        feat_a = self._get_deep_feature(path_a)
        feat_b = self._get_deep_feature(path_b)
        return self.feature_extractor.cosine_similarity(feat_a, feat_b)

    # ----------------------------------------------------------
    # 仅细节点匹配
    # ----------------------------------------------------------
    def match_minutiae_only(self, path_a: str, path_b: str) -> float:
        """仅使用细节点匹配计算相似度"""
        minutiae_a = self._get_minutiae(path_a)
        minutiae_b = self._get_minutiae(path_b)
        result = self.minutiae_matcher.match(minutiae_a, minutiae_b)
        return result["score"]

    # ----------------------------------------------------------
    # 缓存辅助方法
    # ----------------------------------------------------------
    def _get_deep_feature(self, path: str) -> np.ndarray:
        if path not in self._deep_cache:
            self._deep_cache[path] = self.feature_extractor.extract_feature(path)
        return self._deep_cache[path]

    def _get_minutiae(self, path: str):
        if path not in self._minutiae_cache:
            skeleton, mask, gray = self._get_skeleton(path)
            minutiae = self.minutiae_extractor.extract(skeleton, mask, gray)
            self._minutiae_cache[path] = minutiae
        return self._minutiae_cache[path]

    def _get_skeleton(self, path: str):
        if path not in self._skeleton_cache:
            intermediates = self.preprocessor.process(path, return_intermediates=True)
            skeleton = intermediates["skeleton"]
            mask = intermediates["mask"]
            gray = intermediates["clahe"]
            self._skeleton_cache[path] = (skeleton, mask, gray)
        return self._skeleton_cache[path]

    # ----------------------------------------------------------
    # 清除缓存
    # ----------------------------------------------------------
    def clear_cache(self):
        """清除所有缓存"""
        self._deep_cache.clear()
        self._minutiae_cache.clear()
        self._skeleton_cache.clear()
