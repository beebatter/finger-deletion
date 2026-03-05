"""
指纹数据库管理模块
===================
管理指纹特征的存储、检索和扩展。
支持：
- 注册新指纹（提取并存储特征）
- 1:1 验证（两枚指纹比对）
- 1:N 识别（在库中搜索）
- 数据库导入/导出
"""

import json
import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_DIR


class FingerprintDatabase:
    """指纹特征数据库"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(DB_DIR, "fingerprint_db.pkl")
        self.entries: Dict[str, dict] = {}  # id -> entry
        self._load()

    # ----------------------------------------------------------
    # 注册
    # ----------------------------------------------------------
    def register(self, fingerprint_id: str, image_path: str,
                 deep_feature: np.ndarray = None,
                 minutiae: list = None,
                 metadata: dict = None):
        """
        注册一枚指纹到数据库。

        Args:
            fingerprint_id: 唯一标识 (如 "001", "person_A_thumb")
            image_path: 原始图像路径
            deep_feature: 深度学习特征向量
            minutiae: 细节点列表（序列化后的）
            metadata: 附加信息 (采集方式、手指、日期等)
        """
        entry = {
            "id": fingerprint_id,
            "image_path": image_path,
            "deep_feature": deep_feature,
            "minutiae": minutiae,
            "metadata": metadata or {},
        }
        self.entries[fingerprint_id] = entry
        self._save()

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------
    def get(self, fingerprint_id: str) -> Optional[dict]:
        """获取一条记录"""
        return self.entries.get(fingerprint_id)

    def get_all_ids(self) -> List[str]:
        """获取所有已注册的指纹ID"""
        return list(self.entries.keys())

    def get_by_method(self, method: str) -> List[dict]:
        """按采集方式查询"""
        return [e for e in self.entries.values()
                if e.get("metadata", {}).get("method") == method]

    def get_deep_features_matrix(self, ids: List[str] = None) -> Tuple[List[str], np.ndarray]:
        """
        获取指定指纹的深度特征矩阵。

        Returns:
            (id_list, feature_matrix)
        """
        if ids is None:
            ids = self.get_all_ids()

        valid_ids = []
        features = []
        for fid in ids:
            entry = self.entries.get(fid)
            if entry and entry.get("deep_feature") is not None:
                valid_ids.append(fid)
                features.append(entry["deep_feature"])

        if not features:
            return [], np.array([])

        return valid_ids, np.vstack(features)

    # ----------------------------------------------------------
    # 1:N 搜索
    # ----------------------------------------------------------
    def search(self, query_feature: np.ndarray,
               top_k: int = 5,
               method_filter: str = None) -> List[dict]:
        """
        在数据库中搜索最相似的指纹。

        Args:
            query_feature: 查询指纹的深度特征向量
            top_k: 返回前 k 个结果
            method_filter: 只搜索指定采集方式的指纹

        Returns:
            排序后的相似结果列表
        """
        results = []
        for fid, entry in self.entries.items():
            if method_filter and entry.get("metadata", {}).get("method") != method_filter:
                continue

            feat = entry.get("deep_feature")
            if feat is None:
                continue

            # 余弦相似度
            sim = float(np.dot(query_feature, feat) /
                        (np.linalg.norm(query_feature) * np.linalg.norm(feat) + 1e-8))
            sim = (sim + 1) / 2  # 映射到 [0, 1]

            results.append({
                "id": fid,
                "similarity": sim,
                "image_path": entry["image_path"],
                "metadata": entry.get("metadata", {}),
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]

    # ----------------------------------------------------------
    # 统计
    # ----------------------------------------------------------
    def stats(self) -> dict:
        """数据库统计信息"""
        total = len(self.entries)
        with_deep = sum(1 for e in self.entries.values()
                        if e.get("deep_feature") is not None)
        with_minutiae = sum(1 for e in self.entries.values()
                            if e.get("minutiae") is not None)

        methods = {}
        for e in self.entries.values():
            m = e.get("metadata", {}).get("method", "unknown")
            methods[m] = methods.get(m, 0) + 1

        return {
            "total": total,
            "with_deep_features": with_deep,
            "with_minutiae": with_minutiae,
            "methods": methods,
        }

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------
    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.entries, f)

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.entries = pickle.load(f)

    def export_json(self, path: str):
        """导出数据库元信息为 JSON（不含特征向量）"""
        data = {}
        for fid, entry in self.entries.items():
            data[fid] = {
                "id": fid,
                "image_path": entry["image_path"],
                "has_deep_feature": entry.get("deep_feature") is not None,
                "minutiae_count": len(entry.get("minutiae", []) or []),
                "metadata": entry.get("metadata", {}),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def clear(self):
        """清空数据库"""
        self.entries.clear()
        self._save()
