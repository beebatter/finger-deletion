"""
指纹识别系统 - 主评估脚本
===========================
核心功能：
1. 构建指纹数据库（注册所有指纹）
2. 方式A内部比对（基准性能）
3. 方式B内部比对
4. 跨方式 A↔B 比对（核心目标）
5. 生成评估报告

用法：
    python evaluate.py                   # 完整评估
    python evaluate.py --mode quick      # 仅深度学习快速评估
    python evaluate.py --mode minutiae   # 仅细节点评估
    python evaluate.py --mode cascade    # 级联匹配评估（默认）
    python evaluate.py --pair 1 42       # 单对比对
"""

import argparse
import os
import sys
import time
import json
import numpy as np
from itertools import combinations

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    DATA_DIR, OUTPUT_DIR, METHOD_A_RANGE, METHOD_B_RANGE,
    get_fingerprint_path, EVALUATION
)
from src.preprocessing import FingerprintPreprocessor
from src.minutiae_extractor import MinutiaeExtractor
from src.minutiae_matcher import MinutiaeMatcher
from src.deep_feature import get_feature_extractor
from src.cascaded_matcher import CascadedMatcher
from src.database import FingerprintDatabase


def get_all_paths(start: int, end: int):
    """安全获取指定范围的指纹路径列表"""
    paths = []
    for i in range(start, end + 1):
        try:
            paths.append((i, get_fingerprint_path(i)))
        except FileNotFoundError:
            print(f"  [警告] 编号 {i:03d} 文件不存在，跳过")
    return paths


def build_database():
    """步骤 1: 构建指纹数据库"""
    print("=" * 60)
    print("步骤 1: 构建指纹数据库")
    print("=" * 60)

    db = FingerprintDatabase()
    db.clear()

    extractor = get_feature_extractor(prefer_deep=True)

    # 注册方式 A
    a_start, a_end = METHOD_A_RANGE
    print(f"\n注册方式A指纹 ({a_start:03d} ~ {a_end:03d})...")
    a_paths = get_all_paths(a_start, a_end)
    for idx, path in a_paths:
        feat = extractor.extract_feature(path)
        db.register(
            fingerprint_id=f"{idx:03d}",
            image_path=path,
            deep_feature=feat,
            metadata={"method": "A", "index": idx},
        )
        print(f"  ✓ {idx:03d} 注册成功")

    # 注册方式 B
    b_start, b_end = METHOD_B_RANGE
    print(f"\n注册方式B指纹 ({b_start:03d} ~ {b_end:03d})...")
    b_paths = get_all_paths(b_start, b_end)
    for idx, path in b_paths:
        feat = extractor.extract_feature(path)
        db.register(
            fingerprint_id=f"{idx:03d}",
            image_path=path,
            deep_feature=feat,
            metadata={"method": "B", "index": idx},
        )
        print(f"  ✓ {idx:03d} 注册成功")

    stats = db.stats()
    print(f"\n数据库统计: {stats}")
    return db


def evaluate_within_group(matcher: CascadedMatcher, paths: list,
                          group_name: str, mode: str = "cascade") -> dict:
    """评估同一组内部的匹配分数（组内一致性）"""
    print(f"\n{'—' * 50}")
    print(f"评估: {group_name} 内部一致性")
    print(f"{'—' * 50}")

    if len(paths) < 2:
        print("  样本不足，跳过")
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "scores": []}

    scores = []
    pairs = list(combinations(paths, 2))
    total = len(pairs)

    for count, ((idx_a, path_a), (idx_b, path_b)) in enumerate(pairs, 1):
        if mode == "quick":
            score = matcher.match_deep_only(path_a, path_b)
        elif mode == "minutiae":
            score = matcher.match_minutiae_only(path_a, path_b)
        else:
            result = matcher.match(path_a, path_b, verbose=False)
            score = result["final_score"]

        scores.append(score)
        if count % 50 == 0 or count == total:
            print(f"  进度: {count}/{total}, 当前均值: {np.mean(scores):.4f}")

    stats = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "count": len(scores),
        "scores": [float(s) for s in scores],
    }

    print(f"  均值: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  中位数: {stats['median']:.4f}")
    return stats


def evaluate_cross_group(matcher: CascadedMatcher,
                         paths_a: list, paths_b: list,
                         mode: str = "cascade") -> dict:
    """评估跨组匹配分数（A ↔ B 核心比对）"""
    print(f"\n{'—' * 50}")
    print(f"评估: 方式A ↔ 方式B 跨组匹配")
    print(f"{'—' * 50}")

    scores = []
    details = []
    total = len(paths_a) * len(paths_b)
    count = 0

    for idx_a, path_a in paths_a:
        for idx_b, path_b in paths_b:
            count += 1
            if mode == "quick":
                score = matcher.match_deep_only(path_a, path_b)
                detail = {"a": idx_a, "b": idx_b, "score": score}
            elif mode == "minutiae":
                score = matcher.match_minutiae_only(path_a, path_b)
                detail = {"a": idx_a, "b": idx_b, "score": score}
            else:
                result = matcher.match(path_a, path_b, verbose=False)
                score = result["final_score"]
                detail = {
                    "a": idx_a, "b": idx_b,
                    "final_score": result["final_score"],
                    "deep_score": result["deep_score"],
                    "minutiae_score": result["minutiae_score"],
                    "stage1_passed": result["stage1_passed"],
                    "matched_minutiae": result["matched_minutiae"],
                }

            scores.append(score)
            details.append(detail)

            if count % 20 == 0 or count == total:
                print(f"  进度: {count}/{total}, 当前均值: {np.mean(scores):.4f}")

    stats = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "count": len(scores),
        "details": details,
        "scores": [float(s) for s in scores],
    }

    print(f"  均值: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  中位数: {stats['median']:.4f}")
    return stats


def single_pair_match(idx_a: int, idx_b: int):
    """单对指纹详细比对"""
    print("=" * 60)
    print(f"单对比对: {idx_a:03d} vs {idx_b:03d}")
    print("=" * 60)

    path_a = get_fingerprint_path(idx_a)
    path_b = get_fingerprint_path(idx_b)

    matcher = CascadedMatcher(prefer_deep=True)
    result = matcher.match(path_a, path_b, verbose=True)

    print(f"\n{'=' * 40}")
    print(f"最终相似度: {result['final_score']:.4f} ({result['final_score']*100:.1f}%)")
    print(f"深度学习分: {result['deep_score']:.4f}")
    print(f"细节点分:   {result['minutiae_score']:.4f}")
    print(f"初筛通过:   {'是' if result['stage1_passed'] else '否'}")
    print(f"匹配细节点: {result['matched_minutiae']} 对")
    print(f"耗时:       {result['time_ms']:.1f} ms")
    print(f"{'=' * 40}")

    return result


def generate_report(within_a: dict, within_b: dict, cross_ab: dict,
                    mode: str, elapsed: float):
    """生成评估报告"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)

    # 性能达标评估
    if within_a["mean"] > 0 and cross_ab["mean"] > 0:
        ratio = cross_ab["mean"] / within_a["mean"]
        pass_threshold = EVALUATION["pass_ratio"]
        passed = ratio >= pass_threshold

        print(f"\n[核心指标] 方式B跨组性能 / 方式A内部性能")
        print(f"  方式A内部平均相似度:   {within_a['mean']:.4f}")
        print(f"  方式B内部平均相似度:   {within_b['mean']:.4f}")
        print(f"  跨组A↔B平均相似度:    {cross_ab['mean']:.4f}")
        print(f"  性能比 (跨组/A内部):   {ratio:.4f} ({ratio*100:.1f}%)")
        print(f"  达标阈值:              {pass_threshold} ({pass_threshold*100:.0f}%)")
        print(f"  评估结果:              {'✓ 达标' if passed else '✗ 未达标'}")
    else:
        ratio = 0
        passed = False
        print("  [错误] 无法计算性能比（样本不足）")

    print(f"\n  匹配模式: {mode}")
    print(f"  总耗时:   {elapsed:.1f} 秒")

    # 保存详细报告
    report = {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "method_a_internal": {k: v for k, v in within_a.items() if k != "scores"},
        "method_b_internal": {k: v for k, v in within_b.items() if k != "scores"},
        "cross_ab": {k: v for k, v in cross_ab.items() if k != "scores"},
        "performance_ratio": ratio,
        "pass_threshold": EVALUATION["pass_ratio"],
        "passed": passed,
    }

    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细报告已保存: {report_path}")

    # 保存分数列表（用于后续分析）
    scores_path = os.path.join(OUTPUT_DIR, "all_scores.json")
    scores_data = {
        "within_a": within_a.get("scores", []),
        "within_b": within_b.get("scores", []),
        "cross_ab": cross_ab.get("scores", []),
        "cross_ab_details": cross_ab.get("details", []),
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"分数详情已保存: {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="指纹识别评估系统")
    parser.add_argument("--mode", choices=["quick", "minutiae", "cascade"],
                        default="cascade", help="匹配模式")
    parser.add_argument("--pair", nargs=2, type=int, default=None,
                        help="单对比对，指定两个指纹编号")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过数据库构建")
    parser.add_argument("--within-a-sample", type=int, default=10,
                        help="方式A内部评估采样数量（0=全部）")
    args = parser.parse_args()

    # 单对比对模式
    if args.pair:
        single_pair_match(args.pair[0], args.pair[1])
        return

    start_time = time.time()

    # Step 1: 构建数据库
    if not args.skip_build:
        db = build_database()
    else:
        db = FingerprintDatabase()
        print(f"使用已有数据库: {db.stats()}")

    # Step 2: 初始化匹配器
    print(f"\n匹配模式: {args.mode}")
    matcher = CascadedMatcher(prefer_deep=True)

    # Step 3: 获取路径
    a_start, a_end = METHOD_A_RANGE
    b_start, b_end = METHOD_B_RANGE
    paths_a = get_all_paths(a_start, a_end)
    paths_b = get_all_paths(b_start, b_end)

    # 如果方式A数据太多，采样以节省时间
    if args.within_a_sample > 0 and len(paths_a) > args.within_a_sample:
        import random
        random.seed(42)
        paths_a_eval = random.sample(paths_a, args.within_a_sample)
        print(f"\n方式A采样 {args.within_a_sample}/{len(paths_a)} 张用于内部评估")
    else:
        paths_a_eval = paths_a

    # Step 4: 评估方式A内部一致性
    within_a = evaluate_within_group(matcher, paths_a_eval, "方式A", args.mode)

    # Step 5: 评估方式B内部一致性
    within_b = evaluate_within_group(matcher, paths_b, "方式B", args.mode)

    # Step 6: 评估跨组 A ↔ B
    cross_ab = evaluate_cross_group(matcher, paths_a, paths_b, args.mode)

    # Step 7: 生成报告
    elapsed = time.time() - start_time
    generate_report(within_a, within_b, cross_ab, args.mode, elapsed)


if __name__ == "__main__":
    main()
