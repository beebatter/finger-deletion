"""
快速测试脚本 - 验证系统各模块是否正常工作
用法: python quick_test.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import get_fingerprint_path, DATA_DIR


def test_preprocessing():
    """测试预处理模块"""
    print("\n[测试 1] 预处理模块")
    print("-" * 40)
    from src.preprocessing import FingerprintPreprocessor

    preprocessor = FingerprintPreprocessor()
    path = get_fingerprint_path(1)
    print(f"  测试图像: {os.path.basename(path)}")

    t0 = time.time()
    result = preprocessor.process(path, return_intermediates=True)
    elapsed = (time.time() - t0) * 1000

    print(f"  ✓ 原始图像尺寸:   {result['raw'].shape}")
    print(f"  ✓ CLAHE增强:      {result['clahe'].shape}")
    print(f"  ✓ ROI掩码:        {result['mask'].shape}, 前景占比: "
          f"{result['mask'].sum()/255/result['mask'].size:.1%}")
    print(f"  ✓ Gabor增强:      {result['enhanced'].shape}")
    print(f"  ✓ 二值化:         {result['binary'].shape}")
    print(f"  ✓ 骨架化:         {result['skeleton'].shape}")
    print(f"  耗时: {elapsed:.0f} ms")

    # 保存中间结果
    import cv2
    out_dir = os.path.join(ROOT, "output", "test_preprocess")
    os.makedirs(out_dir, exist_ok=True)
    for name, img in result.items():
        if name == "orientation":
            # 方向场转为可视化
            vis = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype("uint8")
            cv2.imwrite(os.path.join(out_dir, f"{name}.jpg"), vis)
        else:
            cv2.imwrite(os.path.join(out_dir, f"{name}.jpg"), img)
    print(f"  中间结果已保存到: {out_dir}")
    return True


def test_minutiae_extraction():
    """测试细节点提取"""
    print("\n[测试 2] 细节点提取")
    print("-" * 40)
    from src.preprocessing import FingerprintPreprocessor
    from src.minutiae_extractor import MinutiaeExtractor

    preprocessor = FingerprintPreprocessor()
    extractor = MinutiaeExtractor()

    path = get_fingerprint_path(1)
    intermediates = preprocessor.process(path, return_intermediates=True)

    t0 = time.time()
    minutiae = extractor.extract(
        intermediates["skeleton"],
        intermediates["mask"],
        intermediates["clahe"]
    )
    elapsed = (time.time() - t0) * 1000

    endings = sum(1 for m in minutiae if m.type == "ending")
    bifurcations = sum(1 for m in minutiae if m.type == "bifurcation")
    avg_quality = sum(m.quality for m in minutiae) / max(len(minutiae), 1)

    print(f"  ✓ 提取细节点: {len(minutiae)} 个")
    print(f"    端点（ending）:    {endings}")
    print(f"    分叉点（bifurcation）: {bifurcations}")
    print(f"    平均质量分: {avg_quality:.3f}")
    print(f"  耗时: {elapsed:.0f} ms")

    # 可视化
    out_path = os.path.join(ROOT, "output", "test_minutiae.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = extractor.visualize(intermediates["clahe"], minutiae, out_path)
    print(f"  可视化已保存: {out_path}")
    return True


def test_minutiae_matching():
    """测试细节点匹配"""
    print("\n[测试 3] 细节点匹配")
    print("-" * 40)
    from src.preprocessing import FingerprintPreprocessor
    from src.minutiae_extractor import MinutiaeExtractor
    from src.minutiae_matcher import MinutiaeMatcher

    preprocessor = FingerprintPreprocessor()
    extractor = MinutiaeExtractor()
    matcher = MinutiaeMatcher()

    # 同一方式的两张指纹
    path1 = get_fingerprint_path(1)
    path2 = get_fingerprint_path(2)

    int1 = preprocessor.process(path1, return_intermediates=True)
    int2 = preprocessor.process(path2, return_intermediates=True)

    m1 = extractor.extract(int1["skeleton"], int1["mask"], int1["clahe"])
    m2 = extractor.extract(int2["skeleton"], int2["mask"], int2["clahe"])

    t0 = time.time()
    result = matcher.match(m1, m2)
    elapsed = (time.time() - t0) * 1000

    print(f"  指纹 001 vs 002 (同方式A)")
    print(f"  ✓ 匹配分数:   {result['score']:.4f}")
    print(f"  ✓ 匹配点对:   {result['matched_pairs']}")
    print(f"  ✓ A总细节点:  {result['total_a']}")
    print(f"  ✓ B总细节点:  {result['total_b']}")
    print(f"  耗时: {elapsed:.0f} ms")

    # 跨方式测试
    path3 = get_fingerprint_path(41)
    int3 = preprocessor.process(path3, return_intermediates=True)
    m3 = extractor.extract(int3["skeleton"], int3["mask"], int3["clahe"])
    result2 = matcher.match(m1, m3)
    print(f"\n  指纹 001 vs 041 (跨方式A-B)")
    print(f"  ✓ 匹配分数:   {result2['score']:.4f}")
    print(f"  ✓ 匹配点对:   {result2['matched_pairs']}")
    return True


def test_deep_feature():
    """测试深度学习特征提取"""
    print("\n[测试 4] 深度学习特征提取")
    print("-" * 40)
    from src.deep_feature import get_feature_extractor

    try:
        extractor = get_feature_extractor(prefer_deep=True)
        backend = "PyTorch (深度学习)"
    except Exception:
        extractor = get_feature_extractor(prefer_deep=False)
        backend = "轻量级 (Gabor+LBP)"

    print(f"  后端: {backend}")

    path1 = get_fingerprint_path(1)
    path2 = get_fingerprint_path(2)
    path3 = get_fingerprint_path(41)

    t0 = time.time()
    f1 = extractor.extract_feature(path1)
    f2 = extractor.extract_feature(path2)
    f3 = extractor.extract_feature(path3)
    elapsed = (time.time() - t0) * 1000

    sim_12 = extractor.cosine_similarity(f1, f2)
    sim_13 = extractor.cosine_similarity(f1, f3)

    print(f"  ✓ 特征维度: {f1.shape}")
    print(f"  ✓ 001 vs 002 (同方式A): {sim_12:.4f}")
    print(f"  ✓ 001 vs 041 (跨方式):  {sim_13:.4f}")
    print(f"  耗时: {elapsed:.0f} ms (3张)")
    return True


def test_cascaded_matching():
    """测试级联匹配"""
    print("\n[测试 5] 级联匹配")
    print("-" * 40)
    from src.cascaded_matcher import CascadedMatcher

    matcher = CascadedMatcher(prefer_deep=True)

    path1 = get_fingerprint_path(1)
    path2 = get_fingerprint_path(2)
    path3 = get_fingerprint_path(41)

    print("  001 vs 002 (同方式A):")
    result1 = matcher.match(path1, path2, verbose=True)
    print(f"  → 最终分数: {result1['final_score']:.4f}\n")

    print("  001 vs 041 (跨方式A-B):")
    result2 = matcher.match(path1, path3, verbose=True)
    print(f"  → 最终分数: {result2['final_score']:.4f}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("指纹识别系统 - 模块测试")
    print("=" * 60)

    tests = [
        ("预处理模块", test_preprocessing),
        ("细节点提取", test_minutiae_extraction),
        ("细节点匹配", test_minutiae_matching),
        ("深度学习特征", test_deep_feature),
        ("级联匹配", test_cascaded_matching),
    ]

    results = []
    for name, func in tests:
        try:
            success = func()
            results.append((name, "✓ 通过" if success else "✗ 失败"))
        except Exception as e:
            results.append((name, f"✗ 错误: {e}"))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")
