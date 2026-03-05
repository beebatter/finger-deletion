import os
import sys
import argparse
from src.cascaded_matcher import CascadedMatcher
from config import CASCADE

def recognize_fingerprints(path_a: str, path_b: str):
    """
    查看任意两个指纹的相似度，根据系统配置阈值(90%)判定是否为同一个指纹
    """
    if not os.path.exists(path_a):
        print(f"找不到第一张指纹图片: {path_a}")
        return
    if not os.path.exists(path_b):
        print(f"找不到第二张指纹图片: {path_b}")
        return

    print("=" * 50)
    print(f"正在比对指纹...")
    print(f"指纹 A: {path_a}")
    print(f"指纹 B: {path_b}")
    print("=" * 50)

    # 实例化级联匹配器（结合深度学习特征和细节点）
    matcher = CascadedMatcher(prefer_deep=True)
    
    # 获取结果
    result = matcher.match(path_a, path_b, verbose=False)
    
    similarity_score = result["final_score"]
    threshold = CASCADE.get("match_decision_threshold", 0.90)
    is_match = similarity_score >= threshold
    
    # 转换为百分比以便直观显示
    sim_percent = similarity_score * 100
    thresh_percent = threshold * 100
    
    print("\n【比对结果】")
    print(f"详细情况: 深度相似度={result['deep_score']:.4f}, 细节点相似度={result['minutiae_score']:.4f}")
    if not result["stage1_passed"]:
        print("注意: 未通过第一阶段的深度相似度初筛！")
    
    print(f"综合相似度: {sim_percent:.2f}%")
    print(f"判定阈值:   {thresh_percent:.2f}%")
    print("-" * 50)
    
    if is_match:
        print("结论: ✅ 匹配成功！系统认为这是【同一个指纹】。")
    else:
        print("结论: ❌ 匹配失败！相似度低于阈值，系统认为这是【不同的指纹】。")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指纹 1:1 相似度对比 (90%判定为同一人)")
    parser.add_argument("image1", type=str, help="第一张指纹图片路径")
    parser.add_argument("image2", type=str, help="第二张指纹图片路径")
    args = parser.parse_args()
    
    recognize_fingerprints(args.image1, args.image2)
