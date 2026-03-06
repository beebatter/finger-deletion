"""
SOCOFing 级联模型评测脚本
===========================
核心功能：
随机抽取 SOCOFing 数据集中的图像切片进行测试，评估修改权重后的系统准确率。

评测指标：
1. 真正例率 (TPR / Recall) - 同一指纹成功匹配的概率 (原图 vs 损坏图)
2. 误识率 (FAR / False Accept Rate) - 不同指纹被错误匹配的概率 (原图 vs 其他人原图)
3. 准确率 (Accuracy) - 整体判断正确的比例

用法：
    python eval_socofing.py --samples 50
"""

import os
import sys
import random
import argparse
import glob
from tqdm import tqdm
import re

# 将项目目录加入系统路径
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.cascaded_matcher import CascadedMatcher
from config import CASCADE

def get_identity(filename):
    """从文件名中提取指纹的唯一身份(去除后缀)"""
    basename = os.path.basename(filename)
    # SOCOFing 格式: 100__M_Left_index_finger_CR.BMP 或 100__M_Left_index_finger.BMP
    # 正则去除 _CR, _Zcut, _Obl 以及扩展名
    identity = re.sub(r'_(?:CR|Zcut|Obl)$', '', os.path.splitext(basename)[0])
    return identity

def evaluate_system(samples=50):
    val_real_dir = os.path.join(ROOT, "data/socofing/SOCOFing/Real")
    val_alt_easy = os.path.join(ROOT, "data/socofing/SOCOFing/Altered/Altered-Easy")
    val_alt_hard = os.path.join(ROOT, "data/socofing/SOCOFing/Altered/Altered-Hard")

    real_images = glob.glob(os.path.join(val_real_dir, "*.BMP"))
    if not real_images:
        print("未找到 SOCOFing 数据集，请确保数据在 data/socofing/ 目录下。")
        return

    # 随机选择测试样本
    selected_reals = random.sample(real_images, min(samples, len(real_images)))
    
    print("=" * 60)
    print(f" 开始执行系统级评估 (评测对数: {samples} 正样本 + {samples} 负样本)")
    print(f" 当前决策阈值: {CASCADE['match_decision_threshold']}")
    print(f" 当前分数权重: 深度={CASCADE['deep_weight']}, 细节={CASCADE['minutiae_weight']}")
    print("=" * 60)

    matcher = CascadedMatcher(prefer_deep=True)
    
    # 记录统计
    true_accepts = 0   # 应该匹配，且确实匹配
    false_rejects = 0  # 应该匹配，但却被拒绝
    
    true_rejects = 0   # 应该不匹配，且确实被拒绝
    false_accepts = 0  # 应该不匹配，但却错误匹配 (极其危险)
    
    positive_scores = []
    negative_scores = []

    print("\n[1/2] 测试正样本 (同源指纹: 原图 vs 破损图)...")
    for real_img in tqdm(selected_reals, desc="正例评估"):
        ident = get_identity(real_img)
        # 寻找对应的破损图 (在 Easy 和 Hard 中随机找一个存在的)
        alt_candidates = glob.glob(os.path.join(val_alt_hard, f"{ident}_*.BMP")) + \
                         glob.glob(os.path.join(val_alt_easy, f"{ident}_*.BMP"))
        
        if not alt_candidates:
            continue
            
        alt_img = random.choice(alt_candidates)
        
        # 级联匹配
        result = matcher.match(real_img, alt_img, verbose=False)
        score = result["final_score"]
        is_match = result.get("is_match", score >= CASCADE["match_decision_threshold"])
        
        positive_scores.append(score)
        if is_match:
            true_accepts += 1
        else:
            false_rejects += 1
            
    print("\n[2/2] 测试负样本 (异源指纹: 原图 vs 其他人原图)...")
    for i, real_img in tqdm(enumerate(selected_reals), desc="负例评估", total=len(selected_reals)):
        # 随机抽取一个不同的人
        other_img = random.choice(real_images)
        while get_identity(other_img) == get_identity(real_img):
            other_img = random.choice(real_images)
            
        result = matcher.match(real_img, other_img, verbose=False)
        score = result["final_score"]
        is_match = result.get("is_match", score >= CASCADE["match_decision_threshold"])
        
        negative_scores.append(score)
        if not is_match:
            true_rejects += 1
        else:
            false_accepts += 1

    # 统计指标
    tot_pos = true_accepts + false_rejects
    tot_neg = true_rejects + false_accepts
    
    tpr = (true_accepts / tot_pos) * 100 if tot_pos > 0 else 0
    far = (false_accepts / tot_neg) * 100 if tot_neg > 0 else 0
    accuracy = ((true_accepts + true_rejects) / (tot_pos + tot_neg)) * 100 if (tot_pos + tot_neg) > 0 else 0
    
    avg_pos_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    avg_neg_score = sum(negative_scores) / len(negative_scores) if negative_scores else 0

    print("\n" + "=" * 60)
    print(" 📊 评估结果报告 (Evaluation Report)")
    print("=" * 60)
    print(f"▶ 总评估对数   : {tot_pos + tot_neg} 对 ({tot_pos} 正样本 / {tot_neg} 负样本)")
    print(f"▶ 综合准确率   : {accuracy:.2f}%  (系统整体做出的正确判断比例)")
    print(f"▶ 真正例率(TPR): {tpr:.2f}%   (同源指纹被系统正确通过的比例)")
    print(f"▶ 误识率 (FAR) : {far:.2f}%   (【安全指标】异源指纹欺骗系统成功的比例)")
    print("-" * 60)
    print(f"▶ 同源指纹 平均得分 : {avg_pos_score * 100:.2f}%")
    print(f"▶ 异源指纹 平均得分 : {avg_neg_score * 100:.2f}%")
    print("=" * 60)
    print("【总结】")
    if far > 0:
        print("! 警告: FAR 大于0，说明有指纹骗过了系统，请考虑调高 `match_decision_threshold` 或 `minutiae_weight`！")
    else:
        print("✅ 极佳: 系统防伪造(FAR)表现满分，没有发生任何跨身份越权！")
        
    if tpr < 90:
        print("! 提示: 正常匹配识别率偏低，合法用户容易被拒绝，可稍微调低 `match_decision_threshold`。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50, help="抽取的正例和负例对数，默认为50对")
    args = parser.parse_args()
    
    evaluate_system(args.samples)
