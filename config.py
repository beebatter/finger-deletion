"""
指纹识别系统 - 全局配置文件
Fingerprint Recognition System - Global Configuration
"""
import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "database")

# ============================================================
# 数据库配置：指纹分组
# ============================================================
# 方式A采集的指纹编号范围（专业设备）
METHOD_A_RANGE = (1, 40)
# 方式B采集的指纹编号范围（待测设备）
METHOD_B_RANGE = (41, 50)

# 指纹文件名模板（支持 .bmp 和 .jpg 混合）
def get_fingerprint_path(index: int) -> str:
    """根据编号获取指纹文件路径，自动检测扩展名"""
    for ext in [".bmp", ".jpg", ".png", ".tif"]:
        path = os.path.join(DATA_DIR, f"{index:03d}{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"找不到编号 {index:03d} 的指纹文件")

# ============================================================
# 图像预处理参数
# ============================================================
PREPROCESS = {
    # 标准化图像尺寸（宽, 高）
    "target_size": (320, 400),
    # CLAHE 自适应直方图均衡化
    "clahe_clip_limit": 3.0,
    "clahe_tile_size": (8, 8),
    # 高斯模糊核大小
    "gaussian_blur_ksize": 5,
    # Gabor 滤波器参数
    "gabor_ksize": 31,
    "gabor_sigma": 4.0,
    "gabor_lambd": 10.0,
    "gabor_gamma": 0.5,
    "gabor_num_orientations": 16,  # 方向数量
    # 方向场估计：分块大小
    "orient_block_size": 16,
    # 二值化方法："adaptive" or "otsu"
    "binarize_method": "adaptive",
    "adaptive_block_size": 15,
    "adaptive_C": 8,
}

# ============================================================
# 细节点提取参数
# ============================================================
MINUTIAE = {
    # 最小特征间距（像素），过滤密集伪特征
    "min_distance": 10,
    # 边界排除区域（像素），忽略图像边缘的伪特征
    "border_margin": 20,
    # 有效ROI掩码的最小前景像素比（去除背景区域的假特征）
    "min_foreground_ratio": 0.05,
    # 细节点质量过滤：局部对比度阈值
    "min_local_contrast": 15,
    # 最大合理细节点数量（超过说明提取有问题）
    "max_minutiae_count": 150,
    # 细节点类型
    "types": {"ending": 1, "bifurcation": 3},
}

# ============================================================
# 细节点匹配参数
# ============================================================
MINUTIAE_MATCH = {
    # 位置容差（像素）
    "distance_threshold": 20,
    # 角度容差（弧度）
    "angle_threshold": 0.35,  # ~20度
    # 最少匹配点数
    "min_matched_points": 8,
    # 匹配分数权重
    "score_weight": 0.5,
}

# ============================================================
# 深度学习特征参数
# ============================================================
DEEP_FEATURE = {
    # 使用的预训练模型
    "model_name": "resnet18",  # resnet18, resnet50, mobilenet_v2
    # 微调后的模型权重路径（如果存在则加载，用于提取指纹专用鲁棒特征）
    "custom_weights_path": os.path.join(MODEL_DIR, "best_fingerprint_model.pth"),
    # 特征向量维度（模型输出后降维）
    "embedding_dim": 256,
    # 输入图像尺寸（适配模型）
    "input_size": (224, 224),
    # 是否使用GPU
    "use_gpu": True,
    # 批处理大小
    "batch_size": 16,
}

# ============================================================
# 级联匹配参数
# ============================================================
CASCADE = {
    # 第一阶段：深度学习初筛阈值（提取指纹特征后可适当提高，如设为0.4-0.5）
    "stage1_threshold": 0.4,
    # 第二阶段：细节点精确匹配阈值（高阈值 -> 高精度）
    "stage2_threshold": 0.4,
    # 最终相似度融合权重
    "deep_weight": 0.4,      # 深度学习分数权重
    "minutiae_weight": 0.6,  # 细节点分数权重
    
    # 【重构：常规识别设定】1:1判定最终得分阈值（设为90%，超过此分则判定为同一枚指纹）
    "match_decision_threshold": 0.90, 
}

# ============================================================
# 评估参数
# ============================================================
EVALUATION = {
    # 性能达标比率（B方式相对A方式的性能比）
    "pass_ratio": 0.9,
    # 输出详细报告
    "verbose": True,
    # 保存可视化结果
    "save_visualization": True,
}
