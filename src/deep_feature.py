"""
深度学习特征提取模块
=====================
使用预训练 CNN 将指纹图像编码为高维特征向量，
通过余弦相似度计算两枚指纹的全局相似程度。

优势：
- 对畸变、模糊、脱皮有极强鲁棒性
- 不依赖传统图像预处理流水线
- 天然适合快速初筛
"""

import cv2
import numpy as np
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEP_FEATURE

# 延迟导入 PyTorch（可能未安装）
_torch = None
_torchvision = None


def _ensure_torch():
    """确保 PyTorch 已导入"""
    global _torch, _torchvision
    if _torch is None:
        try:
            import torch
            import torchvision
            import torchvision.transforms as transforms
            _torch = torch
            _torchvision = torchvision
        except ImportError:
            raise ImportError(
                "深度学习特征提取需要 PyTorch。\n"
                "请安装: pip install torch torchvision\n"
                "或访问: https://pytorch.org"
            )


class DeepFeatureExtractor:
    """基于深度学习的指纹特征提取器"""

    def __init__(self, config=None):
        self.cfg = config or DEEP_FEATURE
        _ensure_torch()
        self.device = self._get_device()
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _get_device(self):
        """选择计算设备"""
        if self.cfg["use_gpu"] and _torch.cuda.is_available():
            return _torch.device("cuda")
        return _torch.device("cpu")

    def _build_model(self):
        """构建特征提取模型（去掉分类头）"""
        import torchvision.models as models

        model_name = self.cfg["model_name"]
        embed_dim = self.cfg["embedding_dim"]

        if model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = backbone.fc.in_features
            backbone.fc = _torch.nn.Identity()
        elif model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = backbone.fc.in_features
            backbone.fc = _torch.nn.Identity()
        elif model_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = _torch.nn.Identity()
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 添加投影头，将特征降维到指定维度
        model = _torch.nn.Sequential(
            backbone,
            _torch.nn.Linear(feature_dim, embed_dim),
            _torch.nn.BatchNorm1d(embed_dim),
        )

        # 【新增】：加载指纹特征微调权重（如有）
        custom_weights = self.cfg.get("custom_weights_path")
        if custom_weights and os.path.exists(custom_weights):
            print(f"[DeepFeature] 加载指纹特征微调模型: {custom_weights}")
            model.load_state_dict(_torch.load(custom_weights, map_location=self.device))
        else:
            print(f"[DeepFeature] 未找到微调权重，使用ImageNet预训练参数提取粗特征")

        model = model.to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        """构建图像预处理变换"""
        import torchvision.transforms as T

        input_size = self.cfg["input_size"]
        return T.Compose([
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # ----------------------------------------------------------
    # 单张图像特征提取
    # ----------------------------------------------------------
    def extract_feature(self, image_path: str) -> np.ndarray:
        """
        从单张指纹图像提取特征向量。

        Args:
            image_path: 图像文件路径

        Returns:
            归一化后的特征向量 (embedding_dim,)
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # --- 加入同训练高度一致的物理预处理（CLAHE增强脊线） ---
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # 转为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 应用变换
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with _torch.no_grad():
            feature = self.model(tensor)
            # L2 归一化
            feature = _torch.nn.functional.normalize(feature, p=2, dim=1)

        return feature.cpu().numpy().flatten()

    # ----------------------------------------------------------
    # 批量特征提取
    # ----------------------------------------------------------
    def extract_features_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        批量提取多张指纹的特征向量。

        Returns:
            特征矩阵 (N, embedding_dim)
        """
        import torchvision.transforms as T

        batch_size = self.cfg["batch_size"]
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            tensors = []

            for path in batch_paths:
                img = cv2.imread(path)
                if img is None:
                    print(f"[警告] 无法读取: {path}，跳过")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(img))

            if not tensors:
                continue

            batch = _torch.stack(tensors).to(self.device)
            with _torch.no_grad():
                features = self.model(batch)
                features = _torch.nn.functional.normalize(features, p=2, dim=1)

            all_features.append(features.cpu().numpy())

        if not all_features:
            return np.array([])

        return np.vstack(all_features)

    # ----------------------------------------------------------
    # 相似度计算
    # ----------------------------------------------------------
    @staticmethod
    def cosine_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度。

        Returns:
            相似度分数 (0-1)，1表示完全相同
        """
        dot = np.dot(feat_a, feat_b)
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        similarity = dot / (norm_a * norm_b)
        # 归一化到 [0, 1]（余弦相似度范围为 [-1, 1]）
        return float((similarity + 1) / 2)

    @staticmethod
    def compute_similarity_matrix(features: np.ndarray) -> np.ndarray:
        """
        计算特征矩阵的成对相似度矩阵。

        Args:
            features: (N, D) 特征矩阵

        Returns:
            (N, N) 相似度矩阵
        """
        # 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = features / norms
        # 余弦相似度矩阵
        sim_matrix = np.dot(normalized, normalized.T)
        # 映射到 [0, 1]
        return (sim_matrix + 1) / 2


# ============================================================
# 轻量级替代方案（纯 OpenCV，不依赖 PyTorch）
# ============================================================
class LightweightFeatureExtractor:
    """
    不依赖深度学习框架的轻量级特征提取器。
    使用 ORB + 直方图 + Gabor 纹理特征。
    当 PyTorch 不可用时自动使用此方案。
    """

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.orb = cv2.ORB_create(nfeatures=500)
        # 构建 Gabor 滤波器组
        self.gabor_kernels = self._build_gabor_bank()

    def _build_gabor_bank(self):
        """构建多尺度多方向 Gabor 滤波器组"""
        kernels = []
        for sigma in [3, 5]:
            for theta in np.linspace(0, np.pi, 8, endpoint=False):
                for lambd in [8, 12]:
                    kern = cv2.getGaborKernel(
                        (21, 21), sigma, theta, lambd, 0.5, 0, cv2.CV_64F
                    )
                    kern /= kern.sum() + 1e-8
                    kernels.append(kern)
        return kernels

    def extract_feature(self, image_path: str) -> np.ndarray:
        """提取混合特征向量"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        img = cv2.resize(img, self.target_size)

        features = []

        # 1. Gabor 纹理特征（每个核的均值和方差）
        for kern in self.gabor_kernels:
            filtered = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kern)
            features.extend([np.mean(filtered), np.var(filtered)])

        # 2. 方向直方图
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(gy, gx)
        hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        features.extend(hist / (hist.sum() + 1e-8))

        # 3. 局部二值模式近似（简化LBP）
        lbp = self._simple_lbp(img)
        lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        features.extend(lbp_hist / (lbp_hist.sum() + 1e-8))

        feature_vec = np.array(features, dtype=np.float64)
        # L2 归一化
        norm = np.linalg.norm(feature_vec)
        if norm > 1e-8:
            feature_vec /= norm

        return feature_vec

    def _simple_lbp(self, img: np.ndarray) -> np.ndarray:
        """简化的 LBP（本地二值模式）"""
        h, w = img.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = img[i, j]
                code = 0
                code |= (1 << 7) if img[i - 1, j - 1] >= center else 0
                code |= (1 << 6) if img[i - 1, j] >= center else 0
                code |= (1 << 5) if img[i - 1, j + 1] >= center else 0
                code |= (1 << 4) if img[i, j + 1] >= center else 0
                code |= (1 << 3) if img[i + 1, j + 1] >= center else 0
                code |= (1 << 2) if img[i + 1, j] >= center else 0
                code |= (1 << 1) if img[i + 1, j - 1] >= center else 0
                code |= (1 << 0) if img[i, j - 1] >= center else 0
                lbp[i - 1, j - 1] = code
        return lbp

    @staticmethod
    def cosine_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """余弦相似度"""
        dot = np.dot(feat_a, feat_b)
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float((dot / (norm_a * norm_b) + 1) / 2)


def get_feature_extractor(prefer_deep=True):
    """
    获取特征提取器实例。
    优先使用深度学习版本，如果 PyTorch 不可用则回退到轻量级方案。
    """
    if prefer_deep:
        try:
            _ensure_torch()
            return DeepFeatureExtractor()
        except ImportError:
            print("[信息] PyTorch 不可用，使用轻量级特征提取器")
            return LightweightFeatureExtractor()
    return LightweightFeatureExtractor()
