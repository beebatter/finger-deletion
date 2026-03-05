"""
指纹图像预处理模块
===================
包含：
1. 图像标准化（尺寸、灰度）
2. CLAHE 自适应直方图均衡化
3. 方向场估计
4. Gabor 滤波增强（基于方向场的脊线增强）
5. 二值化与细化（骨架化）
6. ROI 掩码提取
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREPROCESS


class FingerprintPreprocessor:
    """指纹图像预处理器"""

    def __init__(self, config=None):
        self.cfg = config or PREPROCESS

    # ----------------------------------------------------------
    # 1. 图像读取与标准化
    # ----------------------------------------------------------
    def load_and_normalize(self, image_path: str) -> np.ndarray:
        """读取图像并转换为标准化灰度图"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        # 缩放到标准尺寸
        img = cv2.resize(img, self.cfg["target_size"], interpolation=cv2.INTER_AREA)
        return img

    # ----------------------------------------------------------
    # 2. CLAHE 自适应直方图均衡化
    # ----------------------------------------------------------
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """CLAHE 增强局部对比度，突出脊线与谷线"""
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg["clahe_clip_limit"],
            tileGridSize=self.cfg["clahe_tile_size"],
        )
        return clahe.apply(img)

    # ----------------------------------------------------------
    # 3. ROI 掩码提取（区分前景/背景）
    # ----------------------------------------------------------
    def extract_roi_mask(self, img: np.ndarray) -> np.ndarray:
        """
        通过局部方差检测，分离指纹前景区域和空白背景。
        返回二值掩码：前景=255，背景=0
        """
        block = self.cfg["orient_block_size"]
        h, w = img.shape
        mask = np.zeros_like(img, dtype=np.uint8)

        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                patch = img[y : y + block, x : x + block].astype(np.float64)
                var = np.var(patch)
                if var > 100:  # 方差阈值，背景通常很平坦
                    mask[y : y + block, x : x + block] = 255

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # ----------------------------------------------------------
    # 4. 方向场估计
    # ----------------------------------------------------------
    def estimate_orientation_field(self, img: np.ndarray) -> np.ndarray:
        """
        使用梯度法估计每个块的脊线主方向（弧度制）。
        返回与输入同尺寸的方向场图。
        """
        block = self.cfg["orient_block_size"]
        h, w = img.shape

        # Sobel 梯度
        img_f = img.astype(np.float64)
        gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)

        orient = np.zeros_like(img_f)

        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                gx_block = gx[y : y + block, x : x + block]
                gy_block = gy[y : y + block, x : x + block]

                # 计算主方向
                vx = np.sum(2 * gx_block * gy_block)
                vy = np.sum(gx_block**2 - gy_block**2)
                angle = 0.5 * np.arctan2(vx, vy)

                orient[y : y + block, x : x + block] = angle

        return orient

    # ----------------------------------------------------------
    # 5. Gabor 滤波增强（多方向融合）
    # ----------------------------------------------------------
    def gabor_enhance(self, img: np.ndarray, orientation: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        """
        基于方向场使用 Gabor 滤波器增强脊线。
        这是专业指纹预处理的核心步骤，能修复断裂、消除粘连。
        """
        ksize = self.cfg["gabor_ksize"]
        sigma = self.cfg["gabor_sigma"]
        lambd = self.cfg["gabor_lambd"]
        gamma = self.cfg["gabor_gamma"]
        num_orient = self.cfg["gabor_num_orientations"]

        # 预构建多方向 Gabor 核
        gabor_kernels = []
        angles = np.linspace(0, np.pi, num_orient, endpoint=False)
        for theta in angles:
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F
            )
            kern /= kern.sum() + 1e-8
            gabor_kernels.append((theta, kern))

        # 对每个像素，选择最接近其方向的 Gabor 核进行滤波
        enhanced = np.zeros_like(img, dtype=np.float64)
        img_f = img.astype(np.float64)

        # 先计算所有方向的滤波结果
        filtered_results = []
        for theta, kern in gabor_kernels:
            result = cv2.filter2D(img_f, cv2.CV_64F, kern)
            filtered_results.append((theta, result))

        block = self.cfg["orient_block_size"]
        h, w = img.shape
        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                local_orient = orientation[y + block // 2, x + block // 2]
                # 找最近方向
                best_idx = np.argmin(
                    [abs(self._angle_diff(theta, local_orient))
                     for theta, _ in filtered_results]
                )
                enhanced[y : y + block, x : x + block] = \
                    filtered_results[best_idx][1][y : y + block, x : x + block]

        # 应用掩码
        enhanced = enhanced * (mask / 255.0)
        # 归一化到 0-255
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced.astype(np.uint8)

    # ----------------------------------------------------------
    # 6. 二值化
    # ----------------------------------------------------------
    def binarize(self, img: np.ndarray) -> np.ndarray:
        """将增强后的灰度图二值化"""
        if self.cfg["binarize_method"] == "adaptive":
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.cfg["adaptive_block_size"],
                self.cfg["adaptive_C"],
            )
        else:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    # ----------------------------------------------------------
    # 7. 骨架化（细化）
    # ----------------------------------------------------------
    def skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """将二值指纹图细化为单像素宽的骨架"""
        # skimage 的 skeletonize 期望布尔类型
        skeleton = skeletonize(binary > 0)
        return (skeleton * 255).astype(np.uint8)

    # ----------------------------------------------------------
    # 完整预处理流水线
    # ----------------------------------------------------------
    def process(self, image_path: str, return_intermediates: bool = False):
        """
        完整的预处理流程。
        
        Returns:
            如果 return_intermediates=False: (skeleton, mask)
            如果 return_intermediates=True:  dict 包含所有中间步骤
        """
        # Step 1: 读取并标准化
        raw = self.load_and_normalize(image_path)

        # Step 2: CLAHE 增强
        clahe_img = self.apply_clahe(raw)

        # Step 3: ROI 掩码
        mask = self.extract_roi_mask(clahe_img)

        # Step 4: 方向场
        orientation = self.estimate_orientation_field(clahe_img)

        # Step 5: Gabor 增强
        enhanced = self.gabor_enhance(clahe_img, orientation, mask)

        # Step 6: 二值化
        binary = self.binarize(enhanced)
        binary = binary & mask  # 应用掩码

        # Step 7: 骨架化
        skeleton = self.skeletonize(binary)
        skeleton = skeleton & mask  # 再次应用掩码

        if return_intermediates:
            return {
                "raw": raw,
                "clahe": clahe_img,
                "mask": mask,
                "orientation": orientation,
                "enhanced": enhanced,
                "binary": binary,
                "skeleton": skeleton,
            }
        return skeleton, mask

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """计算两个角度差（考虑周期性）"""
        diff = a - b
        while diff > np.pi / 2:
            diff -= np.pi
        while diff < -np.pi / 2:
            diff += np.pi
        return diff


# 模块级便捷函数
def preprocess_fingerprint(image_path: str, return_intermediates=False):
    """便捷函数：一键预处理"""
    preprocessor = FingerprintPreprocessor()
    return preprocessor.process(image_path, return_intermediates)
