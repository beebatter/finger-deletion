import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import cv2
import sys
import random
import numpy as np
from tqdm import tqdm

# 保证能正确加载上一级配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEEP_FEATURE, DATA_DIR, MODEL_DIR

class StandardFingerprintDataset(Dataset):
    """
    通用型大库指纹训练数据集（已加入预处理和严格同源匹配）
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 1. 搜集所有图片
        image_paths = []
        for ext in ["*.jpg", "*.bmp", "*.png", "*.tif", "*.JPG", "*.BMP", "*.PNG", "*.TIF"]:
            image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
            
        # 2. 根据文件名逻辑做聚合：将相同手指的不同变形整理到一起！
        # 比如："100__M_Left_index_finger_CR.BMP" 和 "100__M_Left_index_finger_Zcut.BMP" 都是同一根手指
        self.identities = {}
        for path in image_paths:
            filename = os.path.basename(path)
            # 剥离 Altered 后缀得到最纯净的手指唯一身份
            base_id = filename.split('_CR')[0].split('_Obl')[0].split('_Zcut')[0].replace('.BMP', '').replace('.bmp', '')
            # 也兼容您 /data 目录下直接命名为 "001.bmp" 等的情况
            base_id = base_id.replace('.jpg', '').replace('.png', '')
            
            if base_id not in self.identities:
                self.identities[base_id] = []
            self.identities[base_id].append(path)
            
        self.identity_keys = list(self.identities.keys())
        # 数据集长度定义为您数据集中具有独立身份手指的倍数，确保每个 Epoch 能看足够多的图
        self.length = len(image_paths) 
        
    def __len__(self):
        return self.length
        
    def load_processed_image(self, path):
        """
        *** 响应您的极其专业的建议：指纹必须做图像预处理！ ***
        由于干湿不均、灰度差异，直接喂给浅层网络是不行的。我们需要在这里加入 CLAHE 等计算
        """
        try:
            # 1. 灰阶读取
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return Image.new('RGB', (224, 224))
            
            # 2. 对比度受限的自适应直方图均衡化 (CLAHE) - 极致增强脊线（指纹纹理）清晰度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # 3. 再转换回 RGB 欺骗 ResNet 的三通道输入
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(img)
        except Exception:
            return Image.new('RGB', (224, 224))
        
    def __getitem__(self, idx):
        # 50% 概率正样本对，50%概率负样本对
        is_positive = random.random() > 0.5
        
        # 随机抽取一根手指（一个人）
        anchor_id = self.identity_keys[idx % len(self.identity_keys)]
        anchor_images = self.identities[anchor_id]
        
        # 抽出基准图
        img1_path = random.choice(anchor_images)
        
        if is_positive:
            # 如果是正样本对，我们从该人的其他照片（如人为扭曲图）中抽一张。这才是真正的"同一个手指不同按压"！
            # 而不是像以前那样取相同图片做微强扭曲
            if len(anchor_images) > 1:
                img2_path = random.choice([p for p in anchor_images if p != img1_path])
            else:
                img2_path = img1_path
            # CosineEmbeddingLoss 里，1 表示应该相似
            label = 1.0 
        else:
            # 如果是负对，抽另外一个人的指纹
            neg_id = random.choice(self.identity_keys)
            while neg_id == anchor_id:
                neg_id = random.choice(self.identity_keys)
            img2_path = random.choice(self.identities[neg_id])
            # CosineEmbeddingLoss 里，-1 表示应该推远
            label = -1.0 
            
        img1 = self.load_processed_image(img1_path)
        img2 = self.load_processed_image(img2_path)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)

def train_general_model():
    print("====== 开始常规范式指纹训练（增强预处理重构版） ======")
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and DEEP_FEATURE["use_gpu"] else "cpu")
    print(f"计算设备: {device}")
    
    # 因为上游 cv2.CLAHE 已经做了核心提纯，这里的增强专注于几何变化即可（旋转+偏移）
    train_transforms = T.Compose([
        T.Resize(DEEP_FEATURE["input_size"]),
        T.RandomRotation(30),  # 几何旋转
        T.RandomResizedCrop(DEEP_FEATURE["input_size"], scale=(0.85, 1.0)), # 残缺模拟
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    from torch.utils.data import random_split
    full_dataset = StandardFingerprintDataset(DATA_DIR, transform=train_transforms)
    
    if len(full_dataset) == 0:
        print("请检查 data 目录，未找到图像！")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=DEEP_FEATURE["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=DEEP_FEATURE["batch_size"], shuffle=False, num_workers=2)
    
    print(f"数据总大小: {len(full_dataset)} 张 | 唯一手指身份数: {len(full_dataset.identities)}")
    
    # 构建 ResNet18 特征提取器
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity() # 去掉原有的类预测头，我们要提取的是指纹的高维几何向量
    
    embed_dim = DEEP_FEATURE["embedding_dim"]
    model = nn.Sequential(
        backbone,
        nn.Linear(feature_dim, embed_dim),
        nn.BatchNorm1d(embed_dim),
    ).to(device)
    
    # === 使用更稳定科学的 CosineEmbeddingLoss 代替原先难收敛的 ContrastiveLoss ===
    criterion = nn.CosineEmbeddingLoss(margin=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # AdamW 可以更好地正则化

    num_epochs = 3
    print(f"Batch size: {DEEP_FEATURE['batch_size']}")
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] 训练", leave=False)
        for i, (img1, img2, label) in enumerate(train_pbar):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            out1 = model(img1)
            out2 = model(img2)
            
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / max(1, len(train_loader))
        
        # ----------- 验证集评估 -----------
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] 验证", leave=False)
        with torch.no_grad():
            for img1, img2, label in val_pbar:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1 = model(img1)
                out2 = model(img2)
                loss = criterion(out1, out2, label)
                val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}] 训练集 Loss: {avg_train_loss:.4f} | 验证集 Loss: {avg_val_loss:.4f}")
        
        save_path = DEEP_FEATURE["custom_weights_path"]
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> 发现更好模型，已保存至 {save_path}")

    print("====== 训练完毕 ======")

if __name__ == "__main__":
    train_general_model()
