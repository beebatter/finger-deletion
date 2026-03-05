import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import random
import sys

# 保证能正确加载上级目录的配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEEP_FEATURE, DATA_DIR, MODEL_DIR

class SiameseFingerprintDataset(Dataset):
    """
    孪生网络数据集：动态生成正样本对和负样本对
    这里假设数据都在 DATA_DIR 下，以不同的文件夹(如 person_001) 或 命名前缀区分手指
    为简化，我们暂时用一个通用策略：前3位数字表示同一根手指 (如 001.jpg 和 001_1.jpg 属于同一手指)
    由于你是小部分数据，我们在内存中直接匹配
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 收集所有图像
        self.image_paths = []
        for ext in ["*.jpg", "*.bmp", "*.png", "*.tif"]:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
            
        # 根据规范假定：文件名或父目录决定了手指ID
        # 你的配置里好像有 001.bmp 到 050.jpg。如果1个编号就是1个手指，那其实是 unpaired，需要通过增强构建 pair。
        # 这里使用"强数据增强"策略，将同一张图片增强两次作为【正对】，将不同图片增强作为【负对】。
        # 这种自监督对比学习模式（类似SimCLR框架思想）非常适合小数据、少配对的场景。
        self.finger_ids = list(range(len(self.image_paths)))
        
    def __len__(self):
        return len(self.image_paths) * 5 # 虚拟放大数据集，一轮epoch可以多次采样
        
    def __getitem__(self, idx):
        # 决定是生成正对还是负对 (1:1 比例)
        is_positive = random.random() > 0.5
        
        base_img_idx = random.randint(0, len(self.image_paths) - 1)
        img1_path = self.image_paths[base_img_idx]
        
        if is_positive:
            img2_path = img1_path
            label = 1.0 # 相同手指
        else:
            neg_idx = random.randint(0, len(self.image_paths) - 1)
            while neg_idx == base_img_idx:
                neg_idx = random.randint(0, len(self.image_paths) - 1)
            img2_path = self.image_paths[neg_idx]
            label = 0.0 # 不同手指
            
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        # Label: 1代表一致(距离应近), 0代表不一致(距离应远)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class ContrastiveLoss(nn.Module):
    """对比损失函数
    正对拉近，负对推远（距离大于 margin）
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        # label=1 相同，label=0 相异
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

def train_model():
    print("====== 正在准备启动指纹模型微调 (度量学习) ======")
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and DEEP_FEATURE["use_gpu"] else "cpu")
    print(f"使用的计算设备: {device}")
    
    # 针对指纹识别的高度定制化【数据增强】策略，解决手指旋转、压力差异、不完整等鲁棒性问题
    train_transforms = T.Compose([
        T.RandomRotation(25), # 模拟不同偏航角
        T.RandomResizedCrop(DEEP_FEATURE["input_size"], scale=(0.7, 1.0)), # 模拟侧偏和遮挡，局部信息
        T.ColorJitter(brightness=0.3, contrast=0.3), # 模拟按压力度、干湿度引起的偏白偏黑
        T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.3), # 模拟模糊或油污
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = SiameseFingerprintDataset(DATA_DIR, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=DEEP_FEATURE["batch_size"], shuffle=True, num_workers=2)
    
    # 构建与 deep_feature.py 一致的网络结构
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    embed_dim = DEEP_FEATURE["embedding_dim"]
    model = nn.Sequential(
        backbone,
        nn.Linear(feature_dim, embed_dim),
        nn.BatchNorm1d(embed_dim),
    ).to(device)
    
    # 冻结前半部分骨干网络，只微调后面的层（防止小样本过拟合破坏预训练的纹理特征能力）
    for param in list(model.parameters())[:-6]:
        param.requires_grad = False
        
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 20
    print(f"数据量(增强后): {len(dataset)} 对 | Batch size: {DEEP_FEATURE['batch_size']}")
    
    best_loss = float('inf')
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            out1 = model(img1)
            out2 = model(img2)
            
            # L2 norm (与推理时保持一致)
            out1 = nn.functional.normalize(out1, p=2, dim=1)
            out2 = nn.functional.normalize(out2, p=2, dim=1)
            
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = DEEP_FEATURE["custom_weights_path"]
            torch.save(model.state_dict(), save_path)
            print(f"  --> 保存最佳权重到 {save_path}")

    print("====== 训练完毕 ======")

if __name__ == "__main__":
    train_model()
