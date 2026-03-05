# 指纹识别级联匹配系统

## 系统架构

```
指纹图像 A, B
       │
  ┌────▼────────────────────┐
  │ 第一阶段: 深度学习初筛   │  CNN 全局特征 → 余弦相似度
  │ 低阈值 → 高召回率        │  快速排除明显不匹配
  └────┬────────────────────┘
       │ 通过
  ┌────▼────────────────────┐
  │ 第二阶段: 细节点精确匹配  │  端点/分叉点 → 拓扑比对
  │ 高阈值 → 高精度          │  最终确认，避免误判
  └────┬────────────────────┘
       │
  ┌────▼────────────────────┐
  │ 分数融合                  │  加权融合 → 最终相似度 (0~1)
  └─────────────────────────┘
```

## 项目结构

```
finger deletion/
├── config.py                    # 全局配置（阈值、参数、路径）
├── evaluate.py                  # 主评估脚本
├── requirements.txt             # Python 依赖
├── data/                        # 指纹图像数据
│   ├── 001.bmp ~ 040.jpg       # 方式A采集（专业设备）
│   └── 041.jpg ~ 050.jpg       # 方式B采集（待测设备）
├── src/                         # 核心算法模块
│   ├── preprocessing.py         # 图像预处理（CLAHE + Gabor + 骨架化）
│   ├── minutiae_extractor.py    # 细节点提取（含5层伪特征过滤）
│   ├── minutiae_matcher.py      # 细节点匹配（对齐 + 拓扑比对）
│   ├── deep_feature.py          # 深度学习特征提取（ResNet/轻量级备选）
│   ├── cascaded_matcher.py      # 级联匹配引擎（组合两阶段）
│   └── database.py              # 指纹数据库管理
├── output/                      # 评估结果输出
├── database/                    # 持久化数据库文件
└── models/                      # 模型文件（如有微调）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行评估

```bash
# 完整级联匹配评估（推荐）
python evaluate.py --mode cascade

# 仅深度学习快速评估
python evaluate.py --mode quick

# 仅细节点匹配评估
python evaluate.py --mode minutiae

# 单对指纹详细比对
python evaluate.py --pair 1 42
```

### 3. 查看结果

评估完成后，结果保存在 `output/` 目录：
- `evaluation_report.json` — 完整评估报告
- `all_scores.json` — 所有匹配分数原始数据

## 数据库扩展指南

### 添加新指纹

将新指纹图像放入 `data/` 目录，按顺序编号（如 `051.jpg`, `052.jpg`...）。

### 添加新的采集方式

在 `config.py` 中修改范围配置：

```python
# 示例：添加方式C（编号 051~060）
METHOD_A_RANGE = (1, 40)    # 方式A
METHOD_B_RANGE = (41, 50)   # 方式B
# METHOD_C_RANGE = (51, 60) # 方式C（需在 evaluate.py 中添加对应逻辑）
```

### 多人指纹数据库

推荐按如下方式组织：

```
data/
├── person_001/
│   ├── method_a/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── method_b/
│       ├── 001.jpg
│       └── ...
├── person_002/
│   ├── method_a/
│   └── method_b/
└── ...
```

## 关键配置说明

| 参数 | 位置 | 说明 |
|------|------|------|
| `CASCADE.stage1_threshold` | config.py | 深度学习初筛阈值，降低可提高召回率 |
| `CASCADE.stage2_threshold` | config.py | 细节点匹配阈值，提高可降低误判 |
| `CASCADE.deep_weight` | config.py | 最终分数中深度学习的权重 |
| `CASCADE.minutiae_weight` | config.py | 最终分数中细节点的权重 |
| `MINUTIAE.min_distance` | config.py | 细节点最小间距，增大可减少伪特征 |
| `EVALUATION.pass_ratio` | config.py | 方式B需达到方式A性能的百分比 |

## 伪细节点过滤机制

系统采用 **5 层过滤** 消除假特征，解决客户之前遇到的问题：

1. **边界排除** — 图像边缘的伪端点
2. **ROI 掩码验证** — 背景区域的噪声点
3. **最小间距过滤** — 密集伪特征簇（毛刺/断裂）
4. **局部对比度验证** — 低对比度区域的劣质点
5. **数量上限控制** — 按质量排序，截断过多的可疑点

## 技术说明

- **深度学习模型**: 默认使用 ResNet-18 预训练权重提取全局纹理特征。如 PyTorch 不可用，自动回退到 Gabor + LBP + ORB 混合特征方案
- **细节点匹配**: 基于交叉数（Crossing Number）算法提取端点和分叉点，通过锚点对齐 + 贪心匹配计算拓扑相似度
- **Gabor 增强**: 在预处理阶段使用基于方向场的 Gabor 滤波修复断裂脊线、消除粘连，从源头减少伪特征
