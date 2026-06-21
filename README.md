# U-Net Image Segmentation — Oxford-IIIT Pet Dataset

从零手写 U-Net 语义分割网络，在 Oxford-IIIT Pet Dataset 上对比三种损失函数（Cross-Entropy、Dice Loss、CE + Dice）的性能。

![Training Curves](output/training_curves.png)

## 项目结构

```
unet-segmentation/
├── train.py              # 训练主脚本
├── model.py              # U-Net 网络定义（从零手写）
├── loss.py               # 损失函数（Dice Loss、Combined Loss、mIoU 计算）
├── README.md             # 本文件
├── data/                 # 数据集（自动下载）
├── output/               # 实验结果输出
│   ├── training_curves.png       # 训练曲线图
│   ├── best_miou_comparison.png  # 最佳 mIoU 对比图
│   ├── results_table.png         # 结果汇总表
│   ├── comparison.png            # 自动生成的原始对比图
│   ├── CE_only/                  # CE Loss 实验结果
│   ├── Dice_only/                # Dice Loss 实验结果
│   └── CE_Dice/                  # CE + Dice 实验结果
└── training_30epochs.log  # 训练日志
```

## 环境配置

### 依赖

- Python 3.8+
- PyTorch 2.0+（推荐 2.5+）
- CUDA 11.8+（GPU 训练必需）
- torchvision
- matplotlib
- numpy

### 安装

```bash
# 创建 conda 环境（推荐）
conda create -n unet python=3.10
conda activate unet

# 安装 PyTorch（以 CUDA 12.1 为例，请根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install matplotlib numpy
```

### 数据准备

数据集使用 Oxford-IIIT Pet Dataset。训练脚本会自动下载：

```python
# 在 train.py 中：
OxfordIIITPet(root="data", split="trainval", target_types="segmentation", download=True)
```

如果自动下载失败，可以手动下载并解压到 `data/` 目录：

```bash
# 下载
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# 解压到 data/ 目录
tar -xzf images.tar.gz -C data/
tar -xzf annotations.tar.gz -C data/
```

## 快速开始

### 训练完整实验（3 种损失函数对比）

```bash
# 默认配置：CE_only + Dice_only + CE_Dice，各 8 epoch
python train.py

# 指定自定义参数
python train.py --epochs 12 --batch-size 64 --lr 0.001
```

### 训练单个实验

```bash
# 只运行 CE_only
python train.py --experiments CE_only

# 只运行 Dice_only
python train.py --experiments Dice_only

# 只运行 CE_Dice
python train.py --experiments CE_Dice
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 30 | 每个实验的训练轮数 |
| `--batch-size` | 16 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--experiments` | 全部 | 指定实验（CE_only / Dice_only / CE_Dice） |
| `--val-interval` | 1 | 验证间隔（每 N 轮验证一次） |

### 使用验证间隔加速训练

每轮都做验证会延长训练时间。可以降低验证频率：

```bash
# 每 2 轮验证一次，训练时间约减少 30%
python train.py --epochs 8 --batch-size 96 --val-interval 2
```

## 网络结构

U-Net 包含编码器（下采样）、瓶颈层、解码器（上采样）和跳跃连接，全部从零手写。

```
输入: 256×256×3
  │
  ├─ Encoder (下采样 × 4)
  │   64 → 128 → 256 → 512 → 1024  (通道数)
  │   256×256 → 128×128 → 64×64 → 32×32 → 16×16  (空间尺寸)
  │
  ├─ Bottleneck (DoubleConv 1024)
  │
  ├─ Decoder (上采样 × 4 + Skip Connection)
  │   1024 → 512 → 256 → 128 → 64  (通道数)
  │   16×16 → 32×32 → 64×64 → 128×128 → 256×256  (空间尺寸)
  │
  └─ 输出: Conv1×1(64→3) → 256×256×3 (logits)
```

总参数量: **31,037,763**

每个 `DoubleConv` = `Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU`

## 损失函数

### Cross-Entropy Loss
标准多分类交叉熵，作为基线。

### Dice Loss（手动实现）
针对像素不平衡问题，直接优化 IoU：

$$Dice = \frac{2 \times |P \cap T| + \varepsilon}{|P| + |T| + \varepsilon}$$

$$L_{Dice} = 1 - \frac{1}{C} \sum_{c=1}^{C} Dice_c$$

### CE + Dice 组合损失
$$L = \alpha \cdot L_{CE} + \beta \cdot L_{Dice}$$

实验中 $\alpha = \beta = 1.0$。

## 实验结果

![Best mIoU Comparison](output/best_miou_comparison.png)

![Results Table](output/results_table.png)

| 实验 | 最佳 mIoU | 最佳 Epoch | 最终 Train Loss | 最终 Val Loss |
|------|-----------|-----------|-----------------|---------------|
| CE_only | 0.6410 | 8 | 0.3758 | 0.3793 |
| Dice_only | 0.6624 | 8 | 0.2181 | 0.2192 |
| **CE_Dice** | **0.6657** | 8 | 0.6581 | 0.6684 |

### 关键结论

1. **Dice Loss 优于 Cross-Entropy**：在像素不平衡的分割任务中，Dice Loss 收敛更快、最终 mIoU 更高（0.6624 vs 0.6410）。
2. **组合损失取得最佳效果**：CE + Dice（0.6657）为三者最高，推荐作为默认配置。
3. **从零训练即可有效收敛**：仅 8 epoch（约 2.5 小时）即可达到 mIoU > 0.66。

## 常见问题

### Q: 训练时 CPU 满载、GPU 空转？

A: 检查 DataLoader 的 `num_workers` 参数。如果共享内存（SHM）较小，需设置为 `num_workers=0`，但这会导致 CPU 数据加载成为瓶颈。有条件的场景建议增加 SHM 并使用多进程加载。

### Q: 显存不足（OOM）？

A: 调小 `--batch-size`，或在 `train.py` 的 Config 类中降低 `batch_size` 默认值。

### Q: 训练卡死？

A: 某些虚拟化环境（如 HAMI）可能与 `torch.compile` 或多进程数据加载冲突。确保：
- `num_workers=0`
- 不使用 `torch.compile`
- 不使用 `mp.set_sharing_strategy("file_system")`

## 参考

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
