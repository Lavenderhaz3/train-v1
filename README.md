# FLIR 设备识别模型训练管道

在 Windows 上训练 YOLOv8 设备检测模型，用于 FLIR 红外热像主项目的自动检测功能。

## 设备类型（5 类）

| class_id | 标签 | 中文 |
|:---:|------|------|
| 0 | transformer | 变压器 |
| 1 | switchgear | 开关柜 |
| 2 | cable | 电缆 |
| 3 | busbar | 母线 |
| 4 | insulator | 绝缘子 |

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.9 ~ 3.11
- NVIDIA GPU（推荐，CPU 也能跑但慢）

### 安装

```bash
git clone https://github.com/Lavenderhaz3/train-v1.git
cd train-v1
pip install -r requirements.txt
```

### 五步工作流

```
准备图片 → 标注 → 训练 → 评估 → U盘部署
```

#### Step 1：准备图片

把 FLIR 红外 JPEG 放进 `data/raw/`。每类至少 20 张，总数 ≥ 100 张效果开始显现。

> 💡 FLIR 红外图里嵌了一张可见光照片（1280×960 RGB），提取命令：
> ```
> exiftool -b -EmbeddedImage IR_xxxxx.jpg > visible.jpg
> ```
> 用可见光图标注比红外图精确得多。

#### Step 2：标注（MakeSense.ai）

1. 浏览器打开 https://www.makesense.ai
2. 拖入 `data/raw/` 下的图片
3. 导入标签列表 `scripts/01_labels.txt`
4. 逐张画框（快捷键：`R` 画框，`Delete` 删框）
5. 导出 → **YOLO** 格式 → 下载 `annotations.zip`
6. 解压到 `data/yolo/labels/`

#### Step 3：准备数据集 + 训练

```bash
# 自动划分 train/val
python scripts/02_prepare_dataset.py

# 训练全部 5 类（每类独立训练）
python scripts/03_train.py

# 或只训练一类
python scripts/03_train.py --class transformer --epochs 50
```

#### Step 4：评估

```bash
python scripts/04_evaluate.py
```

验收标准：mAP@50 ≥ 0.8 可部署。

#### Step 5：部署到 Mac 主项目

```bash
python scripts/05_deploy.py
```

将 `models/` 下的 `.pt` 文件通过 U 盘拷到 Mac：

```
/Users/mba/claude code/detect/backend/models/weights/
```

## 目录结构

```
train-v1/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_labels.txt              # MakeSense.ai 标签导入
│   ├── 02_prepare_dataset.py      # 解压标注 → 划分 train/val → 生成 dataset.yaml
│   ├── 03_train.py                # YOLOv8 训练（GPU/CPU 自动检测）
│   ├── 04_evaluate.py             # mAP / PR 曲线 / 混淆矩阵
│   └── 05_deploy.py               # 复制 best.pt → U 盘
├── data/
│   ├── raw/                       # 原始 FLIR JPEG（到这里）
│   └── yolo/                      # 脚本自动生成
│       ├── dataset.yaml
│       ├── images/{train,val}/
│       └── labels/{train,val}/
├── runs/                          # 训练输出（自动生成）
└── models/                        # 部署用 .pt 文件
    ├── transformer.pt
    ├── switchgear.pt
    ├── cable.pt
    ├── busbar.pt
    └── insulator.pt
```

## 模型选择

| 模型 | 参数量 | 适用 |
|------|:---:|------|
| yolov8n | 3.2M | 数据 <500 张（默认，推荐起步） |
| yolov8s | 11.2M | 数据 500-2000 张 |
| yolov8m | 25.9M | 数据 >2000 张 |

## 主项目集成

训练好的 `.pt` 放入主项目后，创建项目时选择对应的 `model_type`，上传红外图时自动检测生效。

主项目仓库：https://github.com/Lavenderhaz3/flir-thermal-analysis-v1.0
