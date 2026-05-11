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

## 核心思路

FLIR 红外图里嵌了一张可见光照片（1280×960 RGB）。**用可见光图标注设备比用铁红色板红外图精确得多**——看得清形状、纹理、铭牌。标注完成后，标注坐标映射回红外图进行 YOLOv8 训练。

```
FLIR JPEG ──extract──▶ 可见光图 ──MakeSense标注──▶ YOLO标注.txt
     │                                                    │
     └──── 红外显示图 ◀──── 坐标映射（同FOV，归一化一致）◀──┘
                              │
                         YOLOv8 训练
                              │
                         .pt 模型 → U盘 → Mac主项目
```

## 环境要求

- Windows 10/11
- Python 3.9 ~ 3.11
- **exiftool**（提取可见光图用）：https://exiftool.org 下载 Windows 版，加入 PATH
- NVIDIA GPU（推荐，CPU 也能跑但慢）

## 安装

```bash
git clone https://github.com/Lavenderhaz3/train-v1.git
cd train-v1
pip install -r requirements.txt
```

## 标准工作流（六步）

### Step 0：提取可见光图

```bash
# 把 FLIR 红外 JPEG 放进 data/raw/
# 然后提取可见光照片
python scripts/00_extract_visible.py
# 输出：data/visible/*.jpg + data/visible/mapping.json
```

### Step 1：标注（MakeSense.ai）

1. 浏览器打开 https://www.makesense.ai
2. 拖入 `data/visible/` 下的**可见光图**（不要拖红外图）
3. 导入标签列表 `scripts/01_labels.txt`
4. 逐张画框（快捷键：`R` 画框，`Delete` 删框，`→` 下一张）
5. 标注完 → 点右上角 **Actions** → **Export Labels**
6. 选 **YOLO** 格式 → 下载 `annotations.zip`
7. 解压到 `data/yolo/labels/`

### Step 2：准备数据集

```bash
python scripts/02_prepare_dataset.py --source visible
# 自动：校验标注 → 匹配 FLIR 原图 → 划分 train/val → 生成 dataset.yaml
```

### Step 3：训练

```bash
# 训练全部 5 类（每类独立训练一个模型）
python scripts/03_train.py

# 或只训练一类
python scripts/03_train.py --class transformer --epochs 50

# 用更大的模型
python scripts/03_train.py --model yolov8s --epochs 200
```

### Step 4：评估

```bash
python scripts/04_evaluate.py
```

验收标准：mAP@50 ≥ 0.8 可部署。

### Step 5：部署到 Mac 主项目

```bash
python scripts/05_deploy.py
# 输出 models/*.pt，并显示 U 盘拷贝路径
```

将 `models/` 下的 `.pt` 文件通过 U 盘拷到 Mac：

```
/Users/mba/claude code/detect/backend/models/weights/
```

---

## 备选工作流：直接在红外图上标注

如果可见光提取失败（非 FLIR 相机的普通 JPEG），可以直接在红外显示图上标注：

```bash
# 把图片放进 data/raw/
# 上传 data/raw/ 下的图片到 MakeSense 标注
# 导出标注 zip → 解压到 data/yolo/labels/
python scripts/02_prepare_dataset.py --source display
python scripts/03_train.py
```

---

## 目录结构

```
train-v1/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 00_extract_visible.py       # 从 FLIR JPEG 提取可见光图
│   ├── 01_labels.txt               # MakeSense.ai 标签导入（5 类）
│   ├── 02_prepare_dataset.py       # 标注 → train/val 划分 → dataset.yaml
│   ├── 03_train.py                 # YOLOv8 训练（GPU/CPU 自动检测）
│   ├── 04_evaluate.py              # mAP / PR / 混淆矩阵
│   └── 05_deploy.py                # best.pt → U 盘部署
├── data/
│   ├── raw/                        # FLIR 红外 JPEG（放到这里）
│   ├── visible/                    # 提取的可见光图（自动生成）
│   └── yolo/                       # 训练数据集（自动生成）
│       ├── dataset.yaml
│       ├── images/{train,val}/
│       └── labels/{train,val}/
├── runs/                           # 训练输出（自动生成）
└── models/                         # 部署用 .pt 文件
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

训练好的 `.pt` 放入主项目 `backend/models/weights/` 后，创建项目时选择对应的 `model_type`，上传红外图即自动检测。

主项目：https://github.com/Lavenderhaz3/flir-thermal-analysis-v1.0
