# CLIP-LoRA Ship：基于 YOLO 数据集的视觉-语言模型船舶细粒度分类

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-CLIP-orange)](https://huggingface.co/openai/clip-vit-base-patch32)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)

将 YOLO/目标检测格式的 SeaShips 数据集改造为 CLIP 图文对比学习任务，配合 LoRA 轻量微调，实现 6 类船舶细粒度识别。验证集 Top-1 准确率 **92.72%**，仅训练 0.65% 的模型参数。全程独立完成。

---

## 项目背景

硕士课题。导师给了方向之后没有提供任何后续指导——从数据处理、方法设计到实验迭代全部独立完成。

课题的核心挑战是**跨范式迁移**：把一个为目标检测（YOLO）设计的数据集，改造成适合 CLIP 图文对比学习的格式，并让 CLIP 像分类器一样在固定的 N 个类别中做选择，而非开放式零样本推理。

---

## 数据

**[SeaShips(7000)](https://www.kaggle.com/datasets/dongly/shipdataset)** — 一个为目标检测任务设计的 VOC 格式数据集，共 7000 张海事监控图像，6 个类别：

```
数据集目录结构：
SeaShips(7000)/
├── JPEGImages/      # 7000 张 jpg，1920×1080
├── Annotations/     # 7000 个 VOC XML（每个含 bbox + 类别名）
├── train.txt        # 训练集 ID 列表（5147 张）
├── val.txt          # 验证集 ID 列表（1593 张）
└── test.txt         # 测试集 ID 列表
```

| 类别 | 英文名 |
|:-----|:-------|
| 散货船 | bulk cargo carrier |
| 集装箱船 | container ship |
| 渔船 | fishing boat |
| 杂货船 | general cargo ship |
| 矿砂船 | ore carrier |
| 客船 | passenger ship |

---

## 核心挑战：从 YOLO 数据到 CLIP 任务

### 问题

SeaShips 是为 YOLO/目标检测设计的——标注是 VOC XML，包含 bounding box 和类别名。CLIP 需要的是 **(image, text) 图文对**。两者之间存在一个范式鸿沟：

| | YOLO 检测范式 | CLIP 对比学习范式 |
|:--|:------------|:--------------|
| 输入 | 图像 + bbox 坐标 | 图像 + 自然语言描述 |
| 标注 | `(x1, y1, x2, y2, class)` | `("a photo of a container ship")` |
| 输出 | bbox 回归 + 类别概率 | 图文相似度向量 |
| 推理 | 检测框 + 分类 | 图像与候选文本匹配 |

直接套用 CLIP 行不通——需要把检测标注**转换**成 CLIP 能用的格式。

### 解决方案

**1. 数据转换：VOC XML → 图文对 JSON**

从每个 XML 的 `<object>` 标签中提取类别名，丢弃 bbox（本任务不需要定位），用文本模板生成自然语言描述：

```python
# VOC XML → (image_path, text, class_id)
pairs.append({
    "image_path": "JPEGImages/006625.jpg",
    "text": "a photo of a passenger ship in the ocean",   # 随机模板生成
    "class_id": 5
})
```

关键决策：**保留 class_id 用于后续的固定候选集评估**——这是让 CLIP 能像 YOLO 的分类头一样做 N 选 1 的前提。

**2. 评估设计：固定候选集分类（让 CLIP "N选1"）**

这是本项目最核心的方法设计。默认情况下 CLIP 做的是开放式图文匹配——给一张图和一段文字，输出匹配分数。但我们的需求是**给定一张图，判断它属于 6 类中的哪一类**——这和 YOLO 的分类头干的是同一件事。

做法：为全部 6 个类别各生成一条固定的文本描述，构成**固定候选集**。每张测试图像同时与这 6 条文本计算相似度，取最高者为预测：

```python
# 固定候选集（6 条文本，对应 6 个类别）
candidate_texts = [
    "a photo of a bulk cargo carrier ship",
    "a photo of a container ship ship",
    "a photo of a fishing boat ship",
    "a photo of a general cargo ship ship",
    "a photo of a ore carrier ship",
    "a photo of a passenger ship ship",
]

# 图像一次性与全部候选比较
inputs = processor(text=candidate_texts, images=image, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits_per_image.argmax(dim=-1)  # 0~5 选一个
```

这个设计让 CLIP 的行为等价于一个 6 分类器——和 YOLO 最后一层做的事情在逻辑上一致。

---

## 实验历程

因为没有导师反馈，整个实验全靠自己试错和修正。实验经历了n个阶段，第一阶段是自己无用的修改和阅读clip源码，第二阶段的关键改进正来自于对第一阶段失败的诊断。但是由于第一次实验的时间太久了，已经没有任何可以upload上来的东西了。（毕竟我也离开那里好久了）

### 第一次实验：尝试，但评估方式错了

- 训练 16 epoch，报告准确率 31.14%
- 使用单一固定模板，无文本增强
- **评估方式错误**：在 batch 内部做图文匹配（16 选 1），而不是在全部 6 个类别中分类
- 学了一堆调参技巧（CosineAnnealingLR、early stopping），但没意识到根本问题在评估逻辑

### 第二次实验：修正评估逻辑，收敛到 92.72%

三个关键改动：

1. **评估改为固定候选集分类**（上面描述的 N 选 1 方式）——这是最大的修正
2. **训练时 10 模板随机采样**——文本端的简单数据增强
3. **训练延长到 30 epoch**

### 第一阶段的教训

batch 内匹配（第一阶段的做法）看起来像是在做"分类"，但实际上是在做"在 16 个随机干扰中找到自己的配对"。模型不需要学会"ore carrier 和 bulk cargo carrier 有什么区别"，只需要学会"这张图对应 batch 里的第几个文本"。这就是为什么 31.14% 不能叫分类准确率——它测的根本不是分类能力。

从第一阶段到第二阶段，本质上是**从 CLIP 的默认行为（图文检索）转向了基于候选集的分类行为**——这是整个项目的方法论核心。

---

## 方法

### 基座模型

[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)，152M 参数，在 4 亿图文对上预训练。

### LoRA 配置

| 参数 | 值 |
|:-----|:--|
| rank `r` | 8 |
| alpha | 32 |
| target modules | `q_proj`, `v_proj`, `k_proj`, `out_proj` |
| dropout | 0.1 |
| 可训练参数 | 983,040 (0.6456%) |

### 训练超参

| 参数 | 值 |
|:-----|:--|
| Batch Size | 16 |
| Epochs | 30 |
| Learning Rate | 1e-5 |
| Optimizer | AdamW |
| Loss | Symmetric Contrastive Loss |
| GPU | Kaggle P100 16GB |
| 训练时间 | ~1.8 小时 |

### 多模板训练

从 10 个模板中随机采样，作为廉价的文本数据增强：

```python
TEXT_TEMPLATES = [
    "a photo of a {} ship",  "a {} vessel",
    "a ship of type {}",     "this is a {}",
    "a {} in the ocean",     "an image of a {} ship",
    "a {} boat",             "a {} floating on water",
    "a {} from the side",    "a {} seen from above"
]
```

> 注：模板中的 `ship` 后缀与部分类别名（如 `passenger ship`）产生了冗余拼接（`passenger ship ship`），属疏忽导致的 bug，修掉后预期还有小幅提升空间。

---

## 实验结果

| Epoch | Train Loss | Val Acc | | Epoch | Train Loss | Val Acc |
|:-----:|:----------:|:-------:|---|:-----:|:----------:|:-------:|
| 1 | 2.6370 | 57.94% | | 16 | 1.3980 | 90.46% |
| 2 | 2.1558 | 66.85% | | 17 | 1.3824 | 91.09% |
| 3 | 1.9712 | 71.81% | | 18 | 1.3733 | 90.90% |
| 4 | 1.8545 | 75.20% | | 19 | 1.3497 | 91.34% |
| 5 | 1.7861 | 78.09% | | 20 | 1.3515 | 91.34% |
| 6 | 1.7172 | 79.28% | | 21 | 1.3463 | 91.53% |
| 7 | 1.6577 | 82.49% | | 22 | 1.3264 | 91.53% |
| 8 | 1.6225 | 84.93% | | 23 | 1.3311 | 92.47% |
| 9 | 1.5750 | 85.69% | | 24 | 1.3094 | 91.46% |
| 10 | 1.5407 | 85.56% | | 25 | 1.3121 | 91.53% |
| 11 | 1.5115 | 87.19% | | 26 | 1.2938 | 91.84% |
| 12 | 1.4842 | 87.19% | | 27 | 1.3029 | 92.28% |
| 13 | 1.4598 | 89.52% | | **28** | 1.2908 | **92.72%** |
| 14 | 1.4337 | 90.27% | | 29 | 1.2830 | 91.96% |
| 15 | 1.4213 | 90.21% | | 30 | 1.2769 | 91.96% |

- 前 8 个 epoch 快速收敛到 85%——CLIP 的预训练表示非常适配这个领域
- Epoch 8–20 稳步爬升到 91%，LoRA 适配器学习领域细节
- Epoch 20+ 进入平台期，最优点 epoch 28（92.72%），之后训损继续降但准确率不再涨，开始轻微过拟合

---
## 已知局限

这个项目有几个没来得及做的分析：

### 1. 数据集不平衡未评估

SeaShips 是一个**类别不平衡**的数据集——不同船型的样本数量差异可能很大。比如 `fishing boat`（渔船）在海事监控中出现频率天然高于 `ore carrier`（矿砂船）。但本项目从头到尾只看了整体准确率这一个数字。

92.72% 在不平衡数据集上的实际含义可能是：多数类 99% + 少数类 60% 的平均。没有逐类的 precision/recall，不知道模型到底在哪些类别上翻车。这对一个分类项目的完整性来说是硬伤。

### 2. 没有做船舶领域特征分析

6 个类别之间的区分其实很有意思——散货船和矿砂船的外观差异主要在吃水深度和货舱口结构，渔船和其他船型的区别在船体尺寸和甲板设备。但本项目纯粹把 CLIP 当成黑盒在用，没有研究：

- 模型到底在看船的哪个部位做判断？（Grad-CAM 可视化）
- 哪些类别之间最容易混淆？（混淆矩阵）
- 船舶的领域知识（上层建筑、船首形状、货舱布局）是否与模型的注意力对齐？

这些分析如果有，能让项目从"跑通了一个模型"变成"理解了模型在这个领域的行为"。但因为完全靠自己摸索，没顾上这一层。

## 待完成

这些是如果继续推进可以做的事——但感觉做起来也没什么意义：

- [ ] 修复模板中的 `ship` 冗余
- [ ] 训练时加图像数据增强（随机翻转/裁剪）
- [ ] 与 Zero-shot CLIP、Linear Probe、Full Fine-tuning 做对比
- [ ] 混淆矩阵 + 逐类 F1
- [ ] 测试集（`test.txt`）上跑最终结果

---

## 复现

```bash
# Kaggle Notebook (P100/T4)
pip install -q peft accelerate tensorboard

# 按顺序执行 p2-30epoch-clip-ship-finetune-lora.ipynb：
# 1. 解析 VOC XML → train_data.json / val_data.json
# 2. 加载 CLIP + 注入 LoRA
# 3. 30 epoch 训练
# 4. 保存权重
```

数据：ship7000

---

## 文件

```
clip-ship-recognition/
├── 第一次实验16epoch-clip-ship-finetune-lora.ipynb  # Phase I
├── p2-30epoch-clip-ship-finetune-lora.ipynb         # Phase II（主力）
└── README.md
```

---

## 许可

MIT. 基座模型 CLIP 的许可参见 [OpenAI](https://github.com/openai/CLIP)。
