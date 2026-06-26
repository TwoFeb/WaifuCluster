# 🎭 动漫二创图片自动整理工具

一个基于无监督机器学习的动漫同人图片自动整理工具。它能根据图片中的**角色身份**，将杂乱的图片库自动分类到不同的文件夹中，而**无需预先知道角色是谁**或进行任何人工标注。

## ✨ 项目特点

*   **无监督聚类**：无需标注数据，利用对比学习模型和密度聚类算法自动发现角色分组。
*   **动漫专用**：使用针对动漫角色优化的检测模型和特征提取模型，有效处理不同画风、视角和服饰下的同一角色。
*   **自动化流程**：一键完成人物检测、裁切、特征提取、聚类和文件整理全流程。
*   **灵活可调**：通过修改脚本中的参数，可以轻松调整聚类的精细度和性能。

## 📦 环境要求

*   **Python**: 3.8 或更高版本
*   **操作系统**: Windows, macOS, Linux
*   **硬件要求**:
    *   **CPU**: 完全支持运行（推理速度较慢）
    *   **GPU**: 可选（NVIDIA GPU + CUDA 12.x + cuDNN 9.x），可大幅加速特征提取过程

## 🛠️ 安装指南

### 1. 获取项目代码

如果你还没有项目代码，请先克隆或下载本项目。

### 2. 创建虚拟环境

建议使用虚拟环境隔离项目依赖，避免与系统Python冲突。

```bash
# 进入项目目录
cd D:/WaifuCluster

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. 安装依赖库

在激活的虚拟环境中，安装项目所需的所有库。

```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装核心依赖
pip install dghs-imgutils scikit-learn hdbscan pillow numpy pyyaml

# 可选：安装 GPU 版 ONNX Runtime (需要 CUDA 12.x + cuDNN 9.x)
# pip install onnxruntime-gpu
```

> **提示**：如果你不想配置复杂的 CUDA 环境，只需安装 `onnxruntime`（CPU版）即可，项目功能完全不受影响，只是运行速度稍慢。

## 🚀 快速开始

1.  **准备图片**：将你需要整理的动漫图片放入一个文件夹（例如 `./fanart`）。
2.  **配置路径**：打开 `WaifuCluster.py` 脚本，修改开头的 `SRC_DIR` 变量为你的图片文件夹路径。
3.  **运行脚本**：
    ```bash
    python WaifuCluster.py
    ```
4.  **查看结果**：脚本运行完成后，会在 `OUT_DIR`（默认为 `./sorted`）目录下创建多个文件夹（`character_000`, `character_001`, ...），每个文件夹包含一组被判定为同一角色的图片。无法明确分组的图片会放入 `noise` 文件夹。

## ⚙️ 配置参数说明

脚本中的关键参数集中在 `HDBSCAN` 聚类器和检测部分，你可以根据图片数量和期望的聚类效果进行调整：

| 参数名 | 位置 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `min_cluster_size` | `HDBSCAN` | `3` | 形成一个角色簇所需的最少图片数量。值越大，聚类越保守，小角色群可能被判为噪声。 |
| `min_samples` | `HDBSCAN` | `2` | 核心距离计算时的邻居数，影响密度估计的平滑度。通常与 `min_cluster_size` 保持一致或更小。 |
| `cluster_selection_method` | `HDBSCAN` | `'eom'` | 簇提取策略。`'eom'`（Excess of Mass）倾向于生成更稳定、更大的簇；`'leaf'` 则倾向于生成更多、更细碎的簇。 |
| `score` | 检测部分 | `0.5` | YOLO人物检测的置信度阈值，低于此值的检测框会被忽略。 |
| `bbox` 扩展比例 | 检测部分 | `1/4` | 为避免发型被切，将检测框向外扩展的比例。 |

## 📁 输出结构说明

脚本运行后，输出目录结构如下：

```
sorted/
├── character_000/      # 被判定为第0个角色的所有原图
│   ├── image1.jpg
│   └── image2.png
├── character_001/      # 被判定为第1个角色的所有原图
│   └── ...
├── ...
└── noise/              # 无法明确归入任何角色簇的图片
    └── ...
```

## 🔧 工作原理简介

本项目的技术流程可以概括为五步：

1.  **人物检测裁切**：使用基于YOLO的动漫人物检测模型（`deepghs/anime_head_detection`）定位图片中的角色头部区域，并裁切成单人子图，避免多人图干扰特征提取。
2.  **特征提取**：使用专为动漫角色设计的对比学习模型 **CCIP**（`deepghs/ccip`）将每个单人子图编码为一个768维的特征向量。该向量能有效捕捉角色身份信息，对画风、视角变化具有鲁棒性。
3.  **距离计算**：计算所有特征向量两两之间的余弦距离，生成一个 N×N 的距离矩阵。
4.  **密度聚类**：使用 **HDBSCAN** 算法对距离矩阵进行层次密度聚类。它无需预设簇数，能自动发现不同密度的角色簇，并将离群点标记为噪声。
5.  **文件整理**：根据聚类标签，将原始图片复制到对应的角色文件夹中。

## ❓ 常见问题

**Q: 运行时出现 `cublasLt64_12.dll missing` 或 `Failed to create CUDAExecutionProvider` 警告？**
**A:** 这是ONNX Runtime尝试加载GPU支持但失败的无害警告，它会自动回退到CPU模式。如果你不想看到此警告，可以安装纯CPU版的ONNX Runtime：`pip uninstall onnxruntime-gpu onnxruntime -y && pip install onnxruntime`。如果想用GPU，请确保已安装CUDA 12.x和cuDNN 9.x，并将其`bin`目录加入系统`PATH`环境变量。

**Q: 聚类效果不好，很多图片被分到了 `noise` 文件夹？**
**A:** 尝试调小 `min_cluster_size` 和 `min_samples` 参数，让算法对噪声更宽容。也可以尝试将 `cluster_selection_method` 改为 `'leaf'`，看是否能生成更细的簇。

**Q: 同一个角色被分到了多个文件夹？**
**A:** 这可能是因为该角色画风差异过大，或CCIP模型对某些细节（如发型变化）不够敏感。可以尝试调大 `min_cluster_size`，或后处理时手动合并相似文件夹。

**Q: 首次运行下载模型很慢？**
**A:** 模型权重从HuggingFace Hub下载，国内网络可能不稳定。可以尝试设置镜像源或使用代理。下载完成后会缓存在本地（`~/.cache/huggingface`），后续运行无需再次下载。

## 🗺️ 后续优化方向

*   [ ] 将脚本重构为模块化结构，提高可维护性。
*   [ ] 使用 `config.yaml` 集中管理所有配置参数。
*   [ ] 预下载模型权重并内置到项目中，实现离线运行。
*   [ ] 添加 Gradio 或 PyQT 图形界面，提升非技术用户体验。
*   [ ] 引入 FAISS 进行近似最近邻搜索，优化大规模图片库的聚类速度。


## 🙏 致谢

*   [DeepGHS](https://github.com/deepghs) 团队：提供了强大的 `imgutils` 库、CCIP 模型和动漫人物检测模型。
*   [scikit-learn](https://scikit-learn.org/) 和 [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) 社区：提供了核心的聚类算法实现。
*   [ONNX Runtime](https://onnxruntime.ai/)：提供了高效的跨平台模型推理引擎。