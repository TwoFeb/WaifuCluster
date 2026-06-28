# 🎭 动漫二创图片自动整理工具

一个基于无监督机器学习的动漫同人图片自动整理工具。它能根据图片中的**角色身份**，将杂乱的图片库自动分类到不同的文件夹中，而**无需预先知道角色是谁**或进行任何人工标注。

针对不同的显卡生态和聚类偏好，本项目提供了多个版本的脚本，支持 Nvidia CUDA 以及 AMD/Intel DirectML 硬件加速，并集成了多种先进的聚类算法（HDBSCAN、OPTICS、凝聚层次聚类）。

## ✨ 项目特点

*   **多硬件加速支持**：同时提供 Nvidia CUDA 和 DirectML (AMD/Intel) 加速方案，A 卡/I 卡用户也能享受 GPU 加速。
*   **多聚类算法可选**：内置 HDBSCAN、CCIP OPTICS、凝聚层次聚类 (AHC) 三种算法，针对不同的图库特点可选择最优策略。
*   **无监督聚类**：无需标注数据，利用对比学习模型自动发现角色分组。
*   **动漫专用模型**：使用针对动漫角色优化的 `head_detect_v2.0_x_yv11` 检测模型和 **CCIP** 特征提取模型，有效处理不同画风、视角和服饰下的同一角色。
*   **自动化流程**：一键完成人物检测、裁切、特征提取、聚类和文件整理全流程。

## 📂 脚本版本说明

本项目包含 4 个不同的脚本，请根据你的显卡类型和聚类需求选择运行：

| 脚本名称 | 硬件加速 | 聚类算法 | 特点说明 |
| :--- | :--- | :--- | :--- |
| **`WaifuCluster.py`** | Nvidia CUDA | HDBSCAN | 专为 N 卡用户优化，需配置 CUDA 环境。输出**原图**。 |
| **`WaifuClusterDirectML.py`** | DirectML | HDBSCAN | 适用于 A 卡/I 卡。运行时会自动扫描不同 Epsilon 下的聚类效果以供参考。输出**原图**。 |
| **`WaifuClusterDirectMLOptics.py`** | DirectML | CCIP OPTICS | 适用于 A 卡/I 卡。采用 `imgutils` 官方的 OPTICS 算法，更适合多角色大杂烩图库。输出**裁切图**。 |
| **`WaifuClusterAHC.py`** | DirectML | 凝聚层次聚类 (AHC) | 适用于 A 卡/I 卡。基于距离矩阵的层次聚类，**不会产生噪声点**，强制将所有图片归类。输出**裁切图**。 |

> **注意**：DirectML 版本的脚本使用了一种“黑魔法”（Monkey Patch），会拦截并强制将 ONNX Runtime 的计算提供者替换为 `DmlExecutionProvider`，从而让显卡接管计算。

## 📦 环境要求与安装指南

### 1. 创建虚拟环境
建议使用虚拟环境隔离项目依赖。
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

### 2. 安装依赖库

根据你选择的脚本，安装对应的依赖：

**通用依赖（所有脚本都需要）：**
```bash
python -m pip install --upgrade pip
pip install dghs-imgutils scikit-learn hdbscan pillow numpy pyyaml
```

**选择安装 ONNX Runtime 运行后端：**
- **Nvidia 显卡 (运行 `WaifuCluster.py`)**:
  需要安装 CUDA 12.x + cuDNN 9.x 环境，并安装 GPU 版 ONNX Runtime：
  ```bash
  pip install onnxruntime-gpu
  ```
- **AMD/Intel 显卡 (运行 DirectML 系列脚本)**:
  无需繁琐的 CUDA 配置，只需安装 DirectML 版 ONNX Runtime：
  ```bash
  pip install onnxruntime-directml
  ```
- **仅使用 CPU (应急退路)**:
  ```bash
  pip install onnxruntime
  ```

## 🚀 快速开始

1.  **准备图片**：将你需要整理的动漫图片放入项目根目录下的 `./fanart` 文件夹中。
2.  *(可选)* **配置代理**：DirectML 系列脚本默认开启了代理 `http://127.0.0.1:7897`（用于下载 HuggingFace 模型）。如果你的网络不需要代理，请注释掉脚本开头的 `os.environ["HTTP_PROXY"]` 和 `os.environ["HTTPS_PROXY"]`。
3.  **运行脚本**：根据你的硬件选择对应脚本运行。例如使用 DirectML + HDBSCAN：
    ```bash
    python WaifuClusterDirectML.py
    ```
4.  **查看结果**：脚本运行完成后，会在 `./sorted` 目录下创建多个文件夹（`character_000`, `character_001`, ...），包含被判定为同一角色的图片。无法明确分组的图片（如果有）会放入 `noise` 文件夹。中间产生的裁切图保存在 `./crops` 目录，距离矩阵保存在 `diff_matrix.npy`。

## ⚙️ 配置参数说明

不同脚本的聚类参数有所不同，你可以打开对应的 `.py` 文件直接修改：

**1. 检测参数 (通用)：**
*   `score < 0.5`：YOLO 头部检测的置信度阈值，低于此值的检测框会被忽略。
*   `w // 4` / `h // 4`：检测框向外扩展的比例（默认 1/4），避免发型被切。

**2. HDBSCAN 参数 (`WaifuCluster.py` / `WaifuClusterDirectML.py`)：**
*   `min_cluster_size=2`：形成一个角色簇所需的最少图片数量。
*   `min_samples=1`：核心距离计算时的邻居数。
*   `cluster_selection_epsilon`：距离阈值（CUDA版为0.05，DirectML版默认0.08）。值越小，同类要求越严格。
*   `cluster_selection_method="leaf"`：簇提取策略，`leaf` 倾向于生成更细碎的簇，`eom` 倾向于更大的簇。

**3. OPTICS 参数 (`WaifuClusterDirectMLOptics.py`)：**
*   `min_samples=2`：相当于最小簇大小，控制聚类的严格度。

**4. AHC 参数 (`WaifuClusterAHC.py`)：**
*   `distance_threshold=0.116`：层次聚类的合并距离阈值。越小同类要求越严格。脚本会打印出距离矩阵的 min/max/mean/median，建议根据 median 进行微调。
*   *注意：AHC 算法默认会将所有样本强制分入某个簇，不会产生 `noise` 文件夹。*

## 🔧 工作原理简介

1.  **人物检测裁切**：使用 YOLO 模型 (`head_detect_v2.0_x_yv11`) 定位图片中的角色头部区域，并适当扩边裁切成单人子图。
2.  **特征提取**：使用 **CCIP** 模型 (`deepghs/ccip`) 将每个单人子图编码为特征向量。
3.  **距离计算**：计算所有特征向量两两之间的差异，生成一个 N×N 的距离矩阵 (`diff_matrix.npy`)。
4.  **密度/层次聚类**：根据所选脚本，使用 HDBSCAN、OPTICS 或 凝聚层次聚类 (AHC) 对距离矩阵进行计算，得出聚类标签。
5.  **文件整理**：根据聚类标签，将原图或裁切图复制到 `./sorted` 目录对应的角色文件夹中。

## ❓ 常见问题

**Q: 运行 DirectML 脚本报错 `DmlExecutionProvider` not available？**

**A:** 请确保正确安装了 `onnxruntime-directml`。如果同时安装了 `onnxruntime-gpu` 可能会产生冲突，建议在虚拟环境中仅保留一个版本的 onnxruntime。

**Q: N 卡运行 `WaifuCluster.py` 报 `cublasLt64_12.dll missing`？**

**A:** 脚本默认硬编码了 CUDA 12.4 的路径 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`)。如果你的 CUDA 版本或安装路径不同，请修改脚本顶部的 `cuda_bin_path` 变量。

**Q: 聚类效果不好，同一个角色被分成了好几个文件夹？**

**A:** 这是密度/层次聚类的常见现象。你可以尝试调大 `distance_threshold` (AHC) 或 `cluster_selection_epsilon` (HDBSCAN)，或者将 HDBSCAN 的 `cluster_selection_method` 改为 `"eom"`。

**Q: 首次运行卡在加载模型/下载模型很慢？**

**A:** 模型权重从 HuggingFace Hub 下载。DirectML 脚本已内置代理设置，如果你没有开启代理软件，请务必注释掉脚本开头的代理代码，否则可能导致网络连接失败。下载完成后模型会缓存在项目目录的 `.hf_cache` 中。

## 🙏 致谢

*   [DeepGHS](https://github.com/deepghs) 团队：提供了强大的 `imgutils` 库、CCIP 模型和动漫人物检测模型。
*   [scikit-learn](https://scikit-learn.org/) 和 [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) 社区：提供了核心的聚类算法实现。
*   [ONNX Runtime](https://onnxruntime.ai/) 与 DirectML：提供了高效的跨平台及跨硬件模型推理引擎。