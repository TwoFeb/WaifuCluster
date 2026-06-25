import os
# 【新增】强行指定 Windows 加载 CUDA DLL 的路径
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)
else:
    print(f"警告：未找到 CUDA 安装路径: {cuda_bin_path}")
# 锁死独显：在 CUDA 逻辑里，你的 RTX 3070 是第 0 块卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, ".hf_cache")

import glob, shutil
import numpy as np
from PIL import Image
from imgutils.detect import detect_heads
from imgutils.metrics import ccip_batch_differences
import hdbscan
import onnxruntime as ort

print("Providers before imports:", ort.get_available_providers())

SRC_DIR = "./fanart"          # 原始图库
CROP_DIR = "./crops"          # 裁切后的人物子图
OUT_DIR = "./sorted"          # 最终按角色分文件夹输出
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 1. 检测+裁切：把每张图里的每个角色单独存一张
crop_records = []   # [(crop_path, src_path, head_idx), ...]
for img_path in glob.glob(os.path.join(SRC_DIR, "*.*")):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        continue
    heads = detect_heads(img, model_name='head_detect_v2.0_x_yv11')
    for i, (bbox, label, score) in enumerate(heads):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = bbox
        # 适当向外扩 1/4，避免发型被切掉
        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - w // 4); y1 = max(0, y1 - h // 4)
        x2 = min(img.width, x2 + w // 4); y2 = min(img.height, y2 + h // 4)
        crop = img.crop((x1, y1, x2, y2))
        crop_name = f"{os.path.basename(img_path)}__{i}.png"
        crop_path = os.path.join(CROP_DIR, crop_name)
        crop.save(crop_path)
        crop_records.append((crop_path, img_path, i))

# 2. CCIP 提取特征：直接用 imgutils 的批量接口
#    ccip_batch_differences 返回 N×N 距离矩阵，转成预距离矩阵给 HDBSCAN
crop_paths = [r[0] for r in crop_records]
diff_matrix = ccip_batch_differences(crop_paths)   # 越小越相似
np.save("./diff_matrix.npy", diff_matrix)

# 3. HDBSCAN 聚类（precomputed 距离）

# 【修复】将矩阵强制转换为 float64 类型，并确保数据是连续的
diff_matrix_64 = diff_matrix.astype(np.float64)
clusterer = hdbscan.HDBSCAN(
    metric="precomputed",
    min_cluster_size=3,        # 至少3张才算一个角色簇，可调
    min_samples=2,
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(diff_matrix.astype(np.float64))

# 4. 落盘
for (crop_path, src_path, _), label in zip(crop_records, labels):
    if label == -1:
        target_dir = os.path.join(OUT_DIR, "noise")
    else:
        target_dir = os.path.join(OUT_DIR, f"character_{label:03d}")
    os.makedirs(target_dir, exist_ok=True)
    # 复制原图（而非裁切图），方便人工核对
    shutil.copy2(src_path, target_dir)

print(f"共 {len(set(labels)) - (1 if -1 in labels else 0)} 个角色簇，"
      f"噪声 {list(labels).count(-1)} 张")