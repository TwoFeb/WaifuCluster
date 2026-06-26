import os
import time
import glob
import shutil
import numpy as np
from PIL import Image
from imgutils.detect import detect_heads
from imgutils.metrics import ccip_batch_differences
import hdbscan
import onnxruntime as ort

# ---------- 设置代理 ----------
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# ---------- 计时开始 ----------
total_start = time.perf_counter()

# ---------- 硬件与路径设置 ----------
# 屏蔽 CUDA 环境变量，防止残留的配置干扰 DirectML
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, ".hf_cache")

# 检查当前可用的计算提供者
# 成功安装 onnxruntime-directml 后，这里应显示包含 'DmlExecutionProvider'
available_providers = ort.get_available_providers()
print("当前支持的 ONNX Providers:", available_providers)
if 'DmlExecutionProvider' not in available_providers:
    print("⚠️ 警告: 未检测到 DmlExecutionProvider，请确认是否已正确安装 onnxruntime-directml。程序可能会回退到 CPU 计算。")

# ---------- 定义文件夹 ----------
SRC_DIR = "./fanart"          # 原始图库
CROP_DIR = "./crops"          # 裁切后的人物子图
OUT_DIR = "./sorted"          # 最终按角色分文件夹输出

# ---------- 清空并重建 crops/ 和 sorted/ ----------
for dir_path in [CROP_DIR, OUT_DIR]:
    shutil.rmtree(dir_path, ignore_errors=True)   # 删除
    os.makedirs(dir_path, exist_ok=True)          # 新建空目录

# ---------- 1. 裁切阶段 ----------
stage_start = time.perf_counter()
crop_records = []
for img_path in glob.glob(os.path.join(SRC_DIR, "*.*")):   # 把每张图里的每个角色单独存一张
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

print(f"✅ 裁切完成，共提取 {len(crop_records)} 个头部，耗时 {time.perf_counter() - stage_start:.2f} 秒")

# ---------- 2. CCIP 特征提取 ----------
stage_start = time.perf_counter()
crop_paths = [r[0] for r in crop_records]
diff_matrix = ccip_batch_differences(crop_paths)   # 越小越相似
print("min =", diff_matrix.min())
print("max =", diff_matrix.max())
print("mean =", diff_matrix.mean())
print("median =", np.median(diff_matrix))
np.save("./diff_matrix.npy", diff_matrix)
print(f"✅ 特征提取完成，耗时 {time.perf_counter() - stage_start:.2f} 秒")

# ---------- 3. HDBSCAN 聚类 ----------
stage_start = time.perf_counter()
diff_matrix_64 = diff_matrix.astype(np.float64)
clusterer = hdbscan.HDBSCAN(
    metric="precomputed", # 使用预计算的距离矩阵
    min_cluster_size=2,   # 至少2张才算一个角色簇，可调
    min_samples=1,        # 至少1张才算一个核心点，可调
    cluster_selection_epsilon=0.05,  # 聚类时的距离阈值，可调
    cluster_selection_method="leaf", #  "eom" or "leaf"
)
labels = clusterer.fit_predict(diff_matrix_64)
n_noise = list(labels).count(-1)
print(f"噪声率: {n_noise / len(labels):.1%}")
print(f"✅ 聚类完成，耗时 {time.perf_counter() - stage_start:.2f} 秒")

# ---------- 4. 落盘 ----------
stage_start = time.perf_counter()
for (crop_path, src_path, _), label in zip(crop_records, labels):
    if label == -1:
        target_dir = os.path.join(OUT_DIR, "noise")
    else:
        target_dir = os.path.join(OUT_DIR, f"character_{label:03d}")
    os.makedirs(target_dir, exist_ok=True)
    # 复制原图（而非裁切图），方便人工核对
    shutil.copy2(src_path, target_dir)

print(f"✅ 落盘完成，耗时 {time.perf_counter() - stage_start:.2f} 秒")

# ---------- 汇总 ----------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"共 {n_clusters} 个角色簇，噪声 {list(labels).count(-1)} 张")
print(f"🏁 整个脚本运行总耗时 {time.perf_counter() - total_start:.2f} 秒")