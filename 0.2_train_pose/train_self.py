# -*- coding: utf-8 -*-
"""
纯自采集数据从头训练脚本（个人专用头部检测器，灰度单通道）

只使用 self_captured/ 里自己标注的帧（有头帧为正样本，
按 n 标记的无头帧为负样本），网络从零初始化，不依赖任何外部数据集。
"过拟合到自己的使用环境"正是个人检测器的目标。

设计要点：
  - 数据极少（几十张），场景单一，强增强保证每张图每轮都不重复
  - 正样本增强：整帧补边 60% / 随机裁剪 40%，
    50% 水平翻转，亮度/对比度/Gamma/噪声光度增强
  - 每 batch 按 13 正 : 3 负混合（贴近自然有头/无头比例）
  - 从头训练用较大学习率 1e-3、较多轮次（小数据多重复）
  - 输入 192×192×1 灰度单通道：头部检测靠轮廓纹理，
    单通道让首层参数与端侧预处理量都减为 RGB 的 1/3

运行：python train_self.py
产出：models/head_model.keras
下一步：python convert_head.py
"""
import csv
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent
SELF_DIR = BASE_DIR / "self_captured"
LABEL_CSV = SELF_DIR / "head_labels.csv"
MODEL_DIR = BASE_DIR / "models"

IMG_SIZE = 192              # 网络输入边长（正方形灰度图）
HEATMAP_SIZE = 96           # 热度图边长（只下采样一次，定位更准）
BATCH_SIZE = 16
EPOCHS = 50                 # 从头训练轮数（小数据反复学，增强保证不重复）
STEPS_PER_EPOCH = 20        # 每轮 batch 数
LR = 1e-3                   # 从头训练学习率
NEG_PER_BATCH = 3           # 每 batch 无头负样本数（其余 13 格为正样本）
# 小数据 + 加权 MSE 处于临界平衡，网络初始化随机性决定收敛解域
# （锐峰解：脸 0.93/背景低；弥散解：脸 0.66/背景 0.55）：固定多个
# 初始化种子各训一遍，按训练集"有头峰值均值-无头峰值均值"
# 自动选分离度最好的一版保存；已有模型作为基线，全都更差则不覆盖
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)  # 旧数据(36负样本)下种子1~8分离度
                            # +0.114/+0.096/+0.090/+0.064/+0.074/+0.073/
                            # +0.095/+0.104（最优种子1=当前基线）；补拍27帧
                            # 无头负样本（共63帧）后数据分布已变，全部重测


def load_self_labels():
    """
    读取自采集标注 CSV，返回 (正样本列表, 无头帧路径列表)。

    正样本: [(图片路径, cx, cy), ...]（归一化坐标）
    无头帧: 标注时按 n 的帧（cx<0），作为"用户场景负样本"——
            教模型抑制自家置物架/风扇等特定干扰物。
    """
    positives, negatives = [], []
    with open(LABEL_CSV, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            cx, cy = float(row[1]), float(row[2])
            path = str(SELF_DIR / row[0])
            if cx < 0:
                negatives.append(path)
            else:
                positives.append((path, cx, cy))
    print(f"✓ 加载 {len(positives)} 条自采集标注，"
          f"{len(negatives)} 帧无头负样本")
    return positives, negatives


def self_full_image(frame, cx, cy):
    """
    整帧路径（与摄像头 demo 预处理一致）：补边成正方形再缩放，
    标签坐标同步映射到补边坐标系。返回 (192×192 灰度图, nx, ny)。
    """
    H, W = frame.shape[:2]
    side = max(W, H)
    top = (side - H) // 2
    left = (side - W) // 2
    padded = cv2.copyMakeBorder(
        frame, top, side - H - top, left, side - W - left,
        cv2.BORDER_REPLICATE)
    img = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
    nx = (cx * W + left) / side
    ny = (cy * H + top) / side
    return img, nx, ny


def self_random_crop(frame, cx, cy, rng):
    """
    随机裁剪路径：以标注头中心为参考随机取正方形窗口
    （边长 0.45~1.05 倍长边，中心随机偏移），覆盖不同构图与距离，
    越界用边缘像素填充。返回 (192×192 灰度图, nx, ny)。
    """
    H, W = frame.shape[:2]
    side = int(rng.uniform(0.45, 1.05) * max(W, H))
    hx, hy = cx * W, cy * H
    ccx = hx + float(rng.uniform(-0.3, 0.3)) * side
    ccy = hy + float(rng.uniform(-0.3, 0.3)) * side
    x0, y0 = int(round(ccx - side / 2)), int(round(ccy - side / 2))

    top = max(0, -y0)
    bottom = max(0, y0 + side - H)
    left = max(0, -x0)
    right = max(0, x0 + side - W)
    padded = cv2.copyMakeBorder(frame, top, bottom, left, right,
                                cv2.BORDER_REPLICATE)
    roi = padded[y0 + top:y0 + top + side, x0 + left:x0 + left + side]
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    # 头部在裁剪框中的归一化坐标（补边不影响相对位置）
    return img, (hx - x0) / side, (hy - y0) / side


def generate_head_heatmap(cx, cy, rx, ry, out_size=96):
    """
    生成单通道、小而锐的各向同性高斯斑（固定 sigma）。

    不让高斯大小编码头部尺寸（实测模型会把斑学弥散），
    只让模型学头部中心在哪：sigma 固定 2.5 像素，斑小而锐，
    argmax 定位准；rx/ry 仅为占位参数不参与形状。
    """
    yy, xx = np.mgrid[:out_size, :out_size].astype(np.float32)
    gx, gy = cx * out_size, cy * out_size
    sigma = 2.5
    g = np.exp(-(((xx - gx) ** 2) + ((yy - gy) ** 2)) / (2 * sigma ** 2))
    return g[..., None].astype(np.float32)


def augment_gray(crop, rng):
    """
    灰度光度增强：返回归一化 float32 (192,192,1) 图。

    依次做随机对比度/亮度、Gamma（重点模拟室内暗光与摄像头曝光差）、
    归一化后加轻微高斯噪声；水平翻转由调用方连同标签一起处理。
    """
    img = crop.astype(np.float32)
    alpha = float(rng.uniform(0.7, 1.3))        # 对比度
    beta = float(rng.uniform(-25, 25))          # 亮度
    img = np.clip(img * alpha + beta, 0, 255)
    gamma = float(rng.uniform(0.7, 1.4))        # Gamma（<1 提亮暗部）
    img = 255.0 * np.power(img / 255.0, gamma)
    img = img / 255.0
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    return img[..., None].astype(np.float32)


def weighted_mse(y_true, y_pred):
    """
    加权 MSE：头部像素权重 ×401、背景 ×1。

    小而锐的高斯斑（sigma=2.5，正样本等效像素仅约 40 个）下，
    权重 100 时正样本总权重仍小于背景，模型会走全零捷径；
    提到 400 后正/背景总权重大致均衡。
    """
    weight = 1.0 + 400.0 * y_true
    err = tf.square(y_true - y_pred) * weight
    return tf.reduce_sum(err) / tf.reduce_sum(weight)


def build_head_model():
    """
    构建头部检测网络：192×192×1 灰度单通道输入，仅下采样一次到 96 输出。

    保留 96×96 分辨率让小头也能被准确定位；
    单通道 sigmoid 输出头部热度图。
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)                 # 96×96
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


class SelfDataset(keras.utils.Sequence):
    """纯自采集数据加载器：每 batch 13 张有头 + 3 张无头。"""

    def __init__(self, positives, negatives, seed=0):
        """保存正/负样本列表并建随机数发生器（增强用）。"""
        self.positives = positives
        self.negatives = negatives
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """每轮固定 STEPS_PER_EPOCH 个 batch。"""
        return STEPS_PER_EPOCH

    def __getitem__(self, idx):
        """组一个 batch：正样本按序轮换（每张图均匀出现），负样本随机取。"""
        imgs = np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), np.float32)
        hms = np.zeros((BATCH_SIZE, HEATMAP_SIZE, HEATMAP_SIZE, 1), np.float32)
        n_pos = BATCH_SIZE - NEG_PER_BATCH

        for i in range(n_pos):
            p, cx, cy = self.positives[
                (idx * n_pos + i) % len(self.positives)]
            frame = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            # 60% 整帧补边（与 demo 输入一致），40% 随机裁剪（多构图）
            if self.rng.random() < 0.6:
                img, nx, ny = self_full_image(frame, cx, cy)
            else:
                img, nx, ny = self_random_crop(frame, cx, cy, self.rng)
            if self.rng.random() < 0.5:      # 水平翻转，中心 x 镜像
                img = img[:, ::-1]
                nx = 1.0 - nx
            imgs[i] = augment_gray(img, self.rng)
            hms[i] = generate_head_heatmap(
                nx, ny, 0.1, 0.1, HEATMAP_SIZE)  # 半径占位：锐斑只用中心

        for i in range(NEG_PER_BATCH):
            if not self.negatives:
                break
            slot = n_pos + i
            p = self.negatives[int(self.rng.integers(0, len(self.negatives)))]
            frame = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                img, _, _ = self_full_image(frame, 0.5, 0.5)  # 整帧补边
                imgs[slot] = augment_gray(img, self.rng)
            # 标签保持全零（负样本），读图失败的全零图同样合法
        return imgs, hms


def _frame_peak(model, path):
    """
    对单张图走无增强整帧补边预处理（与 head_demo 输入一致），
    返回热度图全局峰值；读图失败返回 0.0。
    """
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return 0.0
    img, _, _ = self_full_image(frame, 0.5, 0.5)   # 坐标占位，补边图与坐标无关
    x = (img.astype(np.float32) / 255.0)[None, :, :, None]
    return float(model.predict(x, verbose=0).max())


def eval_separation(model, positives, negatives):
    """
    在训练集无增强视图上评估正负样本峰值分离度。

    返回 (score, 正样本峰值数组, 负样本峰值数组)；
    score = 有头峰值均值 − 无头峰值均值，越大说明模型越敢在脸上
    给高分且越能压住杂物，据此在多个初始化种子间选优。
    """
    pos_peaks = np.array([_frame_peak(model, p) for p, _, _ in positives])
    neg_peaks = np.array([_frame_peak(model, n) for n in negatives])
    score = float(pos_peaks.mean() - neg_peaks.mean())
    return score, pos_peaks, neg_peaks


def train_once(seed, positives, negatives):
    """
    固定种子训练一版模型：设置全局随机种子（决定网络初始化与 shuffle），
    数据增强序列按同种子重建以保证各版本公平，训练 EPOCHS 轮后返回模型。
    """
    tf.keras.utils.set_random_seed(seed)
    model = build_head_model()
    model.compile(optimizer=keras.optimizers.Adam(LR), loss=weighted_mse)
    train_ds = SelfDataset(positives, negatives, seed=seed)
    model.fit(train_ds, epochs=EPOCHS, verbose=2)
    return model


def main():
    """加载标注 → 已有模型作基线 → 多种子各训一版 → 选分离度最好的保存。"""
    print("=" * 60)
    print("步骤 1/3: 加载自采集标注")
    positives, negatives = load_self_labels()
    if not positives:
        print("错误：未找到自采集标注，请先运行 capture_self.py + annotate_self.py")
        return
    print(f"正样本 {len(positives)} 张 / 负样本 {len(negatives)} 张，"
          f"每 batch {BATCH_SIZE - NEG_PER_BATCH} 正 + {NEG_PER_BATCH} 负")

    model_path = MODEL_DIR / "head_model.keras"
    print("=" * 60)
    print(f"步骤 2/3: 多种子训练（{len(SEEDS)} 个初始化各 {EPOCHS} 轮，自动选优）")
    # 已有模型作为基线候选：新一轮种子全部更差时保留旧模型不被覆盖
    best_weights, best_score, best_seed, best_stat = None, -9.0, None, None
    if model_path.exists():
        try:
            base_model = keras.models.load_model(str(model_path), compile=False)
            best_score, b_pp, b_nn = eval_separation(
                base_model, positives, negatives)
            best_weights = [w.copy() for w in base_model.get_weights()]
            best_seed, best_stat = "已有模型", (b_pp, b_nn)
            print(f"基线（已有模型）: 有头 {b_pp.mean():.3f} / 无头 {b_nn.mean():.3f}"
                  f" / 分离度 {best_score:+.3f}")
        except Exception as e:                       # 损坏文件不阻塞训练
            print(f"已有模型加载失败（{e}），忽略基线")
    for k, seed in enumerate(SEEDS, 1):
        print("-" * 60)
        print(f"种子 {seed}（{k}/{len(SEEDS)}）训练中…")
        model = train_once(seed, positives, negatives)
        score, pp, nn = eval_separation(model, positives, negatives)
        print(f"种子 {seed}: 有头峰值均值 {pp.mean():.3f}（最小 {pp.min():.3f}）"
              f" / 无头峰值均值 {nn.mean():.3f}（最大 {nn.max():.3f}）"
              f" / 分离度 {score:+.3f}")
        if score > best_score:
            best_score, best_seed, best_stat = score, seed, (pp, nn)
            best_weights = [w.copy() for w in model.get_weights()]
            print(f"  ↑ 当前最优（{best_seed}）")

    print("=" * 60)
    print("步骤 3/3: 保存最优模型")
    if best_seed == "已有模型":
        print(f"✓ 本轮种子均未超过已有模型（分离度 {best_score:+.3f}），保留原模型")
        return
    best_model = build_head_model()
    best_model.set_weights(best_weights)
    MODEL_DIR.mkdir(exist_ok=True)
    best_model.save(str(model_path))
    pp, nn = best_stat
    print(f"✓ 选用种子 {best_seed}（分离度 {best_score:+.3f}），已保存 {model_path}")
    print(f"  有头峰值 {pp.mean():.3f}（最小 {pp.min():.3f}），"
          f"无头峰值 {nn.mean():.3f}（最大 {nn.max():.3f}）")
    print("下一步: python convert_head.py")


if __name__ == "__main__":
    main()
