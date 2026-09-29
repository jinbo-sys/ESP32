# -*- coding: utf-8 -*-
r"""
训练层验证示例：用最简单的模型跑一次 model.fit() 并画出 loss 曲线。
对应《01_环境与软件清单.md》第七节验证清单：
  “能跑一次最简单的 model.fit() 并画出 loss 曲线”
运行方式：
  .\.venv\Scripts\Activate.ps1
  python demo_fit.py

【整体原理】
机器学习（监督学习）的本质：给定一堆"输入 → 正确答案"的样本，
让模型自动学到一个从输入映射到答案的函数。
本例的完整流程是神经网络的最小闭环：
  1. 造数据：生成两类已知标签的点
  2. 建模型：定义一个带可学习参数的函数 f(x; w)
  3. 训练：  用梯度下降不断调整 w，让预测误差（loss）变小
  4. 评估：  看模型在数据上的最终表现
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def build_data(n_samples=2000, seed=42):
    """构造一个简单的二分类合成数据集（两个高斯簇），无需联网下载。

    原理：真实世界的数据往往是"同一类样本聚集在特征空间的某个区域"。
    这里用二维高斯（正态）分布模拟这一现象：
      - 每个样本是一个二维点 (x, y)，可理解为传感器采到的两个特征值
      - 类别 0 的点围绕 (-1,-1) 散布，类别 1 的点围绕 (+1,+1) 散布
      - 两簇之间略有重叠（标准差 0.6），模拟真实数据的噪声，
        因此模型无法做到 100% 正确——这正是需要"学习边界"而非"死记规则"的原因
    """
    rng = np.random.default_rng(seed)  # 固定随机种子，保证每次运行结果可复现
    # 类别 0：均值在 (-1, -1) 附近，scale 是高斯分布的标准差，控制散布范围
    x0 = rng.normal(loc=-1.0, scale=0.6, size=(n_samples // 2, 2))
    # 类别 1：均值在 (+1, +1) 附近
    x1 = rng.normal(loc=1.0, scale=0.6, size=(n_samples // 2, 2))
    # 把两簇拼成完整数据集：x 是特征矩阵 (2000, 2)，y 是标签向量 (2000,)
    # 标签用 0/1 表示，是二分类问题的标准做法
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))]).astype(np.float32)
    # 打乱顺序：否则前半批全是类别 0、后半批全是类别 1，
    # 每个训练 batch 内类别极端不均，梯度方向会来回摆动，训练不稳定
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def build_model():
    """搭建一个最小的 Keras 模型：单隐层二分类网络。

    原理：神经网络就是"矩阵乘法 + 非线性激活"的堆叠。
      输入(2维) → Dense(8, relu) → Dense(1, sigmoid) → 输出(属于类别1的概率)
      - 第一层 Dense(8)：学 8 组权重，把 2 维输入映射到 8 维中间表示。
        单个神经元只能画一条直线做划分；多个神经元各自画线，
        组合起来就能逼近更复杂的边界。8 个对本任务绰绰有余。
      - relu 激活：f(x)=max(0,x)。没有它，两层矩阵乘法可合并为一层，
        网络再深也只能表示线性关系；非线性激活是网络能拟合复杂函数的关键。
      - 输出层 Dense(1, sigmoid)：sigmoid 把任意实数压到 (0,1) 区间，
        输出可直接解释为"属于类别 1 的概率"，>0.5 判为类别 1。
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    # compile 决定"怎么学"：
    # - optimizer="adam"：梯度下降的改进版，自动为每个参数调节步长，
    #   是目前最省心的默认选择。梯度下降的思想：loss 对权重求导（梯度），
    #   沿梯度反方向走一小步，loss 就会下降；反复迭代逼近最优。
    # - loss="binary_crossentropy"：二分类标准损失函数，
    #   预测概率离真实标签越远惩罚越大（且是指数级加大），给模型明确的纠错信号。
    # - metrics=["accuracy"]：只用于展示，不参与训练。
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(model, x, y, boundary_epochs=(1, 25)):
    """训练模型并返回训练历史 + 指定轮次的决策边界快照。

    原理：model.fit() 内部每个 epoch（完整过一遍数据）做的事：
      1. 前向传播：batch 数据喂进网络，算出预测概率
      2. 计算 loss：预测与真实标签的差距
      3. 反向传播：用链式法则求 loss 对每个权重的梯度
      4. 更新权重：optimizer 沿梯度反方向调整权重
      - batch_size=32：每次用 32 个样本估计梯度。比全量数据快，
        且小批量带来的随机性有助于跳出局部最优。
      - epochs=25：整个数据集反复学 25 遍，loss 逐渐收敛。
      - validation_split=0.2：留出 20% 数据不参与训练，只用于评估。
        对比 train loss 与 val loss 可判断过拟合：
        train 降而 val 升 = 过拟合（死记硬背，没学到规律）。

    边界快照原理：用一个 Keras 回调，在第 1 轮和第 25 轮结束时，
    让模型对整个平面的网格点做预测。预测概率的分布就构成了"决策边界"，
    可以直观看到模型从"几乎不会分"到"清晰切开两簇"的学习过程。
    """
    # 在数据范围内铺 200x200 网格，作为"整个平面"的采样点
    pad = 0.5
    gx, gy = np.meshgrid(
        np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, 200),
        np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, 200),
    )
    grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    snapshots = {}  # {epoch: 网格预测概率}

    class BoundarySnapshot(tf.keras.callbacks.Callback):
        """在指定 epoch 结束时，记录模型对整个平面的预测概率。"""

        def on_epoch_end(self, epoch, logs=None):
            cur = epoch + 1  # Keras 内部 epoch 从 0 计，转成人类习惯的从 1 计
            if cur in boundary_epochs:
                prob = self.model.predict(grid, verbose=0).reshape(gx.shape)
                snapshots[cur] = prob
                print(f"\n[快照] 已记录第 {cur} 轮的决策边界")

    history = model.fit(
        x, y,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[BoundarySnapshot()],
    )
    return history, snapshots, (gx, gy)


def plot_loss(history, save_path="loss_curve.png"):
    """画出训练/验证 loss 曲线并保存为图片（不依赖图形窗口）。

    原理：loss 曲线是诊断训练过程最重要的图：
      - 两条曲线都下降并趋于平稳 → 训练正常收敛（本例应如此）
      - train loss 降、val loss 回升 → 过拟合，需要正则化/减少参数量/更多数据
      - 两条都不降 → 学习率不当、数据有问题或模型结构不合适
    """
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"loss 曲线已保存: {save_path}")


def plot_boundary_comparison(snapshots, grid_shape, x, y, save_path="boundary_compare.png"):
    """把第 1 轮与第 25 轮的决策边界画在一起对比。

    原理：背景颜色 = 模型输出的"属于类别 1 的概率"（蓝=类别0，红=类别1），
    圆点是真实样本。第 1 轮时权重接近随机初始化，概率分布几乎一片模糊；
    第 25 轮时模型已学会沿对角线把平面切成两半，边界清晰且贴合数据分布。
    """
    gx, gy = grid_shape
    epochs = sorted(snapshots.keys())
    fig, axes = plt.subplots(1, len(epochs), figsize=(6 * len(epochs), 5))
    for ax, ep in zip(axes, epochs):
        # 画概率背景：pcolormesh 把网格预测值渲染成色块
        ax.pcolormesh(gx, gy, snapshots[ep], cmap="RdBu_r", vmin=0, vmax=1, alpha=0.6)
        # 画 0.5 概率等高线 = 决策边界的精确位置
        ax.contour(gx, gy, snapshots[ep], levels=[0.5], colors="black", linewidths=1.5)
        # 叠加真实样本点（只画 400 个避免太密）
        ax.scatter(x[:400, 0], x[:400, 1], c=y[:400], cmap="RdBu_r",
                   edgecolors="k", linewidths=0.3, s=12)
        ax.set_title(f"Epoch {ep}")
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
    fig.suptitle("Decision Boundary: Epoch 1 vs Epoch 25")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"决策边界对比图已保存: {save_path}")


def main():
    """主流程：造数据 → 建模型 → 训练 → 画曲线/边界对比 → 评估。

    注：最后 evaluate 用的是包含训练集的完整数据，精度会偏乐观，
    本例只为验证环境可用；正式项目必须用模型从未见过的独立测试集。
    """
    x, y = build_data()
    model = build_model()
    model.summary()  # 打印网络结构和参数量：本例仅 33 个参数，端侧模型就是这么小
    history, snapshots, grid_shape = train(model, x, y)
    plot_loss(history)
    plot_boundary_comparison(snapshots, grid_shape, x, y)
    loss, acc = model.evaluate(x, y, verbose=0)
    print(f"最终精度: {acc:.4f}, 最终 loss: {loss:.4f}")


if __name__ == "__main__":
    main()
