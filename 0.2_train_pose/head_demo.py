# -*- coding: utf-8 -*-
"""
自训练头部检测模型（灰度输入版）- 摄像头实时 Demo
==================================================
只识别头部位置并画椭圆圈出，不画躯干、手臂。

模型输入 192×192×1 灰度图，输出 96×96×1 热度图：
  - argmax 找峰值 + 局部对比度双重判据判断有无头
  - 峰值 3×3 邻域亚像素重心 = 头部中心
  - 椭圆大小取画面固定比例

运行:
  ../.venv/Scripts/python.exe head_demo.py
按 q 退出。
"""
import os
import time
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "head_model_int8.tflite")
HEATMAP_SIZE = 96
PEAK_THRESHOLD = 0.60       # 跟踪期峰值下限：实机背景响应阶梯 置物架≈0.50、
                            # 桌面杂物≈0.57，0.45/0.55 都被兜住导致跟踪框
                            # 滞留杂物；0.60 高于背景、与初始化 0.65 配套，
                            # 丢 30 帧即重置走重新初始化
PEAK_INIT = 0.65            # 初始化峰值下限（实测定版）：0.68 过严，脸分数常落
                            # 0.6x 导致正脸迟迟锁不上；0.65 严进宽跟语义自洽
                            # （0.65 > 跟踪 0.60 > 背景最高 0.57），且初始化仅
                            # 搜画面中央区+5 帧中位数，边缘杂物进不来
CONTRAST_MARGIN = 0.12      # 峰值要高出周围背景这么多才算头（抗误检）
HEAD_RATIO_X = 0.11         # 椭圆横向半轴占画面宽度比例（罩住整脸）
HEAD_RATIO_Y = 0.14         # 椭圆纵向半轴占画面高度比例（脸比宽略长）
HEAD_OFFSET_Y_RATIO = -0.04 # 中心 Y 补偿：微调后峰值已落头部，仅需轻微上移（-0.14 过高只罩上半脸）
INIT_FRAMES = 5             # 启动时收集多少帧候选点做中位数初始化
ROI_HALF = 0.18             # 跟踪时局部搜索窗口半边长（归一化），窗外的强纹理（如风扇）不予理会
REACQUIRE_LIMIT = 0.35      # 局部窗口丢失后，全局重捕获允许的最大位移
EMA_ALPHA = 0.5             # 中心指数平滑系数（越大跟手、越小越稳）
RESET_FRAMES = 30           # 连续多少帧无有效检测才放弃跟踪


class HeadDetector:
    """头部检测器：加载单通道 .tflite 模型，输出头部椭圆参数。"""

    def __init__(self):
        """加载模型并读取输入尺寸。模型不存在时给出明确提示。"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"找不到模型 {MODEL_PATH}\n"
                "请先依次运行: python train_self.py 和 python convert_head.py")

        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_w = self.input_detail["shape"][2]
        self.input_h = self.input_detail["shape"][1]
        # 时间滤波状态：track=(cx,cy) 平滑后的跟踪点；lost=连续无效帧计数
        self.track = None
        self.lost = 0
        self.init_points = []   # 启动期候选点，满 INIT_FRAMES 个后取中位数建跟踪
        print(f"✓ 头部模型: {os.path.basename(MODEL_PATH)} "
              f"输入 {self.input_detail['shape']} 输出 {self.output_detail['shape']}")

    def _infer_heatmap(self, frame):
        """
        灰度预处理 + 推理，返回 96×96 热度图。

        与训练的整图路径保持一致：1280×720 先用边缘像素上下补边成
        正方形，再缩放到 192×192（直接拉伸会横向压扁 1.78 倍，
        且补边后人脸有效放大约 1.56 倍，更贴近训练分布）。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        side = max(w, h)
        if w != h:
            pad = side - min(w, h)
                    # 竖屏补左右、横屏补上下
            if w < h:
                gray = cv2.copyMakeBorder(gray, 0, 0, pad // 2, pad - pad // 2,
                                          cv2.BORDER_REPLICATE)
            else:
                gray = cv2.copyMakeBorder(gray, pad // 2, pad - pad // 2, 0, 0,
                                          cv2.BORDER_REPLICATE)
        img = cv2.resize(gray, (self.input_w, self.input_h))
        if self.input_detail["dtype"] == np.uint8:
            x = img.astype(np.uint8)[None, :, :, None]
        else:
            x = (img.astype(np.float32) / 255.0)[None, :, :, None]

        self.interpreter.set_tensor(self.input_detail["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_detail["index"])[0, :, :, 0]

    @staticmethod
    def _peak_in_roi(hm, u0, v0, u1, v1, peak_min):
        """
        在热度图矩形 ROI（归一化坐标）内找峰值并做双重判据和亚像素定位。

        返回 (nx, ny, peak)；ROI 内无合格峰返回 None。
        判据：峰值绝对高度 ≥peak_min，且高出峰邻域外的
        全局背景 ≥CONTRAST_MARGIN（背景在调用方固定，ROI 内外共用）。
        """
        h, w = hm.shape
        x0, x1 = int(u0 * w), int(u1 * w)
        y0, y1 = int(v0 * h), int(v1 * h)
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        roi = hm[y0:y1, x0:x1]

        ly, lx = np.unravel_index(np.argmax(roi), roi.shape)
        py, px = ly + y0, lx + x0
        peak = float(hm[py, px])
        if peak < peak_min:
            return None

        # 背景基线：挖掉峰值周围 11×11 区域后的全图均值
        bg_mask = np.ones_like(hm, bool)
        ay0, ay1 = max(py - 5, 0), min(py + 6, h)
        ax0, ax1 = max(px - 5, 0), min(px + 6, w)
        bg_mask[ay0:ay1, ax0:ax1] = False
        bg = float(hm[bg_mask].mean())
        if (peak - bg) < CONTRAST_MARGIN:
            return None

        # 亚像素定位：3×3 邻域加权重心（只计高于 (peak+bg)/2 的像素）
        sub_level = (peak + bg) / 2.0
        ys, xs = np.mgrid[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
        wts = np.clip(hm[ys, xs] - sub_level, 0, None)
        if wts.sum() > 0:
            return (float((xs * wts).sum() / wts.sum()) / w,
                    float((ys * wts).sum() / wts.sum()) / w, peak)
        return px / w, py / h, peak

    def detect(self, frame):
        """
        带局部窗口跟踪的检测，返回 (cx, cy, ax, ay, score)；未锁定时 None。

        为什么不用全局 argmax + 硬限幅：灰度模型会被风扇等强纹理抢走
        全局峰，硬限幅又会把跟踪焊死在错误首帧上（真实头永远被当跳变
        拒绝）。改为：
          1. 启动期：连续收集 INIT_FRAMES 帧"画面中央区域"的峰，
             取横纵坐标中位数建立跟踪（中位数天然排除边缘风扇离群点）
          2. 跟踪期：只在上一帧中心 ±ROI_HALF 的局部窗口内找峰，
             窗外的任何强响应都不参与选择
          3. 窗口内暂时丢峰时，允许一次 ≤REACQUIRE_LIMIT 的全局重捕获
             （应对头快速移动）；再丢则保持原位，连续 RESET_FRAMES
             帧无效才整体重置
          4. EMA 平滑 + 固定 Y 偏置补偿灰度模型的系统性下偏
        """
        H, W = frame.shape[:2]
        hm = self._infer_heatmap(frame)

        if self.track is None:
            # ---- 启动/重置阶段：内容区取点，中位数初始化 ----
            # 横屏补边后真实画面只占 v∈[pad,1-pad]，搜索区排除补边条带
            side0 = max(W, H)
            v_pad = (side0 - H) / 2 / side0
            u_pad = (side0 - W) / 2 / side0
            r = self._peak_in_roi(
                hm, u_pad + 0.15 * (W / side0), v_pad + 0.10 * (H / side0),
                1 - u_pad - 0.15 * (W / side0), 1 - v_pad - 0.10 * (H / side0),
                PEAK_INIT)          # 初始化用高阈值，置物架类杂物进不来
            if r is not None:
                self.init_points.append((r[0], r[1]))
            if len(self.init_points) < INIT_FRAMES:
                return None
            arr = np.asarray(self.init_points)
            self.track = [float(np.median(arr[:, 0])),
                          float(np.median(arr[:, 1]))]
            self.lost = 0
            peak = r[2] if r else 0.0
        else:
            # ---- 跟踪阶段：局部窗口优先 ----
            tx, ty = self.track
            r = self._peak_in_roi(
                hm,
                max(tx - ROI_HALF, 0.0), max(ty - ROI_HALF, 0.0),
                min(tx + ROI_HALF, 1.0), min(ty + ROI_HALF, 1.0),
                PEAK_THRESHOLD)     # 跟踪期用低阈值，容忍分数波动

            if r is None:
                # 局部窗口丢峰：允许位移连续的全局重捕获
                g = self._peak_in_roi(hm, 0.0, 0.0, 1.0, 1.0, PEAK_THRESHOLD)
                if g is not None and np.hypot(g[0] - tx, g[1] - ty) <= REACQUIRE_LIMIT:
                    r = g

            if r is not None:
                self.track[0] += EMA_ALPHA * (r[0] - self.track[0])
                self.track[1] += EMA_ALPHA * (r[1] - self.track[1])
                self.lost = 0
                peak = r[2]
            else:
                self.lost += 1
                peak = 0.0
                if self.lost >= RESET_FRAMES:
                    self.track = None
                    self.init_points = []
                    return None

        # 正方形补边坐标系 → 原帧坐标系（内容居中，扣除上下/左右补边）
        side = max(W, H)
        cx = int(self.track[0] * side - (side - W) / 2)
        cy = int(self.track[1] * side - (side - H) / 2 + HEAD_OFFSET_Y_RATIO * H)
        ax = int(W * HEAD_RATIO_X)
        ay = int(H * HEAD_RATIO_Y)
        return cx, cy, ax, ay, peak


def main():
    """主循环：开摄像头 → 推理 → 只画头部椭圆和状态 → q 退出。"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HeadDetector()
    print("按 q 退出")
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # 镜像，照镜子效果

        head = detector.detect(frame)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now

        if head is not None:
            cx, cy, ax, ay, score = head
            # 青色椭圆圈出头部
            cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360,
                        (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            status, color = f"Head: Yes ({score:.2f})", (0, 255, 0)
        else:
            status, color = "Head: No", (0, 0, 255)

        cv2.putText(frame, "Self-trained Head Detector", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Head Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("Head Demo", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
