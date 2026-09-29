# -*- coding: utf-8 -*-
r"""
摄像头 + 人脸检测 Demo：打开本机摄像头，实时检测并框出人脸，按 q 退出。

运行方式：
  .\.venv\Scripts\Activate.ps1
  python camera_demo.py

操作说明：
  - 窗口打开后显示实时画面，人脸会被绿框标出
  - 按键盘 q 退出程序
  - 关闭摄像头窗口也会退出

原理：
  使用 OpenCV 的 DNN 模块加载 SSD 人脸检测模型（deploy.prototxt +
  res10_300x300_ssd_iter_140000.caffemodel），对每一帧做前向推理得到
  人脸框坐标与置信度。若模型下载失败，自动降级到 Haar 级联分类器
  （OpenCV 自带，无需下载），保证弱网环境仍能用。
"""

import os
import time
import urllib.request

import cv2
import numpy as np


# ========== 模型文件路径 ==========
# 模型放在脚本同级目录的 models 文件夹里，避免污染源码目录
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
HAAR_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")

# 模型下载源：优先国内镜像（ghproxy），再试 CDN、Gitee，最后 GitHub 原始地址
PROTOTXT_URLS = [
    "https://ghproxy.net/https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "https://cdn.jsdelivr.net/gh/opencv/opencv@master/samples/dnn/face_detector/deploy.prototxt",
    "https://gitee.com/mirrors/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
]
CAFFEMODEL_URLS = [
    "https://ghproxy.net/https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "https://gh-proxy.com/https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "https://cdn.jsdelivr.net/gh/opencv/opencv_3rdparty@dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
]
HAAR_URLS = [
    "https://ghproxy.net/https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_default.xml",
    "https://gitee.com/mirrors/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
]


def download_file(urls, save_path, timeout=30):
    """依次尝试 urls 列表下载文件到 save_path，全部失败才抛异常。

    原理：
      - raw.githubusercontent.com 在国内经常超时，所以优先用 Gitee 镜像
      - 加 User-Agent 模拟浏览器，避免被服务器拦截
      - 设 30 秒超时，单个源卡住不会无限等待
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"下载模型: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with open(save_path, "wb") as f:
                    f.write(resp.read())
            print(f"已保存到: {save_path}")
            return
        except Exception as e:
            last_err = e
            print(f"  失败: {e}")
    raise RuntimeError(f"所有下载源均失败: {last_err}")


class FaceDetector:
    """人脸检测器：优先用 DNN（精度高），加载失败则降级到 Haar。"""

    def __init__(self, conf_threshold=0.5):
        """初始化检测器。

        参数：
          conf_threshold: DNN 模式下的置信度阈值，高于此值才认为是人脸。
                          值越大误检越少，但可能漏检侧脸/小脸。
        """
        self.conf_threshold = conf_threshold
        self.model_type = "DNN"
        self.net = None
        self.haar = None
        self._init_dnn() or self._init_haar()

    def _init_dnn(self):
        """尝试加载 DNN 模型，成功返回 True，失败返回 False。

        原理：readNetFromCaffe 读取 Caffe 格式的模型定义和权重。
        首次运行会自动下载两个文件到 models/ 目录。
        """
        try:
            if not os.path.exists(PROTOTXT_PATH):
                download_file(PROTOTXT_URLS, PROTOTXT_PATH)
            if not os.path.exists(CAFFEMODEL_PATH):
                download_file(CAFFEMODEL_URLS, CAFFEMODEL_PATH)
            self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
            print("人脸检测模型: DNN (SSD)")
            return True
        except Exception as e:
            print(f"DNN 模型加载失败，降级到 Haar: {e}")
            return False

    def _init_haar(self):
        """加载 Haar 级联分类器作为兜底方案。

        原理：OpenCV 4.x 自带 Haar XML 文件（cv2.data.haarcascades 目录），
        若没有则自动下载。Haar 速度快但精度（尤其侧脸/遮挡）不如 DNN。
        """
        try:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(haar_path):
                if not os.path.exists(HAAR_PATH):
                    download_file(HAAR_URLS, HAAR_PATH)
                haar_path = HAAR_PATH
            self.haar = cv2.CascadeClassifier(haar_path)
            if self.haar.empty():
                raise RuntimeError("Haar 分类器加载后为空")
            self.model_type = "Haar"
            print("人脸检测模型: Haar Cascade (兜底)")
            return True
        except Exception as e:
            print(f"Haar 分类器也不可用: {e}")
            return False

    def detect(self, frame):
        """检测一帧图像中的人脸，返回人脸框列表 [(x, y, w, h), ...]。

        原理：
          DNN 模式：
            1. blobFromImage 把图像缩放到 300x300 并做均值归一化，转成网络输入
            2. net.forward 做前向推理，输出形状为 [1,1,N,7]，
               每一行 = [0, label, confidence, x1, y1, x2, y2]（归一化坐标）
            3. 过滤置信度低于阈值的，把归一化坐标还原成像素坐标
          Haar 模式：
            1. 转灰度图（Haar 只需亮度信息）
            2. detectMultiScale 多尺度滑动窗口检测，返回 (x,y,w,h) 列表
        """
        if self.net is not None:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0),
            )
            self.net.setInput(blob)
            detections = self.net.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
            return faces
        elif self.haar is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return []


def open_camera(index=0):
    """尝试打开指定索引的摄像头，成功返回 VideoCapture 对象。

    原理：不同系统摄像头索引可能不同，这里先用给定索引，
    失败时依次尝试 0~2。CAP_DSHOW 是 Windows DirectShow 后端，
    比默认后端打开更快、更稳定。
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"已打开摄像头，索引: {i}")
                return cap
            cap.release()
        raise RuntimeError("无法打开任何摄像头，请检查设备是否被其他程序占用或驱动是否正常")
    print(f"已打开摄像头，索引: {index}")
    return cap


def draw_faces(frame, faces):
    """在画面上画出人脸框，并返回检测到的人脸数量。

    原理：对每个 (x,y,w,h) 框调用 rectangle 画矩形，
    绿色线条、线宽 2，不填充。
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return len(faces)


def main():
    """主流程：打开摄像头 → 初始化检测器 → 循环检测并显示 → 按 q 退出。"""
    cap = open_camera(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = FaceDetector(conf_threshold=0.5)

    print("按 q 退出")
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败，退出")
            break

        # 水平镜像，让画面符合"照镜子"直觉
        frame = cv2.flip(frame, 1)

        # 检测人脸
        faces = detector.detect(frame)
        draw_faces(frame, faces)

        # 计算 FPS：当前帧与上一帧的时间差的倒数
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0
        prev_time = cur_time

        # 画面左上角叠加状态信息
        cv2.putText(frame, f"Model: {detector.model_type}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("Camera Face Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
