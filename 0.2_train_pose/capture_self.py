# -*- coding: utf-8 -*-
"""
自采集录屏脚本（自训练领域适应数据）

功能：打开摄像头，10 秒倒计时后每秒保存一帧，共 30 帧，
     存入 self_captured/ 目录，供后续标注头部中心点。

使用方法：
  cd F:\00_MK\TensorFlow\0.2_train_pose; ..\.venv\Scripts\python.exe capture_self.py
  按 'q' 可提前退出

建议录 3 轮：正常坐姿 / 转头晃动 / 靠近与远离摄像头，
每轮录制期间尽量覆盖典型场景（含杂物、风扇等干扰物）。
"""
import cv2
import os
import time

SAVE_DIR = os.path.join(os.path.dirname(__file__), "self_captured")
NUM_FRAMES = 30       # 录制帧数
INTERVAL = 1.0        # 每帧间隔（秒）


def main():
    """打开摄像头倒计时后逐帧保存。"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：摄像头打开失败")
        return

    print(f"摄像头已开启，10 秒后开始录制 {NUM_FRAMES} 帧（间隔 {INTERVAL}s）...")
    time.sleep(10)

    count = 0
    while count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("错误：读取帧失败")
            break

        # 文件名用已存在数量继续编号，避免覆盖已录数据
        existing = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
        fname = os.path.join(SAVE_DIR, f"{existing:03d}.jpg")
        cv2.imwrite(fname, frame)
        print(f"已存 {os.path.basename(fname)} ({count + 1}/{NUM_FRAMES})")

        cv2.imshow("Capture (press 'q' to exit)", frame)
        if cv2.waitKey(int(INTERVAL * 1000)) & 0xFF == ord("q"):
            break
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"录制结束，本次保存 {count} 帧到 {SAVE_DIR}")


if __name__ == "__main__":
    main()
