# -*- coding: utf-8 -*-
"""
自采集帧标注工具（头部中心点）

逐帧显示 self_captured/ 下的图片，用鼠标左键点击头部中心，
自动生成归一化坐标标签，存入 self_captured/head_labels.csv。

按键：
  鼠标左键   标注头部中心（自动跳到下一张）
  n          本帧没有头（记为负样本，cx=cy=-1）
  u          撤销上一张，返回重标
  q          保存并退出（已标注的不丢失，可下次续标）

标签格式：filename,cx,cy  （cx,cy 为相对原帧的归一化坐标）
"""
import cv2
import csv
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), "self_captured")
LABEL_CSV = os.path.join(SAVE_DIR, "head_labels.csv")
WINDOW = "Annotate: click head center"


class Annotator:
    """逐帧交互标注器：显示图片、接收点击、维护标签列表。"""

    def __init__(self):
        """列出待标图片，加载已有标签（支持断点续标）。"""
        self.files = sorted(f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg"))
        self.labels = {}          # filename -> (cx, cy)，(-1,-1) 表示无头
        if os.path.exists(LABEL_CSV):
            with open(LABEL_CSV, newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if row:
                        self.labels[row[0]] = (float(row[1]), float(row[2]))
        # 从第一个未标注的图片开始
        self.idx = 0
        while (self.idx < len(self.files)
               and self.files[self.idx] in self.labels):
            self.idx += 1
        self.cur_point = None     # 当前帧鼠标点击结果

    def _on_mouse(self, event, x, y, flags, param):
        """鼠标回调：左键点击记录当前帧头部中心像素坐标。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cur_point = (x, y)

    def _save(self):
        """把全部标签写入 CSV（标注中途退出也不丢数据）。"""
        with open(LABEL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for name, (cx, cy) in sorted(self.labels.items()):
                w.writerow([name, f"{cx:.6f}", f"{cy:.6f}"])

    def run(self):
        """主循环：显示当前帧并处理点击与按键，直到全部标完或退出。"""
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

        while self.idx < len(self.files):
            name = self.files[self.idx]
            img = cv2.imread(os.path.join(SAVE_DIR, name))
            if img is None:
                self.idx += 1
                continue
            h, w = img.shape[:2]
            self.cur_point = None

            while True:
                vis = img.copy()
                # 已有标签的画出来（撤销重标时能看到旧点）
                if name in self.labels:
                    cx, cy = self.labels[name]
                    if cx >= 0:
                        cv2.drawMarker(vis, (int(cx * w), int(cy * h)),
                                       (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(vis, f"{self.idx + 1}/{len(self.files)}  {name}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(vis, "click=head  n=no head  u=undo  q=quit",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(50) & 0xFF

                if self.cur_point is not None:
                    x, y = self.cur_point
                    self.labels[name] = (x / w, y / h)
                    break
                if key == ord("n"):
                    self.labels[name] = (-1.0, -1.0)
                    break
                if key == ord("u"):
                    self.idx = max(0, self.idx - 1)
                    self.labels.pop(self.files[self.idx], None)
                    break
                if key == ord("q"):
                    self._save()
                    cv2.destroyAllWindows()
                    print(f"已保存 {len(self.labels)} 条标签到 {LABEL_CSV}")
                    return

            if key == ord("u"):
                continue          # 重标上一张
            self.idx += 1

        self._save()
        cv2.destroyAllWindows()
        n_head = sum(1 for v in self.labels.values() if v[0] >= 0)
        print(f"标注完成：{len(self.labels)} 帧（{n_head} 有头 / "
              f"{len(self.labels) - n_head} 无头），已存 {LABEL_CSV}")


if __name__ == "__main__":
    Annotator().run()
