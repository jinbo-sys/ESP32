# 分册 · 迁移与选型

> 对应总纲的 L6（多平台）与阶段 E。这份回答两个问题：
> **1. 从 ESP32-WROOM 怎么迁到 ESP32-S3？ 2. 什么任务该选什么平台？**

---

# 第一部分 · WROOM → S3 迁移

## 一、结论先行：迁移成本比想象中小

| 层面 | 是否要改 |
|---|---|
| 你的 C++ 推理代码（`main.cpp`） | **完全不用改** |
| TFLM API（arena / resolver / interpreter / Invoke） | **完全不用改** |
| `models/` 模型数据 | 不用改（换大模型才需重转） |
| `platformio.ini` | **改 1 行**（board），另加 2 行可选 |
| IDF 路线 `set-target` | 改 1 处：`esp32` → `esp32s3` |
| 工具链 | **自动下载**，不用手工装 |

**S3 迁移基本只动配置，不动代码。** 这也是先用 WROOM 起步不吃亏的原因。

## 二、两套路径各自改什么

### 路径 A（Arduino + PlatformIO）
```ini
; 改这一行
board = esp32-s3-devkitc-1        ; 原: esp32dev

; S3 原生 USB-CDC 与 PIO 串口监视器偶有兼容问题，建议加：
monitor_rts = 0
monitor_dtr = 0
monitor_speed = 115200
upload_speed = 921600             ; 不稳就删

; ★ 若买的是带 PSRAM 的型号（N8R8 等），还需开 PSRAM：
board_build.arduino.memory_type = qio_opi
build_flags =
    -DBOARD_HAS_PSRAM
```
首次编译 PIO 会自动下载 RISC-V 工具链（`toolchain-riscv32-esp`）与 S3 的 Arduino 内核包，需联网等几分钟。

### 路径 B（ESP-IDF 原生）
```powershell
C:\esp\esp-idf\export.ps1
idf.py set-target esp32s3          # 原: esp32
idf.py build
idf.py -p COM<N> flash monitor
```
IDF 安装器一次装好 xtensa + riscv 工具链，**S3 不需额外装任何东西**。

## 三、S3 相比 WROOM 实际多出什么

| 能力 | ESP32-WROOM | ESP32-S3 | 对端侧 AI 的意义 |
|---|---|---|---|
| 片上 SRAM | 520 KB | 512 KB | 基本持平 |
| PSRAM | **默认无** | **可选**（N8R8 = 8MB） | ⭐ 真正的升级 |
| 内核 | Xtensa LX6 双核 | Xtensa LX7 双核 | 单核性能更高 |
| 向量指令 | 无 | **有**（PIE 128-bit SIMD） | ⭐ int8 卷积加速 |
| 摄像头接口 | 无专用 | **LCD_CAM 外设** | ⭐ 视觉方向的前提 |
| 原生 USB | 无（需桥接芯片） | 有（USB-OTG + JTAG） | 调试方便，但有坑 |

### ⚠️ 两个关键判断
1. **不要为了跑"当前这个 Dense 小模型"换 S3**。16×16 两层网络在 WROOM 上毫无压力。S3 的收益只在**视觉 / 音频 / SIMD 加速的卷积**上体现。
2. **要换就买带 PSRAM 的型号**（S3-WROOM-1 **N8R8**）。没有 PSRAM 的话，S3 对视觉任务**依然不够**，等于白换。

## 四、S3 专属注意点

| 事项 | 说明 |
|---|---|
| GPIO 19 / 20 | 默认给原生 USB（D-/D+），想当普通 IO 要改熔丝，**别乱用** |
| GPIO 26–32 | 接 SPI Flash / PSRAM，**绝对不能当普通 IO**，否则无法启动 |
| GPIO 33–37 | 在 Octal PSRAM 型号上被占用 |
| **两个 USB 口** | S3 开发板常有 UART 口 + 原生 USB 口，**接错口会导致 monitor 无输出**（最常见困惑） |
| monitor 无输出 | 用原生 USB-CDC 时加 `monitor_rts = 0` / `monitor_dtr = 0`，或改用 UART 口 |
| 架构 | S3 是 RISC-V（`riscv32-esp-elf`），**不是** xtensa，报错里的架构名会变 |

## 五、MicroPython 用户必看的一个坑（C++ 路线不走此坑）

若你在 S3 上先跑 MicroPython 验证硬件：
- MicroPython 官方**为 ESP32-S3 提供的通用固件带 SPIRAM 支持（OCT 模式）**，但它**只对八线 PSRAM 模组（如 N8R8）有效**
- 若模组**无 PSRAM 或只有四线 QSPI PSRAM**，刷通用固件会**启动失败或反复重启**
- 正确做法：用 `esptool` 读芯片信息确认 PSRAM 型号，再挑对应固件

---

# 第二部分 · 平台选型决策树

## 六、先回答三个问题

```
Q1. 模型多大？
    ├─ < 500 KB  → MCU 路线（TFLM / CMSIS-NN / 厂商 SDK）
    └─ > 500 KB  → 嵌入式 Linux（NCNN / TensorRT / ONNX Runtime）

Q2. 实时性要求？
    ├─ 硬实时（< 10 ms 且必须确定性） → MCU（无 OS 抖动）
    └─ 软实时（几十 ms 可接受）      → Linux 也行

Q3. 功耗预算？
    ├─ 电池供电 / µA 级待机 → MCU + 事件驱动（传感器唤醒）
    └─ 常电 / 瓦级          → Linux 板子
```

## 七、平台谱系对照

| 平台 | 类型 | 适用 | 特点 |
|---|---|---|---|
| **TFLM** | MCU 推理框架 | Cortex-M / Xtensa / RISC-V | 开源主流，最灵活，要自己写胶水 |
| **Edge Impulse** | 端到端平台 | MCU + 传感器 | 采集→标注→训练→部署一条龙，快但依赖云 |
| **CMSIS-NN** | ARM 算子库 | Cortex-M | TFLM 在 ARM 上的加速后端 |
| **ESP-NN / esp-tflite-micro** | Espressif 封装 | ESP32 系列 | 自带 SIMD 优化，你这条路 |
| **STM32Cube.AI (X-CUBE-AI)** | 厂商工具 | STM32 | Keras/ONNX 直接生成 C，商业支持好 |
| **NCNN / MNN / TNN** | 移动端推理 | 手机 / 嵌入式 Linux | 面向 ARM CPU + GPU |
| **TensorRT / OpenVINO** | 厂商引擎 | Jetson / Intel | 桌面级边缘 |
| **ONNX Runtime** | 通用运行时 | 跨平台 | 模型交换格式事实标准 |
| **ExecuTorch** | PyTorch 官方端侧 | 新兴，覆盖 MCU 到手机 | 值得关注 |

### 选型建议（针对你的情况）
| 任务 | 推荐 |
|---|---|
| 传感器时序分类（手势/异常检测） | ESP32 + TFLM，你的 WROOM 就够 |
| 关键词识别 | ESP32-S3 + TFLM + I2S 麦克风 |
| 图像分类 / 人物检测 | ESP32-S3 **N8R8**（带 PSRAM）+ TFLM，或直接上 Linux 板 |
| 快速做原型给非技术方看 | Edge Impulse，几小时出结果 |
| 产品级 ARM 方案 | STM32 + Cube.AI，有商业支持 |
| 多路视频 / 大模型 | Jetson Orin Nano / 树莓派 + Coral |

## 八、迁移检查清单

- [ ] 确认 S3 型号**是否带 PSRAM**（走视觉/音频就必须带）
- [ ] 路径 A：`board` 改 `esp32-s3-devkitc-1`
- [ ] 路径 A：等待 RISC-V 工具链下载完成
- [ ] 路径 A：带 PSRAM 型号加 `board_build.arduino.memory_type` + `-DBOARD_HAS_PSRAM`
- [ ] 路径 A：备好 `monitor_rts=0` / `monitor_dtr=0`
- [ ] 路径 B：`idf.py set-target esp32s3`
- [ ] 确认 USB 线接对了口（UART 口 vs 原生 USB 口）
- [ ] 重新测 `arena_used_bytes()`（S3 上数值会与 WROOM 略有差异）
- [ ] 确认 WiFi 在推理实验中关闭（省 heap）
