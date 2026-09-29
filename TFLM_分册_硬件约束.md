# 分册 · 硬件约束（L0 层）

> 对应总纲的 L0。核心命题：**你必须先能量化目标平台的约束，才能设计模型。**
> 本文分三部分：通用方法 → ESP32 具体约束 → 外设与内存扩容。

---

## 第一部分 · 通用方法：怎么量化一块板子的"AI 预算"

拿到任何平台，先回答这四个数字。答不出来就没法设计模型。

| 数字 | 怎么查 | 为什么重要 |
|---|---|---|
| **可用 RAM** | 数据手册总 SRAM − 框架/RTOS 开销 − 栈 | 决定 arena 上限，即模型上限 |
| **可用 Flash** | 分区表里 app 分区大小 | 决定模型权重上限（权重放 Flash） |
| **算力** | 主频 × 每周期 MAC 数 | 决定时延可行性 |
| **功耗** | 数据手册 + 实测 | 决定电池寿命与是否需要事件驱动 |

### ⚠️ 关键认知：数据手册的 RAM ≠ 你能用的 RAM
以 `esp32dev` 为例（实测读自 PlatformIO 板定义）：
```
upload.maximum_ram_size = 327680   # 320 KB
```
但片上 SRAM 标称 **520 KB**。差的 200 KB 去哪了？
- Arduino 框架自身
- WiFi / BT 协议栈缓冲（开 WiFi 时吃得很凶）
- FreeRTOS 任务栈
- 堆碎片预留

**真正留给 TFLM arena 的通常是几十 KB 量级**，不是 300 KB。所以：
> **不要用数据手册数字做设计。必须实测。**

### 实测方法（三条，按可靠性排序）
1. **运行时打印**（最准）
   ```cpp
   printf("arena used = %u\n", interpreter.arena_used_bytes());
   printf("free heap  = %u\n", esp_get_free_heap_size());
   ```
2. **二分试探 arena**：从 4KB 起，每次翻倍，看 `AllocateTensors()` 何时通过
3. **编译期看链接报告**：`pio run -t size` 或 IDF 的 `idf.py size-components`

---

## 第二部分 · ESP32 具体约束

### 一、命名辨析（最易出错的地方）

`ESP32-WROOM` 是**模组名，不是芯片名**。子型号芯片都是**经典 ESP32**：

| 型号 | 芯片 | 内核 |
|---|---|---|
| ESP32-WROOM-32 / 32D / 32E / 32U / 32UE | ESP32 | Xtensa LX6 双核 |

**易混三兄弟：**

| 名称 | 芯片 | 说明 |
|---|---|---|
| ESP32-WROOM-32 | 经典 ESP32 | ← 你当前的板子 |
| ESP32-WROVER | 经典 ESP32 + **PSRAM** | 内存大得多，可跑视觉 |
| ESP32-S3-WROOM-1 | **ESP32-S3** | 不同的芯片，RISC-V 时代前的 LX7 |

⚠️ PlatformIO 板定义里有 `freenove_esp32_s3_wroom.json` —— 名字带 wroom，但它是 **ESP32-S3**，别选错。
你的板子对应 `board = esp32dev`。

### 二、内存现实

| 项 | 数值 | 来源 |
|---|---|---|
| 片上 SRAM | 520 KB | 经典 ESP32 规格 |
| PlatformIO `esp32dev` 声明可用 RAM | 327,680 B（320 KB） | `esp32dev.json` 的 `upload.maximum_ram_size` |
| PSRAM | **默认没有** | WROOM 无 PSRAM；要 PSRAM 得选 WROVER 或 S3 N8R8 |
| Flash | 4 MB | `esp32dev.json` 的 `upload.flash_size` |

**能做什么：**
- ✅ 小模型定点推理（Dense / Conv1D / 小 Conv2D）
- ⚠️ 稍大的 CNN（需量化 + arena 调优才挤得下）
- ❌ 摄像头视觉类（模型 + 输入张量 + arena 叠加，基本没戏）

> 关于具体 arena 数值：不同 TFLM 版本、不同模型差别很大，**不要记数字，要用 `arena_used_bytes()` 实测**。

### 三、外设现实（决定了你能做哪些任务）

ESP32-WROOM 的常见开发板（如 ESP32-DevKitC）**没有麦克风、没有摄像头、没有屏幕**。

| 官方例程 | WROOM 可行性 | 说明 |
|---|---|---|
| `hello_world` | ✅ | 最小推理单元，无外设，**第一个目标** |
| `micro_speech` | ⚠️ 需外接 I2S 麦克风 | 要自己接 INMP441 / ICS-43434 |
| `person_detection` | ❌ 不建议 | 需摄像头 + LCD，且内存大概率不够 |

**别照搬 Nano 33 BLE 的教程** —— 那些板子有板载麦克风，WROOM 没有。

### 四、经典 ESP32 专属注意点

| 事项 | 说明 |
|---|---|
| 电平 | GPIO 是 **3.3V**，接外设别给 5V 逻辑 |
| 启动相关引脚 | GPIO 0 / 2 / 12 / 15 影响启动模式，外设别乱占 |
| 输入专用引脚 | GPIO 34–39 **只能输入**，不能做输出 |
| ADC2 与 WiFi 冲突 | 用 WiFi 时 ADC2 通道不可用；读模拟量用 **ADC1**（GPIO 32–39） |
| 内存被 WiFi 吃掉 | 开 WiFi 显著减少可用 heap，**推理实验先关 WiFi** |
| 双核 | 可把推理放独立 task 固定到另一核，初学阶段不必 |
| 无原生 USB | 靠 USB-UART 桥（CP2102 / CH340），需装驱动；**好处是串口监视器比 S3 原生 USB 稳** |

---

## 第三部分 · 要换 S3 的话（内存扩容的正解）

### 一、S3 相比 WROOM 多出什么

| 能力 | ESP32-WROOM | ESP32-S3 | 对端侧 AI 的意义 |
|---|---|---|---|
| 片上 SRAM | 520 KB | 512 KB | 基本持平（**别指望这个**） |
| PSRAM | **默认无** | **可选**（N8R8 = 8MB） | ⭐ 真正的升级 |
| 内核 | Xtensa LX6 双核 | Xtensa LX7 双核 | 单核性能更高 |
| 向量指令 | 无 | **有**（PIE 128-bit SIMD） | ⭐ int8 卷积加速 |
| 摄像头接口 | 无专用 | **LCD_CAM 外设** | ⭐ 视觉方向的前提 |
| 原生 USB | 无 | 有 | 调试方便，但有坑 |

### ⚠️ 两个关键判断
1. **不要为了跑当前的小 Dense 模型换 S3** —— 在 WROOM 上毫无压力，S3 收益用不上。
2. **要换就买带 PSRAM 的型号**（S3-WROOM-1 **N8R8**）。没有 PSRAM 的 S3 对视觉任务**依然不够**，等于白换。

### S3 专属注意点

| 事项 | 说明 |
|---|---|
| GPIO 19 / 20 | 默认给原生 USB（D-/D+），想当普通 IO 要改熔丝 |
| GPIO 26–32 | 接 SPI Flash / PSRAM，**不能当普通 IO**，否则无法启动 |
| GPIO 33–37 | Octal PSRAM 型号上被占用 |
| **两个 USB 口** | 常有 UART 口 + 原生 USB 口，**接错口会导致 monitor 无输出** |
| monitor 无输出 | 加 `monitor_rts = 0` / `monitor_dtr = 0`，或改用 UART 口 |
| 架构 | RISC-V（`riscv32-esp-elf`），不是 xtensa |

### MicroPython 用户必看的坑（C++ 路线不走）
- MicroPython 官方为 S3 提供的通用固件**带 SPIRAM 支持（OCT）**，但**只对八线 PSRAM 模组（如 N8R8）有效**
- 模组**无 PSRAM 或只有四线 QSPI PSRAM** 时，刷通用固件会**启动失败或反复重启**
- 正确做法：用 `esptool` 读芯片信息确认 PSRAM 型号，再挑对应固件

---

## 第四部分 · 内存不够时的处置顺序

按性价比排序，**从下往上试**：

| 顺序 | 手段 | 收益 | 代价 |
|---|---|---|---|
| 1 | **量化到 int8** | 权重 ↓4×，激活 ↓4× | 轻微精度损失 |
| 2 | 缩小输入尺寸 | 激活内存 ↓平方级 | 精度可能明显下降 |
| 3 | 减少通道数 / 层数 | 线性下降 | 欠拟合风险 |
| 4 | 关掉 WiFi / BT / 日志 | 释放几十 KB heap | 功能受限 |
| 5 | 调 `menuconfig` 减任务栈、降日志等级 | 几 KB ~ 几十 KB | 调试变难 |
| 6 | 用 PSRAM 存权重 | 释放内部 SRAM | 访问慢；**需硬件支持** |
| 7 | 换板子（S3 N8R8 / WROVER） | 质变 | 花钱 |

**优先做 1–4。** 换板子是最后手段 —— 而且往往不是真需要。

---

## 第五部分 · 验收清单

- [ ] 能说出板子的**实测**可用 RAM（不是数据手册数字）
- [ ] 能用 `arena_used_bytes()` 读出真实 arena 占用
- [ ] 知道自己的板子**有没有 PSRAM**
- [ ] 知道哪些 GPIO 不能乱用
- [ ] 知道推理实验要关 WiFi
- [ ] `LongPathsEnabled = 1`
- [ ] 若走视觉/音频：已确认要买带 PSRAM 的 S3
