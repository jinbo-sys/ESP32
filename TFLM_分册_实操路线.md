# 分册 · 实操路线（平台、工具链、代码、报错）

> 对应总纲的阶段 A–C。这份是"手怎么动"的部分。
> 目标平台：ESP32 系列（WROOM 起步，后续 S3）。IDE：VSCode + PlatformIO。

---

## 一、先想清楚 TFLM 从哪来（三条路径）

TFLM 在 ESP32 上有一条主流路径，但 PlatformIO 官方平台不在那条路上。这是你必须先做的选择。

| | A. Arduino 框架 + PIO | B. ESP-IDF 原生 + VSCode | C. IDF 框架 + PIO |
|---|---|---|---|
| 工具链 | 已有 | 需装 IDF 安装器 | 让 PIO 装 `framework-espidf` |
| TFLM 来源 | 第三方库（`Arduino_TensorFlowLite_ESP32` / `Chirale_TensorFlowLite`） | 官方 `espressif/esp-tflite-micro` **v1.4.1** | 同左，手工搬进 `components/` |
| TFLM 新鲜度 | ⚠️ 常滞后 | ✅ 最新 | ✅ 最新 |
| 官方三个例程 | ❌ 要自己移植 | ✅ 一条命令拉取 | ⚠️ 要手工移植 |
| 中文资料 | ✅ 多 | ✅ 多 | ❌ 少 |
| 推荐度 | **阶段 A 用这个**（快） | **阶段 B 之后用这个**（正） |

### 为什么 C 会麻烦
PlatformIO 的 ESP-IDF 构建走 **SCons + 自己的 `espidf.py`**，而官方例程依赖 **CMake + IDF 组件管理器**。
- ✅ 能用：手工把 `esp-tflite-micro` 及其依赖 `esp-nn` clone 到项目 `components/`
- ❌ 不能用：`idf.py create-project-from-example`、`idf.py add-dependency`
- ⚠️ 版本卡点：`esp-tflite-micro` v1.4.x 需要 **ESP-IDF v5.x**

### 建议
**A 起步建立反馈，B 做正式学习。** 两者不冲突，可先后走。

---

## 二、环境搭建

### 2.1 系统级（管理员 PowerShell，之后重开终端）
```powershell
# ① 开长路径 —— 必做
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1

# ② 工具
winget install --id Git.Git -e
winget install --id Kitware.CMake -e
winget install --id Ninja-build.Ninja -e
```
> 你机器实测 `LongPathsEnabled = 0`，TFLM 源码路径极深，不开会在编译中途报 `cannot open file`，且报错位置有误导性。

### 2.2 路径 A（已具备条件）
PlatformIO IDE 3.3.4 + `espressif32` 7.0.1 + `toolchain-xtensa-esp32` 均已就绪，直接可用。

### 2.3 路径 B（正式学习时再装）
```powershell
# 装 ESP-IDF Windows 安装器，路径填浅路径 C:\esp\esp-idf
# 下载: https://dl.espressif.com/dl/esp-idf/
# 会一并装好 xtensa/riscv 工具链、Python、ninja

C:\esp\esp-idf\export.ps1     # 每次开工前激活
idf.py --version
```
VSCode 再装官方扩展 **`Espressif IDF`**，与 PlatformIO 可共存（但同一个工程别用两套系统构建）。

### 2.4 训练侧
```powershell
pip install tensorflow
```
另装 **Netron**（可视化 `.tflite`）：
```powershell
winget install --id Netron.Netron -e    # 或直接下 portable 版
```

### 2.5 路径 B 拉取官方例程
```powershell
cd F:\00_MK\TensorFlow
idf.py create-project-from-example "espressif/esp-tflite-micro:hello_world"
cd hello_world
idf.py set-target esp32          # S3 用 esp32s3
idf.py build
idf.py -p COM<N> flash monitor   # 退出: Ctrl + ]
```
换例子：把 `hello_world` 换成 `micro_speech` / `person_detection`。

---

## 三、TFLM API 骨架（背下来）

这是 TFLM 的**全部**调用链，一共 6 步：

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

constexpr size_t kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];   // 对齐不能省

void run_inference() {
  /* 1. 取 Model */
  const tflite::Model* model = tflite::GetModel(g_model_data);

  /* 2. 校验 schema 版本 */
  if (model->version() != TFLITE_SCHEMA_VERSION) { /* 重转模型 */ }

  /* 3. 注册算子：模板参数 = 最多注册几个，超了编译期报错（故意设计，省动态内存） */
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddSoftmax();

  /* 4. 建解释器 + 分配张量 */
  static tflite::MicroInterpreter interpreter(
      model, resolver, g_tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) { /* arena 不够 */ }

  /* 5. 写输入（int8 模型必须先量化） */
  TfLiteTensor* in = interpreter.input(0);
  in->data.int8[0] = quantize(real_value, in->params.scale, in->params.zero_point);

  /* 6. 推理 + 取输出 */
  interpreter.Invoke();
  TfLiteTensor* out = interpreter.output(0);
  float score = dequantize(out->data.int8[0], out->params.scale, out->params.zero_point);

  // 调优必备：打印真实内存占用
  // printf("arena used = %d\n", interpreter.arena_used_bytes());
}
```

**注意**：`static` 关键字不是可有可无。`MicroInterpreter` 和 `MicroMutableOpResolver` 对象较大，放栈上在内存受限的 MCU 上会栈溢出。

---

## 四、量化：端侧 AI 的核心技术

### 4.1 量化数学（必须能手算）
仿射量化把浮点映射到整数：
```
real = (q - zero_point) * scale
q    = round(real / scale) + zero_point
```
- `scale`：浮点步长（每个张量不同）
- `zero_point`：浮点 0 对应的整数值

**关键**：`scale` / `zero_point` 是**存在模型里的**，运行时从张量读，**绝不硬编码**。

### 4.2 训练后量化（PTQ）
```python
import tensorflow as tf

def representative_dataset():
    """⭐ 这是 PTQ 成败的关键。
    必须用真实训练数据（几百条足够），且要覆盖各种工况。
    用随机噪声会导致精度崩塌。"""
    for sample in x_train[:200]:
        yield [sample[None].astype('float32')]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8     # 全整型，MCU 上必须
converter.inference_output_type = tf.int8
open('model_int8.tflite', 'wb').write(converter.convert())
```

### 4.3 量化掉点严重时怎么办
| 症状 | 处理 |
|---|---|
| 掉点 5%+ | representative dataset 换成真实数据、增加条数、覆盖极端工况 |
| 某一层掉点集中 | 该层用 QAT，或保持 float 混合量化 |
| 输出饱和 | 检查是否有异常值没做裁剪（clipping） |
| 建议起步 | 先用 float32 跑通，再上 int8 对比精度 |

### 4.4 模型转 C 数组
```powershell
xxd -i model_int8.tflite > model.cc
```
⚠️ 生成后手动补 `alignas(16)`，并把数组名改成不含路径的合法标识符。

---

## 五、兼容性红线（训练时就要遵守，否则部署返工）

| 红线 | 说明 |
|---|---|
| 只用 TFLM 支持的算子 | 以 `tensorflow/lite/micro/kernels/` 目录实际存在的为准 |
| 必须静态 shape | 不要动态 reshape |
| 不用动态层 | 如默认 `LSTM` 实现 |
| 输入尺寸固定 | 输出层别加多余 `Reshape` |
| 训练时就用目标框架 | 别用一堆 TFLM 不支持的算子训完再想办法 |

**排查工具**：Netron 打开 `.tflite`，逐个算子对照 TFLM 内核目录。

---

## 六、报错速查

| 报错 | 原因 | 解决 |
|---|---|---|
| `Didn't find op for builtin opcode 'X'` | 算子没注册 | `resolver.AddX()` + 调大模板参数 |
| `AllocateTensors() failed` | arena 太小 | 加大 arena，用 `arena_used_bytes()` 反推 |
| `Model provided has schema version N other than supported M` | TF 版本不匹配 | 换匹配的 TF 重转，或升级 TFLM |
| `cannot open file` / 路径过长 | 长路径未开 | 开 `LongPathsEnabled`，工具装浅路径 |
| 链接段溢出 / 编译 OOM | MCU RAM 不足 | 开 PSRAM、减层数、降分辨率 |
| 结果与 PC 端差很多 | 量化误差 / scale 配错 | PC 端也用同一 int8 模型对齐；查 zero_point |
| `idf.py: command not found` | 忘了激活环境 | 先跑 `export.ps1` |
| `MissingSectionHeaderError` (PlatformIO) | `platformio.ini` 带 UTF-8 BOM | 见第七节 |

---

## 七、PlatformIO 实测坑：ini 文件不能带 BOM

PlatformIO 用 Python `configparser` 读 `.ini`，**文件开头的 UTF-8 BOM 会让它直接失败**：
```
MissingSectionHeaderError: File contains no section headers.
file: 'platformio.ini', line: 1
'\ufeff; ===...'
```
Windows PowerShell 5.1 的 `Set-Content -Encoding utf8` **会写 BOM**；VSCode 存盘也别选 "UTF-8 with BOM"。

修复：
```powershell
$p = 'platformio.ini'
$text = Get-Content $p -Raw -Encoding utf8
[System.IO.File]::WriteAllText($p, $text, (New-Object System.Text.UTF8Encoding($false)))
```

---

## 八、配套工程

`tflm_wroom/` 是一个最小可编译工程（路径 A）：
- `src/main.cpp` — 上面那 6 步的完整实现，换模型不用改
- `tools/make_model.py` — 训练 + 导出 + 转 C 数组一条龙
- `README.md` — 使用步骤与坑

详见 `tflm_wroom/README.md`。

---

## 九、延伸资源

- TFLM 官方仓库：https://github.com/tensorflow/tflite-micro
- 内存管理文档（**阶段 A 就该读**）：https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md
- Espressif TFLM 组件：https://github.com/espressif/esp-tflite-micro
- 组件注册表（例程列表）：https://components.espressif.com/components/espressif/esp-tflite-micro
- ESP-IDF 入门：https://docs.espressif.com/projects/esp-idf/en/v5.5.3/esp32s3/get-started/index.html
- LiteRT 迁移说明（理解 TF Lite → LiteRT 命名变化）：https://developers.google.com/edge/litert/migration
- 论文：*TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems* (MLSys 2021)
- 课程：edX *Introduction to Embedded Machine Learning*；Harvard TinyMLx
- 书：《TinyML》(Pete Warden & Daniel Situnayake) 第 3–5 章
- 书：《AI at the Edge》(Daniel Situnayake & Jenny Plunkett) —— 偏工程实践
