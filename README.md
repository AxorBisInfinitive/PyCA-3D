# PyCA-3D

基于 Numpy 和 SDL2 制作的软渲染器，用于浏览 Cellular Automata。

代码基于：

- [**AxorBisInfinitive/CellularAutomata**](https://github.com/AxorBisInfinitive/CellularAutomata)
- [**AxorBisInfinitive/PySDL2_Basic**](https://github.com/AxorBisInfinitive/PySDL2_Basic)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

---

## 安装

在目录中运行命令：

```bash
pip install -r requirements.txt
```

若环境中安装了 **CUDA**，也可继续运行命令：

```bash
pip install cupy
```

随后在 `loader.py` 中将 `calculate_module` 的值改为 `"cupy"` 以在 **CA** 模拟时启用 **GPU 硬件加速**。

---

## 运行

在目录中运行命令：

```bash
python src/main.py
```

---

## 使用

- 滚轮：调整步幅
- WASD：移动位置
- RF：调整时间步
- Q：复位
- F2：截图
- ESC：关闭程序

修改 `loader.py` 中的字面量，可控制模拟效果。

---

## 注意

- 请确保输入法为**英文模式**，否则键盘事件可能无效。
- 程序初始化时会进行必要的 **JIT** 编译，会出现短暂冻结。
- 若程序冻结或严重卡顿，请在终端 Ctrl+C 终止程序。
- **LUT** 内存占用：`~128MB`。
- 程序会创建以下**文件**/**目录**，用于存储数据：
  - cell
  - rule
  - screenshots
  - latest.log

---

## 特性

- 程序的 **CA** 规则是 **Wolfram code** 的推广：
  - 邻域：$3^3 = 27$。
  - `rule` 取值：$0 \leq n \leq 2^{2^{27}} - 1$。
  - 修改 `loader.py` 中的 `rule` 函数，以自定义 `rule` 的计算方式。
- 在 **CPU** 上完成渲染。基于软件的渲染管线：
  - 顶点变换
  - 裁剪
  - 透视除法
  - 视口变换
  - 光栅化
  - 深度测试

- 使用 **Numba** 以进行 **JIT** 即时编译。
- 使用 **Pillow** 实现截图功能。
- 包含文件 `SDL2.dll`，版本 **2.32.10**。

---

## 其他

- 严格来说，本项目是 **ECA** 的高维推广，而非 **CA**。**ECA** 是 **CA** 的一个非常小的子集。
- **Wolfram code** 不能使用 **BS**。但 **BS** 是 **Wolfram code** 的一个非常小的子集，因此存在某个方法可以将 **BS** 转为 **Wolfram code**。此处不再赘述。
