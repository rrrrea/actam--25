把 model inference 放在后端 API 或 ONNX/WebAssembly，前端只做播放与可视化。

**浏览器端调用音频生成模型**：前端（Web Audio）+ 后端 inference API（如 MusicGen / TTS / voice conversion），project 同时成为你的 portfolio demo。

**Neural audio codec（神经音频编解码）可视化**：实时展示波形 → EnCodec/SoundStream token → 重建，把你阶段二的学习成果做成交互 demo。这直接服务"音频离散化"这一关键环节的理解。

**关键架构决策：v1 用"离线预计算 + 纯静态前端"**。==浏览器端跑 EnCodec/DAC 推理（ONNX/WASM）或搭后端 API== 都会吃掉十小时的一半以上

### **v1 定义**

**Neural Audio Codec Explorer**：<mark style="background:rgba(240, 200, 0, 0.2)">选择音频样本 → A/B 切换 original vs 不同 bitrate 的重建 → 并排 spectrogram 对比 → RVQ token 网格可视化（codebook × time heatmap）。</mark>

**直接复用考试素材**：field recordings 已有，==codec 处==理 pipeline 运行同时产出考试的 compositional material 和这个 demo 的 assets。  
  
###### **十小时分解**

| **时段**   | **任务**                                                                                                                                                                                | **产出**                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **H1**   | Repo scaffold；选 2–3 段 field recording（各 ≤15s，控制文件体积）                                                                                                                                  | 项目骨架                       |
| **H2–3** | Python 离线 pipeline：EnCodec（或 DAC）在 1.5/3/6/12/24 kbps 编解码；<mark style="background:rgba(136, 49, 204, 0.2)">导出重建 .wav、token 序列 JSON（codes tensor 直接 dump）、mel spectrogram PNG（librosa） | as</mark>sets/ 全部数据        |
| **H4–5** | 前端核心：样本选择器 + **Web Audio A/B** **无缝切换**（两个 buffer 同步播放，切 gain 而非重启，保持时间对齐）+ bitrate 选择控件                                                                                              | 可听                         |
| **H6–7** | 可视化：spectrogram 并排/滑块对比；token grid（canvas 画 heatmap，横轴 time frame、纵轴 codebook level，颜色 = token id）                                                                                    | 可看                         |
| **H8**   | 交互润色：播放头与 token grid 联动高亮（playhead 同步是最廉价的"wow"点）                                                                                                                                     | 交互亮点                       |
| **H9**   | README：照 handwarper 骨架——concept → why neural codecs → **RVQ** **原理解释一节**（复用你阶段二笔记）→ 运行指南                                                                                              | 文档 = portfolio write-up 初稿 |
| **H10**  | GitHub Pages 部署 + buffer                                                                                                                                                              | 可访问链接                      |

###### **砍单原则（时间超支时按序放弃）**

1. 先砍 H8 联动高亮 → 2. token grid 降级为静态图片 → 3. spectrogram 对比降级为并排 <img>。**A/B 听感切换 + bitrate 选择是不可砍的核心**，它承载"compression artifacts as material"的全部论证。

###### **v2 路线（不在十小时内）**

- 用户上传音频 + FastAPI 后端实时推理（或 ONNX Runtime Web 客户端推理）
- ==RVQ codebook stripping 开关（保留前 k 层 → 听 semantic/acoustic 分层退化==）——这是最有研究叙事价值的功能，v2 首选
- ==Mimi/WavTokenizer 多 codec== 横向对比

**一个前置检查**

H2 前先确认本地 CPU 推理可行：pip install encodec 后跑 10 秒样本，EnCodec 24kHz CPU 编解码约几秒到几十秒量级，可接受。若环境有问题，fallback 到 Colab 跑 H2–3，产物下载回本地。

  

==source .venv/bin/activate==

## EnCodec 是什么

**EnCodec** 是 Meta 2022 年发布的 **neural audio codec（神经音频编解码器）**。功能与 MP3 同类——把音频压缩成极小的数据再还原——但实现方式是三段神经网络：

```
波形 → [Encoder] → 连续潜在向量 → [RVQ 量化] → 离散 token → [Decoder] → 重建波形
       卷积网络       每75帧/秒          查表离散化                  卷积网络
```

核心机制是 **RVQ (Residual Vector Quantization，残差向量量化)**：

- 每一帧的连续向量，先用第 1 个 **codebook（码本，1024 个条目的查找表）**找最近的条目，记下编号（一个 0–1023 的整数，即 **token**）
- 量化必有误差（residual），第 2 个 codebook 专门量化这个残差，依此叠加
- **用的 codebook 层数越多 = <mark style="background:rgba(136, 49, 204, 0.2)">bitrate 越高 = 重建越保真**</mark>。1.5 kbps 用 2 层，24 kbps 用 32 层——这就是你刚跑出的 `n_q=2` 到 `n_q=32`

它的双重身份决定了它在你路线图里的位置：

| **压缩工具**                    | 音频 ↔ 极低码率数据，低码率时产生特征性失真                                                                                    |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Audio tokenizer（音频离散化器）** | 把连续波形变成**离散整数序列**——和文字 token 同构，从而能喂给 LLM。MusicGen、VALL-E 生成的"内容"就是 EnCodec token，最后由 EnCodec decoder 变回声音 |

第二个身份是它出现在你**阶段二（音频离散化）**的原因：它是"音频接入大模型"的咽喉。


###### **跑通"音频 → token → 重建音频"这条 pipeline，把中间产物同时做成三样东西。**

刚完成的 H2 脚本对每段 field recording 做的事：

1. 在 5 个 bitrate 档位各跑一遍 encode → decode
2. 落盘三类产物：**重建 wav**（听）、**token JSON**（`[n_q, n_frames]` 整数矩阵，看）、**mel spectrogram**（对比）

三类产物分别服务你的三个并行目标：

|产物|去向|服务|
|---|---|---|
|低 bitrate 重建 wav（1.5/3 kbps 的崩坏声）|导入 DAW|**电声作曲考试**：quantization artifact 即 compositional material|
|token JSON + 谱图 + A/B 音频|H4–H8 的网页前端|**ACTAM final project**：codec 可视化 demo|
|对 RVQ 行为的直觉（层数 ↔ 保真度 ↔ artifact）|你的知识体系|**阶段二核心内容**的动手验证，替代纯读论文|

**当前位置**：十小时计划的 H2 已完成（脚本已验证），你本地跑一遍真实素材后，进入 H4——写网页播放这些文件。剩下的工作不再涉及模型，纯前端。

###### **codes 张量形状**：
已确认是 `[n_q, n_frames]`，`n_q` 随 bitrate 从 2 到 32 变化——**这正是 token grid "越高 bitrate 层数越多" 的可视化叙事**，H6 直接读 JSON 画图。

###### 两个导出物分别是什么

**pipeline 中间和末端的两次"截取"**：

```
你的录音.wav → Encoder → RVQ → [codes tensor] → Decoder → [重建.wav]
                                     ↑截取①                    ↑截取②
```

###### ① codes tensor → token JSON

**codes tensor** 是 RVQ 量化后的输出：一个整数矩阵，形状 `[n_q, n_frames]`。

以 1.5 kbps 结果为例：`2 × 375`——

- **375（n_frames）**：==EnCodec 每秒产 75 帧==，5 秒音频 = 375 帧。==每帧对应约 13ms 的==声音
- **2（n_q）**：该 bitrate ==用 2 层 codebook==
- **矩阵里每个数**：一个 0–1023 的整数，即"这一帧的这一层，在 codebook 里选中了第几号条目"

```
frame:      0     1     2    ...   374
layer 0: [ 517,  882,  517, ...,  103 ]   ← 粗轮廓
layer 1: [  74,  291,  606, ...,  912 ]   ← 第一层的残差修正
```

**这个矩阵就是"音频的 token 序列"**——你的录音在这一刻变成了和文字同构的离散符号。MusicGen/VALL-E 学习生成的就是这种矩阵。导出成 JSON 只是为了让浏览器 JavaScript 能读它、画成 heatmap（H6 的 token grid）。

**关键认知：这 750 个整数是这 5 秒音频在 1.5 kbps 下的全部信息**。原始 wav 有 12 万个采样点，压缩后只剩 750 个查表编号——丢掉的信息就是你听到的 artifact 的来源。

###### ② 重建 .wav

把 codes tensor 喂回 **Decoder**：每个编号查 codebook 换回向量，逐层相加，卷积网络还原成波形，存成普通 wav 文件。

它**不是原录音**，是"模型仅凭那 750 个 token 猜出来的声音"。bitrate 越低（token 越少），猜得越离谱——那个"离谱"就是你作曲要的材料、也是 demo 里 A/B 对比要展示的东西。

**在 demo 里的对应关系**

|导出物|前端用途|
|---|---|
|重建 .wav（×5 档）|`<audio>` 播放，与 original A/B 切换|
|token JSON|canvas 画 heatmap：横轴 frame（时间），纵轴 layer，颜色 = token 编号。用户**同时听到重建、看到它背后的全部离散表示**|

这个"听与看的对应"就是整个 project 的表达核心：**声音的退化 ↔ token 的稀疏，一体两面**。

###### 进入 H4–5：前端核心。

zip 内结构即你的 repo：

```
site/
├── index.html          # 页面骨架
├── css/style.css       # 视觉：冷灰纸面 + mono 仪表风格，单一 signal 红做焦点色
├── js/main.js          # 全部逻辑（下述）
├── scripts/encode.py   # H2 pipeline（已加 manifest.json 生成）
└── assets/             # 我用合成信号跑出的占位数据，页面开箱即可跑
```

###### main.js 实现的核心机制（对应 H4–H8）

**A/B 无缝切换（H4–5，不可砍的核心）**：play 时**所有变体（original + 5 档 bitrate）同时开播**，各接一个 `GainNode`，只有当前选中的 gain=1 其余为 0。切 bitrate 只是 `setTargetAtTime` 翻转 gain（8ms 平滑，无 click）——时间轴永远对齐，这是"听出量化差异"的前提。

**Token grid（H6–7）**：canvas 读 token JSON 画 heatmap，纵轴 codebook layer、横轴 frame、颜色 = token id（inferno 色带，呼应谱图视觉语言）。切 bitrate 时网格从 2 行变 32 行——**"bitrate = 信息量"直接可见**。

**Playhead 联动（H8）**：`requestAnimationFrame` 用 `AudioContext.currentTime` 驱动红色播放线扫过 token grid——听到的每一刻和正在"消费"的 token 列对齐。

**谱图对比**：original 与当前重建并排 `<img>`，切档即换。

###### 你本地的操作（替换占位数据）

```bash
cd site
python scripts/encode.py 你的录音1.wav 你的录音2.wav   # 覆盖 assets/
# VS Code Live Server 打开 index.html，或：
python -m http.server 8000   # 访问 localhost:8000
```

**必须经 HTTP 访问**（Live Server 或 http.server），直接双击 index.html 打开会因 `fetch()` 跨源限制加载不了 assets。

## 剩余预算对照

| 计划            | 状态                                                              |
| ------------- | --------------------------------------------------------------- |
| H1 scaffold   | ✅（zip 即 scaffold）                                               |
| H2–3 pipeline | ✅ 已验证                                                           |
| H4–8 前端       | ✅ 已交付，含原砍单候选的 playhead                                          |
| **H9 README** | 待做——照 handwarper 骨架，RVQ 原理节复用本对话第 6、7 轮的解释文本                    |
| **H10 部署**    | 待做——GitHub Pages（Settings → Pages → main branch → root），静态站直接可上 |

你的十小时预算现在主要剩：本地跑真实素材（~1h 含试听选段）、README（~1.5h）、部署（~0.5h），其余是余量。浏览器端行为（音频解码、gain 切换）我无法在容器里验证，你 Live Server 打开后若有报错，把 console 输出发来。

###### 发生了什么

`pip install torch` 装的不是"装到电脑上"，而是**装进某一个 Python 解释器的 `site-packages` 目录**。每个解释器有自己独立的一套包，互不可见。

有两个"Python 世界"：

**venv（virtual environment，虚拟环境）** 就是从系统 Python 复制出的一个轻量隔离副本：自带独立的 `python` 可执行文件和独立的包目录。当时 `python3 -m venv .venv` 创建了它，激活状态下的 `pip install` 全部落在 `.venv/` 文件夹里。

 `source .venv/bin/activate` 做了什么

只做一件事：**修改当前终端的 `PATH` 环境变量**，把 `.venv/bin/` 插到最前面。效果：

| 状态                  | 输入 `python` 时实际执行           | 能否 import torch  |
| ------------------- | --------------------------- | ---------------- |
| 未激活                 | `/opt/homebrew/bin/python3` | ❌ 它的包目录里没有       |
| 激活后（提示符带 `(.venv)`） | `ACTAM/.venv/bin/python`    | ✅ torch 就装在它的包目录 |


三个推论：

- ==**激活是 per-terminal、per-session 的**：只影响当前这个终端窗口，关掉或新开窗口就失效==——这就是你今天"突然找不到 torch"的原因
- 提示符前缀 `(.venv)` 是唯一可靠的状态指示
- http.server 那个终端不需要激活（它不 import torch）

 **为什么要这套机制**

包依赖按项目隔离：项目 A 要 torch 2.13、项目 B 要 torch 2.1 时不互相污染；删掉 `.venv` 文件夹 = 干净卸载全部依赖。Homebrew Python 甚至**强制**这样做（直接 `pip install` 会报 `externally-managed-environment`），防止第三方包弄坏系统组件。

###### 固定工作流 

```bash
cd ~/开发一半的本地作业/ACTAM
source .venv/bin/activate      # 看到 (.venv) 前缀
python scripts/encode.py sound_source_24khz/*.wav
```

概念对齐你的知识体系：venv 之于 Python 包，等价于 **conda env**（ML 圈更常用，原理相同），也是后续你跑任何模型 repo（CosyVoice、Moshi 等）时 README 第一步 `conda create` / `python -m venv` 的含义——**每个 repo 一个隔离环境**是该领域的标准操作。


作为交付物评估：达到 v0.8，缺的恰是承载论证的两件东西

**已达标**：

- 结构完整：record 选择 → bitrate 档位 → original/reconstructed 对照 → token 元数据（n_q=8, frames=1746 正确对应 6kbps）
- 谱图对比可用，视觉风格成立


**未达标（对照 v1 定义）**：

|缺失|后果|
|---|---|
|**无缝 A/B**：两个独立播放器，各自的进度条|用户无法在同一时刻对齐比较——必须手动对齐两条 timeline，量化差异的听感对比（整个 demo 的核心论证）名存实亡|
|**Token grid 可视化**：只显示了 JSON 文件名和三个数字|"听到的退化 ↔ 看到的离散表示"这条表达主线断了。token JSON 白导出了|
|切 bitrate 是否保持播放位置|原生 audio 换 src 会归零重放，退化对比进一步弱化|

**结论：不符合要求**。现状是一个"文件浏览器"，不是"codec explorer"——它展示了产物，没有展示**关系**（bitrate ↔ 保真度 ↔ token 稀疏性）。而"关系"才是这个项目在 ACTAM 评分和你 portfolio 叙事里的价值所在。


###### **H10 部署**：

GitHub 建 repo → push → Settings → Pages → main branch / root。注意 assets 若入库,repo 会有几十 MB,GitHub 允许但 push 稍慢。