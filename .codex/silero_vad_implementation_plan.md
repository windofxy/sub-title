# Silero VAD 实现方案

日期：2026-09-13

状态：已落地第一版实现；代码完成后仅进行静态分析，未运行测试或模型。

## 目标

在 Qwen3-ASR 前增加 Silero VAD，用于判断输入音频中是否存在人声，解决背景音乐能量较高时 RMS 静音判断失效的问题。

目标行为：

- 纯背景音乐、环境噪声和静音不触发 Qwen3-ASR 推理；
- 有人声时保留完整语音片段，再交给 Qwen3-ASR；
- 通过 VAD 的语音结束信号辅助 Qwen3-ASR 定稿；
- 保留现有重叠窗口和稳定前缀机制，避免 VAD 切掉词首、词尾；
- VAD 不可用时回退到当前 RMS 策略。

## 依赖与模型文件

根据当前环境检查结果，`silero-vad` 依赖已经包含体积较小的模型文件，因此不需要再通过 ModelScope 单独下载 VAD 权重。

依赖关系：

```text
silero-vad
  ├─ VAD Python API
  └─ 随依赖提供的 Silero VAD 模型文件
```

Qwen3-ASR 仍然使用现有 ModelScope 下载流程。ModelScope 只负责 Qwen 模型，不再作为 VAD 权重的必要下载来源。

`silero-vad` 已加入项目依赖文件，但通过设置界面的开关作为 Qwen3-ASR 的可选运行能力：

- 未选择 Qwen3-ASR 时，不导入、不加载 VAD；
- 开关关闭时，不导入、不加载 VAD；
- 开关打开时，在引擎 `load()` 阶段懒加载 VAD；
- 依赖缺失时给出明确日志，并回退到 RMS 判断；
- 不在模块顶层导入 `silero_vad`，避免影响其他 ASR 引擎启动。

## 推荐处理链路

```text
采集音频
  -> 归一化为 16 kHz mono float32
  -> 切成 512 samples（约 32 ms）VAD 帧
  -> Silero VAD 输出语音概率
  -> 语音开始：创建带 pre-roll 的语音窗口
  -> 语音持续：累积音频并按 cadence 调用 Qwen3-ASR
  -> 语音结束：保留 post-roll，触发 Qwen final
  -> Qwen partial/final 输出到现有 UI
```

VAD 只负责“有没有人声”和“语音段边界”，不负责识别文字。Qwen3-ASR 仍负责文字内容、标点和跨窗口纠错。

## VAD 状态机

建议在 `qwen3_asr_engine.py` 中增加一个内部 VAD 状态对象，至少维护：

```text
vad_model              Silero VAD 实例
vad_state              模型内部状态
vad_is_speech          当前是否处于语音段
vad_speech_run         连续达到语音阈值的样本数
vad_silence_run        连续低于静音阈值的样本数
vad_pre_roll           语音开始前保留的音频
vad_post_roll          语音结束后保留的音频
```

不要仅根据单个 VAD 帧切换状态，应使用滞回和最短持续时间：

```text
speech_prob >= 0.60，持续 150~250 ms -> 开始语音段
speech_prob <= 0.30，持续 400~600 ms -> 结束语音段
0.30 <= speech_prob < 0.60             -> 保持上一状态
```

## 分帧与输入块

当前 pipeline 默认每约 0.6 秒向 Qwen 引擎发送一个块，而 VAD 需要更细的时间分辨率。因此不能直接用整个 pipeline 块计算一次 VAD。

建议在 Qwen 引擎内部继续接收现有块，但在内部拆成 20 ms（320 samples）或 30 ms（480 samples）的小帧。末尾不足一帧的样本放入 `_vad_buf`，等待下一次 `feed()` 补齐。VAD 的状态连续维护，不因每次 `feed()` 调用而重置。

## 前置与后置音频缓存

VAD 的开始时间通常晚于实际发声时间，结束时间也可能早于尾音结束。因此需要缓存：

```text
pre-roll  = 200~300 ms
post-roll = 200~300 ms
```

语音开始时，把 `pre-roll` 加入当前 Qwen 音频窗口，避免切掉声母、辅音或句首短音节。语音结束时，不要立即清空窗口，继续保留 `post-roll`，再等待总静音达到约 450~600 ms 后触发 Qwen final。

VAD 边界缓存与 Qwen 的 400 ms 重叠窗口职责不同：VAD 缓存保护语音活动边界，Qwen overlap 保护 ASR 识别窗口边界。两者可以同时存在，但应限制总缓冲区大小。

## 与 Qwen3-ASR 的配合

### 无语音阶段

当 VAD 没有进入语音状态时：

- 不调用 Qwen3-ASR；
- 丢弃纯静音和背景音乐缓存，只保留有限 pre-roll；
- 不发送 partial 或 final；
- 不触发翻译。

### 语音阶段

进入语音状态后：

1. 将 pre-roll 和后续语音加入 Qwen 未定稿缓冲区；
2. 继续使用约 2 秒的识别 cadence；
3. cadence 到期只发送 `is_final=False`；
4. 使用 Qwen 重叠窗口和稳定前缀持续纠错；
5. VAD 结束且静音满足阈值后，发送 `is_final=True`。

VAD 只提供结束候选，不应把单帧低概率直接当成 final：

```text
VAD 连续无语音 >= 450~600 ms
且当前窗口存在有效语音或有效 partial
    -> Qwen final
```

如果 Qwen 最近结果以句末标点结束，可以把静音阈值缩短到约 250~300 ms，但仍需要 VAD 连续确认。

## 模型加载与设备

Qwen3-ASR 使用 GPU 时，Silero VAD 可以优先使用 CPU：

- VAD 模型小，CPU 推理成本低；
- 避免 VAD 和 4-bit Qwen 争抢 GPU 显存；
- 音频帧很小，CPU 延迟通常足够低。

建议：

```text
Qwen3-ASR：沿用 qwen3_asr_device
Silero VAD：默认 CPU
```

如果 Silero API 根据输入模型自动加载权重，不要在项目中复制一份模型文件。只保存 VAD 对象和内部状态，停止识别时释放对象。

## 配置建议

建议增加以下 `AsrConfig` 字段：

```text
qwen3_asr_vad_enabled: false
qwen3_asr_vad_start_threshold: 0.60
qwen3_asr_vad_end_threshold: 0.30
qwen3_asr_vad_start_seconds: 0.20
qwen3_asr_vad_end_seconds: 0.50
qwen3_asr_vad_pre_roll_seconds: 0.25
qwen3_asr_vad_post_roll_seconds: 0.25
```

默认值应只对 Qwen3-ASR 生效，其他引擎不读取这些字段。设置界面的“Silero 语音检测”开关写入 `qwen3_asr_vad_enabled`。关闭时恢复当前 RMS 路径，便于对比 VAD 引入前后的识别效果。

## 背景音乐和歌声的边界

Silero VAD 判断的是“是否存在类似语音的人声活动”，不是“是否为对白”：

- 纯伴奏通常可以被过滤；
- 人声对白叠加音乐通常可以检测；
- 歌曲演唱可能被判定为语音并送入 Qwen；
- 广播、喊叫、配音的结果取决于模型。

如果产品要求过滤歌词，仅引入 Silero VAD 不足，还需要人声/伴奏分离模型、语音/歌声/音乐分类模型，或根据来源只对麦克风通道启用对白识别。这些方案会增加依赖、模型体积和延迟，不应与第一阶段 VAD 同时强制启用。

## 回退与错误处理

建议按以下优先级处理：

```text
Silero VAD 可用 -> VAD + Qwen
silero-vad 缺失 -> 记录一次 warning，回退 RMS + Qwen
VAD 推理异常   -> 重置 VAD 状态，当前窗口继续由 Qwen 处理
Qwen 推理异常  -> 保留现有 Qwen 错误处理和日志
```

回退日志应明确说明：

```text
Silero VAD unavailable; falling back to RMS silence detection
```

避免因为可选 VAD 依赖缺失而阻止 Qwen3-ASR 启动。

## 性能与可观测性

需要记录以下指标：

- VAD 帧处理耗时；
- VAD 语音段数量；
- VAD 判定的语音总时长；
- Qwen 实际推理次数；
- 被 VAD 过滤的非语音时长；
- 音频队列丢块数量。

VAD 可以减少背景音乐阶段的无效 Qwen 推理，但不能降低正在说话时的 Qwen 计算量。重叠窗口仍需遵守已有的 300~500 ms 范围。

## 实现顺序

### 第一阶段：接入 Silero VAD

1. 添加可选依赖检查；
2. 在 Qwen 引擎中懒加载 VAD；
3. 将输入拆成 512 samples（约 32 ms）帧；
4. 实现语音开始/结束滞回；
5. 无语音时跳过 Qwen 推理；
6. VAD 不可用时回退 RMS。

### 第二阶段：加入边界缓存

1. 增加 200~300 ms pre-roll；
2. 增加 200~300 ms post-roll；
3. 将 VAD 结束转为 Qwen final 候选；
4. 与现有重叠窗口和稳定前缀联调。

### 第三阶段：优化背景音乐场景

1. 统计纯伴奏误触发率；
2. 调整开始/结束阈值和持续时间；
3. 针对歌声场景评估是否需要人声分离；
4. 根据 Qwen 推理耗时调整 cadence 和 overlap。

## 验收标准

- `silero-vad` 已安装时，Qwen 可正常加载 VAD；
- VAD 模型使用依赖内置文件，不再额外依赖 ModelScope VAD 权重下载；
- 纯背景音乐阶段不产生 Qwen 字幕；
- 背景音乐叠加对白时，句首和句尾不被 VAD 截断；
- 连续短暂噪声不会触发语音段；
- 语音结束约 450~600 ms 后提交 final；
- partial 仍由现有 UI interim 机制覆盖；
- final 仍只触发一次翻译；
- VAD 依赖缺失时可以回退 RMS，不影响其他 ASR 引擎；
- VAD 不可用、推理异常和采集丢块在日志中可区分。
