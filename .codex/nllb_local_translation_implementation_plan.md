# NLLB-200 本地翻译引擎接入方案

## 1. 当前状态

项目目前已经存在一个 `nllb` 翻译选项，但它并没有在桌面进程内加载 NLLB-200 模型，而是通过 HTTP 调用外部 `nllb-serve` 服务：

- `src/subtitle/translate/nllb_engine.py` 请求 `http://host:port/translate`；
- `src/subtitle/translate/factory.py` 已注册 `nllb`；
- 设置界面已经提供 NLLB-200 选项和 host/port 配置；
- `TranslationConfig` 已包含 `nllb_host`、`nllb_port`。

因此当前功能是“支持 NLLB 服务”，不是“内置 NLLB 模型”。

## 2. 推荐架构

新增独立的 `nllb_local` 引擎，保留现有的 `nllb` 服务模式：

```text
nllb       -> HTTP 调用外部 nllb-serve
nllb_local -> 当前进程直接加载 NLLB-200
```

不建议直接改变现有 `nllb` 的语义。保留远程服务可以兼容已有 WSL 配置，也便于用户在服务模式和内嵌模型模式之间选择。

## 3. 模型与推理方式

推荐使用 Transformers 加载 NLLB-200：

```text
AutoTokenizer
AutoModelForSeq2SeqLM
```

推理流程：

```text
原文句子
  -> tokenizer 设置 src_lang
  -> 编码文本
  -> model.generate()
  -> forced_bos_token_id 指定目标语言
  -> tokenizer.decode()
```

NLLB 的目标语言必须使用专用语言码，并通过 `forced_bos_token_id` 指定，例如：

```text
eng_Latn
zho_Hans
zho_Hant
jpn_Jpan
kor_Hang
```

不能只将普通的 `zh-Hans` 或 `en` 直接传给模型。

## 4. 模型选择

桌面端初始版本建议使用：

```text
facebook/nllb-200-distilled-600M
```

该模型在模型大小、语言覆盖和翻译质量之间相对平衡。更大的 NLLB 模型会明显增加内存、显存和推理延迟。

正式实现前需要确认该模型在 ModelScope 上有可用镜像和稳定的模型 ID。项目当前模型下载策略是不自动回退到 Hugging Face，因此应明确：

- 使用的 ModelScope 模型 ID；
- 模型缓存位置；
- 首次下载大小；
- 下载失败时的用户提示。

## 5. 依赖

NLLB 本地引擎需要以下可选依赖：

```text
transformers
sentencepiece
safetensors
torch
```

用途如下：

- `transformers`：模型和 tokenizer；
- `sentencepiece`：NLLB tokenizer；
- `safetensors`：安全加载模型权重；
- `torch`：模型推理运行时。

这些依赖不应成为纯 API 模式的强制依赖。应沿用当前本地模型依赖分组，并在用户选择 `nllb_local` 但依赖缺失时提供明确安装提示。

## 6. ModelScope 下载

项目已有 `src/subtitle/asr/modelscope_hub.py` 的下载封装，推荐复用：

```python
model_path = download_modelscope(model_id, "NLLB-200")
```

翻译引擎应接收本地快照目录，再传给 Transformers：

```text
ModelScope snapshot
        |
        v
本地模型目录
        |
        v
AutoTokenizer.from_pretrained(local_path)
AutoModelForSeq2SeqLM.from_pretrained(local_path)
```

这样可以避免运行时偷偷访问 Hugging Face，也与项目现有 ASR 模型下载行为保持一致。

## 7. 设备与精度

建议新增本地 NLLB 配置：

```text
nllb_local_model
nllb_local_device
nllb_local_dtype
nllb_local_max_new_tokens
nllb_local_num_beams
```

默认策略：

```text
CUDA -> float16
CPU  -> float32
```

初始版本不建议立即加入 4-bit 或 8-bit 量化。量化会增加 bitsandbytes、CUDA 和平台兼容问题，可以作为后续优化单独加入。

运行时需要预留数 GB 内存或显存，并接受 CPU 模式下秒级翻译延迟。

## 8. 线程与生命周期

当前 `TranslationWorker` 在启动时创建翻译器，并用 `ThreadPoolExecutor(max_workers=2)` 执行翻译。内置 NLLB 不能简单照搬这一模式。

需要满足：

1. 模型加载不能阻塞 UI 线程；
2. 同一个 GPU 模型不应被两个翻译任务同时生成；
3. 翻译结果必须保持字幕顺序。

推荐采用延迟加载：

```text
TranslationWorker.start()
    -> 只创建轻量 NllbLocalTranslator

首次 translate()
    -> 在线程池中加锁加载模型

模型加载完成
    -> 后续句子复用同一模型
```

本地 NLLB 建议使用单线程翻译，或在模型生成阶段使用模型级锁。Azure、Google 等远程引擎可以继续使用现有并发线程池。

如果需要在启动识别时预加载模型，应将预加载任务放到后台线程，并在 UI 中显示加载状态，不能直接在主线程执行 `from_pretrained()`。

## 9. 语言码与自动检测

当前翻译接口只传入：

```python
translator.translate(text)
```

ASR 检测出的语言没有传给翻译器。NLLB 本身不支持真正的 `auto` 源语言。

当前远程 NLLB 实现把 `auto` 退化为 `eng_Latn`，这对中文或其他语言输入是不安全的。

推荐处理方式：

1. `nllb_local` 默认不允许 `auto`；
2. 用户必须选择源语言；
3. 选择 `auto` 时在设置界面提示 NLLB 需要明确源语言；
4. 后续扩展翻译接口，将 ASR 的语言信息传给翻译器。

不建议为了支持 `auto` 额外引入语言检测模型，否则会增加模型体积和翻译延迟。

建议维护独立的 ISO/BCP-47 到 NLLB 语言码映射，例如：

```text
en       -> eng_Latn
zh-Hans  -> zho_Hans
zh-Hant  -> zho_Hant
ja       -> jpn_Jpan
ko       -> kor_Hang
fr       -> fra_Latn
de       -> deu_Latn
es       -> spa_Latn
ru       -> rus_Cyrl
ar       -> arb_Arab
```

## 10. 推理参数

字幕通常是短句，但 Qwen 可能产生过长的定稿句，因此仍需要输入限制：

```text
max_source_length: 512
max_new_tokens: 128
num_beams: 2 或 4
do_sample: False
```

超过最大输入长度时，应按照句末标点或安全词边界拆分翻译，再拼接结果。不能直接静默截断，否则会丢失句尾内容。

初始版本建议使用确定性生成，不使用采样，以减少同一句重复翻译时的结果波动。

## 11. 设置界面

保留现有远程 NLLB 配置，并增加内置模型配置：

```text
翻译引擎：
- Azure
- Google
- LibreTranslate
- NLLB-200 服务
- NLLB-200 内置模型
```

选择 `nllb` 时显示：

- host；
- port；
- 外部服务启动说明。

选择 `nllb_local` 时显示：

- 模型选择；
- 推理设备；
- Beam size；
- 最大输出 token；
- 模型状态；
- 下载和加载提示。

现有“测试连接”按钮对 `nllb_local` 应改为“测试模型”，并在后台线程中执行一次短文本翻译，避免阻塞设置界面。

## 12. 工厂与模块边界

推荐新增文件：

```text
src/subtitle/translate/nllb_local_engine.py
```

工厂新增分支：

```text
engine == "nllb_local"
    -> NllbLocalTranslator
```

不要把本地 Transformers 加载逻辑写入现有 `nllb_engine.py`，避免远程 HTTP 客户端和本地模型生命周期混在一起。

## 13. 错误处理

本地模型需要统一转换为 `TranslatorError` 的情况包括：

- `transformers` 未安装；
- `sentencepiece` 未安装；
- ModelScope 下载失败；
- 模型目录损坏；
- 语言码不存在；
- CPU/GPU 内存不足；
- 输入超过模型限制；
- 模型生成空结果。

错误应沿用当前 `TranslationWorker` 的提示机制，不能因为翻译失败中断 ASR 字幕主流程。

## 14. 许可证与分发

NLLB-200 模型不能视为普通 MIT/Apache 依赖。正式接入前需要确认模型许可证，尤其是：

- 商业用途限制；
- 模型权重再分发限制；
- 是否允许打包进安装程序；
- ModelScope 镜像的再分发条款。

建议应用只在首次使用时自动下载模型，不把权重打包进项目或安装包，并在项目文档中注明模型来源和许可证。

## 15. 推荐落地顺序

1. 保留现有 HTTP `nllb` 引擎。
2. 新增 `nllb_local_engine.py`。
3. 增加 `nllb_local` 工厂分支。
4. 增加可选依赖检测。
5. 复用 ModelScope 下载封装。
6. 实现线程安全的延迟加载。
7. 使用单线程顺序翻译。
8. 增加 NLLB 语言码映射和 `auto` 限制。
9. 在设置界面增加本地模型选项和参数卡片。
10. 在测试按钮中增加本地模型加载和短句翻译检查。

## 16. 结论

项目已经具备 NLLB 的远程服务接口。真正将 NLLB-200 模型并入项目时，最稳妥的方案是新增 `nllb_local`，采用 Transformers、ModelScope 和后台延迟加载，并将本地模型翻译限制为明确源语言和单线程顺序推理。
