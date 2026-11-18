# AIDE 数据集与实验环境说明

> 本文档说明 AIDE（Audio Immunization Against Harmful Audio Editing with Diffusion Models）的数据来源、数据组织方式、预处理规则、模型权重、软件环境和运行前检查。
>
> 为避免把实现约定误写成论文事实，全文使用以下口径：
> - **论文明确说明**：可以直接从论文实验设置或方法部分确认。
> - **当前代码约定**：由 `repro_v2/` 中的实现、配置和脚本决定。
> - **待确认/待补齐**：论文没有给出足够细节，或当前仓库尚未填入真实配置。此类内容不得凭经验猜测后写成实验事实。

## 1. 数据与模型总览

| 数据/模型 | 用途 | 当前流程中的作用 | 状态 |
|---|---|---|---|
| AudioCaps | 主实验数据集 | 音频样本、良性文本描述；主基准评估 | 论文使用；需记录具体版本与官方 split |
| VCTK | 多说话人干净语音数据集 | 语音条件与跨语音数据评估 | 论文使用；具体子集/split 需记录 |
| AART prompts | 有害 Prompt 来源之一 | 提供 PSM 风险分类训练/评估文本 | 需要准备本实验实际使用的 CSV |
| RED-INSTRUCT / HarmfulQA 等 | 有害 Prompt 来源之一 | 补充红队 Prompt | 需要固定具体子集和文件版本 |
| Fixed target pool | Stage I 潜空间目标 | 为每条输入音频确定目标 latent | 论文定义固定目标池；具体来源需记录 |
| AudioLDM-S-Full-v2 | 冻结的扩散生成骨干 | VAE、UNet、CLAP 条件编码器、DDIM 与 vocoder | 当前代码默认模型 |
| RoBERTa-base | PSM 文本编码器 | Prompt 表征与七分类/门控预测 | 当前代码默认模型 |
| AudioLDM evaluation feature extractor | FD/IS/KL 等分布指标 | 生成音频与参考音频的分布比较 | 评估时按需下载/加载权重 |

论文实验设置报告：NVIDIA A100、CUDA 12.1、Python 3.10、PyTorch 2.1.0 和 torchaudio 2.1.0。论文没有报告 GPU 显存容量、CPU 型号或完整磁盘需求，因此本仓库不虚构最低显存/磁盘门槛。

## 2. 数据集来源与使用方式

### 2.1 AudioCaps（主基准）

- 论文用途：主要音频基准；良性 Prompt 主要来自 AudioCaps 原始 caption。
- 当前代码用途：`build-prompts` 从 CSV 的 caption 列读取良性文本；`manifest` 从音频目录读取 WAV/FLAC 文件。
- 官方项目：[AudioCaps 项目主页](https://audiocaps.github.io/)；[官方代码与数据说明](https://github.com/cdjkim/audiocaps)。
- CSV 至少需要包含代码可识别的 caption 列（`caption`、`Caption`、`text` 或 `sentence` 之一）。官方原始 CSV 常见字段包括 `audiocap_id`、`youtube_id`、`start_time` 和 `caption`。
- **版本注意**：AudioCaps 上游仓库已发布 AudioCaps 2.0。论文只写明 AudioCaps，没有在正文中明确标注 V1/V2.0。复现实验时必须记录实际下载版本、split 文件和文件哈希；不得未经记录直接切换版本。
- **划分注意**：论文描述按 AudioLDM protocol 使用官方 train/validation/test split。当前 `build_prompt_dataset` 接收单个 CSV，并按类别以 seed=2025 默认随机拆分为 train/dev/test（默认比例 80%/10%/10%）；这不等同于自动保留 AudioCaps 官方 split。当前 `run_full_pipeline.sh` 也只接收一个 `AUDIOCAPS_CSV`。若要严格复现官方 split，应先提供已按官方 split 整理的 CSV，或在正式实验前扩展脚本为 split-aware 输入，并审计音频与 caption 是否跨 split 泄漏。

### 2.2 VCTK

- 论文用途：第二个数据集，用于干净多说话人英语语音上的评估。
- 推荐固定版本：[CSTR VCTK Corpus v0.92（Edinburgh DataShare）](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)，DOI：[10.7488/ds/2645](https://doi.org/10.7488/ds/2645)。
- 当前代码可以递归扫描 VCTK 解压目录中的 WAV/FLAC 文件；具体训练/测试说话人和句子划分由调用者提供。
- 论文没有在实验配置中列出具体 VCTK speaker ID、句子列表或 split 文件。正式报告必须保存所用 speaker/utterance 清单及其哈希，不能把自行划分描述成论文原始划分。

### 2.3 AART Prompt

- 论文用途：与 RED-INSTRUCT 一起构建有害 Prompt 空间。
- AART 论文/项目：[AART: AI-Assisted Red-Teaming with Diverse Data Generation](https://research.google/pubs/aart-ai-assisted-red-teaming-with-diverse-data-generation-for-new-llm-powered-applications/)。
- AART 是红队数据生成方法/框架；不能仅凭“AART”名称推断存在一个与本论文完全一致、可直接下载的固定 CSV。请保留本实验实际使用的 Prompt CSV、生成 recipe、生成日期、字段和标签来源。
- 当前代码支持的文本列：`prompt`、`Prompt`、`question`、`Question`、`input`、`text`、`Text`。
- 当前代码支持的类别列：`crime`、`Crime`、`category`、`Category`、`label`、`Label`。类别值必须在 `configs/taxonomy.json` 的 `aart_label_map` 中显式映射到 1–6。
- 若原始 CSV 没有类别列，代码允许配置 `__all__`，但这只适用于有明确实验依据的单一类别数据；不能把它当作自动推断六类标签的办法。

### 2.4 RED-INSTRUCT / HarmfulQA

- 论文用途：有害 Prompt 来源之一。
- 上游项目：[declare-lab/red-instruct](https://github.com/declare-lab/red-instruct)；数据集：[HarmfulQA on Hugging Face](https://huggingface.co/datasets/declare-lab/HarmfulQA)。
- RED-INSTRUCT 项目包含不同红队问题集/任务资源。论文没有在正文中完整列出本实验所用文件清单，因此必须在实验记录中明确具体采用的子集、下载版本/commit、语言、去重规则和标签映射；不要默认把仓库中所有文件混在一起。
- 当前代码读取 `RED_ROOT` 下递归找到的 `.json` / `.jsonl` 文件，并按文件名 stem 查找 `red_file_map`。文本抽取字段优先识别 `prompt`、`question`、`input`、`text`、`instruction`、`query`。
- 每个文件名 stem 必须在 `configs/taxonomy.json` 的 `red_file_map` 中显式映射到 1–6；未映射文件会报错，不会静默归类。

### 2.5 Fixed target pool（Stage I）

- 论文方法定义固定目标池，并从目标池选择目标音频 (x_{tar})，用于目标 latent 引导。
- 论文没有明确给出目标池的完整来源、文件列表或组成比例。当前实现要求通过 `--target-root` / `TARGET_POOL` 显式提供。
- 当前实现按 run seed 和 source audio ID 确定性选择目标，并在 `target_manifest.jsonl` 记录目标路径与 SHA256。
- 仓库 README 建议把噪声池、VCTK 语音池和混合池分开做受控实验。这属于当前复现工程的实验组织建议，不应写成论文已经明确规定的目标池构成。
- 目标池不得与当前受保护输入/测试样本发生未记录的重叠。每种池单独保存目录、文件清单、来源和哈希。

### 2.6 分布指标参考音频

FD/IS/KL 评估需要生成音频目录和参考音频目录。当前脚本支持通过 `REFERENCE_BENIGN_DIR` 与 `REFERENCE_HARMFUL_DIR` 分别提供参考集合。

- 参考音频必须与待评估子集在数据域、采样率、时长处理和评估协议上匹配。
- 不要将 AudioCaps 环境音参考集与 VCTK 语音生成结果混合计算同一组分布指标。
- 当前评估器要求生成目录和参考目录至少各有 2 个 WAV 文件。
- 参考音频清单及其 SHA256 应随实验归档。

## 3. 推荐的数据目录结构

以下是建议的本地组织方式，不是数据集官方压缩包的原始目录结构。路径可以自行调整，但运行脚本中的环境变量必须对应。

```text
/data/aide/
├── audiocaps/
│   ├── audio/
│   │   ├── train/                 # 官方 train split 音频
│   │   ├── validation/            # 官方 validation split 音频
│   │   └── test/                  # 官方 test split 音频
│   └── metadata/
│       ├── captions_train.csv
│       ├── captions_validation.csv
│       └── captions_test.csv
├── vctk/
│   ├── wav48/                     # 或解压包中的实际 WAV 根目录
│   └── split_manifests/           # 自行确定并归档的 speaker/utterance 划分
├── prompts/
│   ├── aart_prompts.csv
│   └── red_instruct/
│       ├── <category_file_1>.json
│       └── <category_file_2>.jsonl
├── target_pool/
│   ├── noise/
│   ├── vctk_speech/
│   └── mixed/
├── references/
│   ├── audiocaps_benign/
│   ├── audiocaps_harmful/
│   ├── vctk_benign/
│   └── vctk_harmful/
└── runs/                          # 所有运行输出放在仓库外
```

模型权重建议放在独立缓存目录，例如 `/models/aide/`。数据集、模型权重、生成 WAV、缓存、日志和机器绝对路径均不应提交到 GitHub。

## 4. 音频预处理与数据清单

### 4.1 当前代码预处理规则

当前 `aide_repro/audio.py` 的输入规则如下：

1. 递归读取目录中的 `.wav` 和 `.flac`。
2. 多声道音频按通道求均值转为单声道。
3. 重采样至 16,000 Hz。
4. 去除波形均值，并进行峰值归一化，目标峰值为 0.5。
5. 按 AudioLDM mel 前端长度裁剪或补零。
6. 检查音频是否存在 NaN/Inf。
7. 保存扰动音频时使用 PCM_24 WAV。

**时长口径提醒**：配置中的 `duration_seconds=10.0` 用于得到 1024 个 mel 帧；当前音频工具再按 1024 × 160 个采样点拟合波形。在 16 kHz 下对应 163,840 个采样点，即 10.24 秒。该实现细节应在所有实验中保持一致并记录，不能简单把实际输入长度写成恰好 10.00 秒。

### 4.2 音频 manifest

执行示例：

```bash
python -m aide_repro.cli manifest \
  --input-root /data/aide/audiocaps/audio/train \
  --output /data/aide/runs/audio_manifest_train.jsonl \
  --duration 10.0 \
  --sample-rate 16000
```

manifest 当前记录：
- `audio_id`：文件 stem 与 SHA256 前缀构成的稳定 ID；
- `path`：本机绝对路径；
- `sha256`：原始文件 SHA256；
- 原始采样率、通道数、帧数和时长；
- 模型采样率、模型输入采样点数和配置时长。

manifest 仅在本地运行目录保存。由于其中包含绝对路径，不要直接提交到公开仓库。

## 5. Prompt 数据构建与标签要求

当前模板位于 `configs/taxonomy.json`：

```json
{
  "labels": [
    "benign",
    "harmful_1",
    "harmful_2",
    "harmful_3",
    "harmful_4",
    "harmful_5",
    "harmful_6"
  ],
  "benign_label_id": 0,
  "aart_label_map": {},
  "red_file_map": {}
}
```

- ID 0 为 benign；ID 1–6 为六个 harmful subclasses。
- 当前映射表为空，必须根据实际 AART CSV 标签和 RED-INSTRUCT 文件清单补齐。
- 当前构建器会做小写/空白规范化后的精确去重；同一文本若对应不同标签会直接报错。
- 默认按类别和随机种子 2025 划分 train/dev/test，比例为 80%/10%/10%。某一类别少于 3 条时，当前实现会将该类别样本全部放入 train，因此正式运行前必须检查 `dataset_stats.json`，确保 dev/test 也有可用样本。
- 构建后检查 `label_map.json`、`dataset_stats.json` 和三个 JSONL 文件；人工抽查每个类别的文本和标签。

构建命令：

```bash
python -m aide_repro.cli build-prompts \
  --audiocaps-csv /data/aide/audiocaps/metadata/captions_train.csv \
  --aart-csv /data/aide/prompts/aart_prompts.csv \
  --red-root /data/aide/prompts/red_instruct \
  --taxonomy configs/taxonomy.json \
  --output-dir /data/aide/runs/prompts
```

> **标签核对阻断项**：论文方法部分描述 benign + six harmful subsets（共 7 类），但论文分类稳定性图中可见的逐类 F1 标签列出了 safe 和五个 harmful 名称。论文正文没有完整、无歧义地给出六个 harmful subclasses 的名称与来源映射。当前仓库因此保留 `harmful_1` 至 `harmful_6` 占位符。必须回到实际 AART/RED-INSTRUCT 原始标注和实验记录核实后再填，不得根据常识臆造第六类，也不得在没有依据时把多个类别合并。

## 6. 模型权重与缓存

| 组件 | 当前配置 | 权重来源/说明 |
|---|---|---|
| Diffusion backbone | `audioldm-s-full-v2` | 使用原始 AudioLDM 仓库与对应 checkpoint |
| AudioLDM source | commit `4054cb418ba3d947d94f6ad302f1d6a320e2313c` | 由 `scripts/install_env.sh` 固定 checkout |
| PSM text encoder | `roberta-base` | Hugging Face 模型；需联网下载或预先缓存 |
| CLAP | AudioLDM conditioner 中的 CLAP | 当前实现复用 AudioLDM 加载的权重，不另行指定独立 CLAP checkpoint |
| FD/IS/KL evaluator | AudioLDM evaluator，backbone=`mert` | 评估时可能需要额外依赖与预训练特征权重 |

AudioLDM checkpoint 默认在首次初始化模型时下载；也可以通过 `--audio-checkpoint` 显式指定本地权重。记录 checkpoint 的文件名、来源、大小和 SHA256。不要只记录模型名称。

## 7. 软件与硬件环境

### 7.1 论文报告环境

| 项目 | 论文报告值 |
|---|---|
| GPU | NVIDIA A100 |
| CUDA | 12.1 |
| Python | 3.10 |
| PyTorch | 2.1.0 |
| torchaudio | 2.1.0 |

论文未报告 A100 的具体显存规格，也未报告最低显存、CPU、内存或磁盘要求。运行前应以实际机器的 `nvidia-smi`、`torch.cuda.get_device_name()` 和显存占用日志为准。

### 7.2 当前仓库依赖

核心版本由 `requirements.txt` 固定：

| 包 | 版本 |
|---|---|
| torch | 2.1.0 |
| torchaudio | 2.1.0 |
| torchvision | 0.16.0 |
| numpy | 1.26.4 |
| librosa | 0.10.1 |
| transformers | 4.36.2 |
| timm | 0.9.12 |
| torchlibrosa | 0.1.0 |
| pytorch-lightning | 2.1.3 |
| scikit-image | 0.22.0 |
| torchmetrics | 1.3.2 |
| omegaconf | 2.3.0 |

此外还安装 requirements 中列出的 scipy、soundfile、tqdm、PyYAML、pandas、scikit-learn、sentencepiece、einops、progressbar2、ftfy、braceexpand、webdataset 和 chardet。未固定版本的依赖以本次实际安装导出的环境清单为准。

### 7.3 安装环境

在仓库根目录执行：

```bash
conda create -n aide python=3.10 -y
conda activate aide

cd repro_v2
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
bash scripts/install_env.sh
```

安装脚本会：
1. 安装 torch 2.1.0、torchaudio 2.1.0、torchvision 0.16.0 的 CUDA 12.1 wheel；
2. 安装 `requirements.txt`；
3. 克隆 AudioLDM 并 checkout 固定 commit；
4. 以 editable 模式安装 AudioLDM 和 `aide_repro`；
5. 执行环境导入检查。

可选地指定模型缓存目录：

```bash
export AUDIOLDM_CACHE_DIR=/models/aide/audioldm
```

CPU-only 主机可在安装前设置：

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

CPU 环境适合静态检查和部分单元测试，不代表能够完成论文要求的完整 GPU 训练与生成实验。

### 7.4 环境检查

```bash
python -m aide_repro.cli check-env
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`check-env` 只检查 Python/PyTorch/torchaudio/CUDA 状态和 AudioLDM 包导入，不代表 checkpoint 已成功加载，也不代表梯度链路已通过。

## 8. 完整运行前需要设置的路径

当前一键脚本 `scripts/run_full_pipeline.sh` 读取以下环境变量：

| 环境变量 | 必填 | 说明 |
|---|---:|---|
| `TRAIN_AUDIO_ROOT` | 是 | Stage I 训练音频目录 |
| `TEST_AUDIO_ROOT` | 是 | Stage I 测试音频目录 |
| `TARGET_POOL` | 是 | 固定目标音频池 |
| `AUDIOCAPS_CSV` | 是 | AudioCaps caption CSV |
| `AART_CSV` | 是 | AART Prompt CSV |
| `RED_ROOT` | 是 | RED-INSTRUCT JSON/JSONL 根目录 |
| `RUN_ROOT` | 是 | 本次实验输出目录 |
| `TAXONOMY` | 否 | taxonomy JSON；默认 `configs/taxonomy.json` |
| `MODEL_NAME` | 否 | 默认 `audioldm-s-full-v2` |
| `AUDIO_CHECKPOINT` | 否 | 本地 AudioLDM checkpoint；不填则由上游模型按默认方式加载 |
| `REFERENCE_BENIGN_DIR` | 否 | 良性生成的参考音频目录，启用分布指标 |
| `REFERENCE_HARMFUL_DIR` | 否 | 有害生成的参考音频目录，启用分布指标 |

示例（路径按实际机器修改）：

```bash
cd repro_v2

export TRAIN_AUDIO_ROOT=/data/aide/audiocaps/audio/train
export TEST_AUDIO_ROOT=/data/aide/audiocaps/audio/test
export TARGET_POOL=/data/aide/target_pool/vctk_speech
export AUDIOCAPS_CSV=/data/aide/audiocaps/metadata/captions_train.csv
export AART_CSV=/data/aide/prompts/aart_prompts.csv
export RED_ROOT=/data/aide/prompts/red_instruct
export TAXONOMY=$PWD/configs/taxonomy.json
export RUN_ROOT=/data/aide/runs/audiocaps_main

# 可选：指定本地权重和匹配的参考音频
# export AUDIO_CHECKPOINT=/models/aide/audioldm-s-full-v2.ckpt
# export REFERENCE_BENIGN_DIR=/data/aide/references/audiocaps_benign
# export REFERENCE_HARMFUL_DIR=/data/aide/references/audiocaps_harmful

bash scripts/run_full_pipeline.sh
```

VCTK 实验应使用独立的 `TRAIN_AUDIO_ROOT`、`TEST_AUDIO_ROOT`、`RUN_ROOT` 和参考音频目录，单独归档其 speaker/utterance split；不要把 AudioCaps 与 VCTK 混在同一个 manifest 或同一组分布指标中。

## 9. 正式实验前的检查门槛

按以下顺序执行，不要跳过失败项：

1. **数据核验**：检查数据版本、官方 split、文件数量、音频可读性、CSV 字段、标签分布和 SHA256。
2. **环境核验**：执行 `check-env`，确认 Python、torch、torchaudio、CUDA 与 GPU 信息。
3. **模型核验**：真实加载 AudioLDM checkpoint，记录 checkpoint SHA256；确认 RoBERTa 与 CLAP 权重可用。
4. **单样本预处理**：生成一个 manifest，检查采样率、通道数、输入长度、归一化和音频听感。
5. **Stage I 单样本**：只运行一个样本，检查 PGD loss、梯度有限性、扰动 (L_\infty\) 是否不超过 epsilon，并检查 raw/delta/adv 三份 WAV。
6. **真实生成梯度检查**：执行 `gradient-check`；若梯度缺失、为零或非有限，停止，不得开始 Eq. (7) 生成耦合训练。
7. **Stage II 单步训练**：先运行一个 generation-coupled PSM 更新，检查 (L_{cls})、(L_{gate})、(L_{gen})、门控梯度和 checkpoint。
8. **小规模评估**：先检查 CLAP/H-CLAP、SNR/LSD 与 FD/IS/KL 输入目录和样本配对，再启动全量实验。

## 10. 每次实验必须归档的信息

建议每次运行在 `RUN_ROOT` 保存一份 `experiment_record.json` 或同等记录，至少包含：

- Git branch、commit SHA、工作区是否干净；
- 操作系统、Python、torch、torchaudio、CUDA、GPU 型号和显存；
- AudioLDM source commit、模型名称、checkpoint SHA256；
- RoBERTa 模型标识/版本及缓存来源；
- AudioCaps、VCTK、AART、RED-INSTRUCT 的具体版本、来源、split 文件和 SHA256；
- target pool 的来源、组成、文件清单和 SHA256；
- taxonomy JSON、label map、各 split 类别计数；
- 所有命令行、环境变量（移除密钥）、随机种子和完整超参数；
- Stage I 的 manifest、目标选择清单、扰动指标；
- Stage II 的训练日志、验证指标、checkpoint SHA256；
- 生成 WAV、sidecar metadata、CLAP/H-CLAP、FD/IS/KL、SNR/LSD 结果。

数据、模型权重和生成结果保存在受控的本地/对象存储中；GitHub 只保存代码、配置模板、文档和不含隐私路径的必要摘要。

## 11. 当前仍需确认的复现事项

以下事项在现有论文或代码中没有完全确定，正式复现实验前必须核实：

1. AudioCaps 实际使用的发布版本，以及官方 split 对应的音频和 caption 文件清单。
2. VCTK 的确切版本、speaker/utterance split 和是否排除任何说话人。
3. AART 实际使用的生成 recipe、Prompt CSV 和类别标注。
4. RED-INSTRUCT 实际使用的子集、文件版本与类别映射。
5. 六个 harmful subclasses 的完整名称和标签映射；当前 taxonomy 仍是占位配置。
6. Fixed target pool 的实际数据来源、组成和文件清单。
7. A100 的显存规格及实际运行时峰值显存；论文未给出这些信息。
8. 评估特征提取器的具体权重版本与下载缓存。

**当前结论**：仓库已提供数据读取、manifest、Prompt 构建和环境安装入口，但数据集文件、标签映射、预训练权重和机器相关路径需要使用者在本地准备。只有完成上述核验并通过真实模型梯度检查后，才能将对应实验标记为可复现/已跑通。
