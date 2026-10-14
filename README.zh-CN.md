# A2Seek

用于航拍异常理解，提供训练、推理和评估代码。

[论文](https://arxiv.org/abs/2505.21962) · [项目主页](https://2-mo.github.io/A2Seek/) · [English](README.md)

[![A2Seek 总览](docs/static/images/overview.png)](docs/static/images/overview.png)

*A2Seek 总览（论文图 1）。*

以下命令均在仓库根目录执行。训练和推理需要 Linux 和 NVIDIA GPU。

| 资源 | Hugging Face 仓库 | 本地目录 |
| --- | --- | --- |
| 数据集：标注和图片 | [tiu2mo/A2Seek](https://huggingface.co/datasets/tiu2mo/A2Seek) | `data/A2Seek/` |
| 最终 3B 权重 | [tiu2mo/A2Seek-R1-3B](https://huggingface.co/tiu2mo/A2Seek-R1-3B) | `models/A2Seek-R1-3B/` |
| 最终 7B 权重 | [tiu2mo/A2Seek-R1-7B](https://huggingface.co/tiu2mo/A2Seek-R1-7B) | `models/A2Seek-R1-7B/` |

## 1. 安装环境

如果尚未安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，先执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

创建环境并安装锁定的依赖：

```bash
uv python install 3.12
uv sync --locked --extra train --extra dataset
```

## 2. 下载数据

```bash
uv run --no-sync hf download tiu2mo/A2Seek \
  --repo-type dataset --local-dir data/A2Seek
uv run --no-sync python scripts/data/materialize_images.py
```

第二条命令将图片恢复到 `data/A2Seek/images/`。

固定的样本 ID 清单位于 `data/A2Seek/metadata/splits/`，校验值、样本数和场景代码记录在 `metadata/domain_protocol.json`：

| 设置 | ID 清单 | 样本数 | 场景 |
| --- | --- | ---: | --- |
| ID 训练 | `train.txt` / `id_train.txt` | 3,489 | 01、02、03、04、07、08 |
| ID 评估 | `validation.txt` / `id_eval.txt` | 2,327 | 01、02、03、04、07、08 |
| OOD 评估 | `test.txt` / `ood_eval.txt` | 2,376 | 00、05、06、09 |

清单保留已有标注的样本归属和顺序。训练与验证按样本互斥，共享 383 个视频。

`metadata/video_annotation_index.jsonl` 将 `video_group_001/video_1` 这样的编码视频 ID 对应到标注文件名、属性编码、帧目录、训练图片及原 DJI 文件名。`video_group` 与 `video_name` 的组合唯一标识视频；属性编码描述事件和环境，可由多个视频共享。

`metadata/dji_clip_mapping.jsonl` 提供全部 542 个广角视频的一一对应关系，经过帧数和画面内容核验。`metadata/dji_video_group_mapping.jsonl` 索引全部 1,616 个原始视频：`verified_clip` 表示对应标注的 W 视频，`paired_capture` 表示拍摄目录与 DJI 时间戳、序号相同的 Z/T/S 视频，`group_candidate` 保留其他组级视频记录。其他相机流使用各自的视角和帧时序。原始视频与图片数据包分开存放。

## 3. 下载权重

下载默认的 3B 模型：

```bash
uv run --no-sync hf download tiu2mo/A2Seek-R1-3B \
  --local-dir models/A2Seek-R1-3B
```

如果使用 7B，改为执行：

```bash
uv run --no-sync hf download tiu2mo/A2Seek-R1-7B \
  --local-dir models/A2Seek-R1-7B
```

使用 7B 推理时，加上 `--model models/A2Seek-R1-7B`。

## 4. 推理

```bash
uv run --no-sync python scripts/infer.py \
  --data data/A2Seek/dataset/cot_single_reflection/test.jsonl \
  --output outputs/test-predictions.jsonl \
  --gpus 0 --batch-size 1
```

首次检查可加 `--limit 8`，续跑可加 `--resume`。

answer 和 seeking 框使用输入图片的像素坐标 `[x1, y1, x2, y2]`，分别表示左上角和右下角。

运行未加载适配器的 Qwen zero-shot 基线：

```bash
uv run --no-sync python scripts/infer.py \
  --preset zero_shot_3b --output outputs/zero-shot-test.jsonl --gpus 0
uv run --no-sync python scripts/evaluate.py \
  --predictions outputs/zero-shot-test.jsonl \
  --references data/A2Seek/dataset/cot_single/test.jsonl \
  --label-aliases src/a2seek/configs/evaluation/label_aliases.json \
  --output outputs/zero-shot-test-metrics.json
```

7B 基础模型使用 `--preset zero_shot_7b`，本地基础权重可用 `--model /path/to/base-model` 指定。配置默认读取 `cot_single/test.jsonl`，采用贪心解码，最多生成 2,048 个 token，不加载 LoRA。ID 评估加上 `--data data/A2Seek/dataset/cot_single/validation.jsonl`。完整用户提示词位于 `src/a2seek/prompts/zero_shot.txt`，配置位于 `src/a2seek/configs/inference/`。评估时使用对应的 `cot_single` 标注文件。

## 5. 评估

```bash
uv run --no-sync python scripts/evaluate.py \
  --predictions outputs/test-predictions.jsonl \
  --references data/A2Seek/dataset/cot_single_reflection/test.jsonl \
  --label-aliases src/a2seek/configs/evaluation/label_aliases.json \
  --output outputs/test-metrics.json
```

`--references` 会核验完整 split 的样本覆盖和标注一致性；小规模检查时省略该参数。`--label-aliases` 统一预测与参考标注中的同义类别名。报告分别提供样本平均（`first_category_accuracy`、`mean_first_box_iou`）和场景平均（`scene_macro_first_category_accuracy`、`scene_macro_mean_first_box_iou`）。IoU 只统计具有非零参考框的样本，没有定位目标的场景不参与场景 IoU 平均。验证集与测试集应分别使用各自的标注文件评估。

## 6. 训练

### 准备训练输入

```bash
uv run --no-sync python scripts/data/prepare_local_data.py
```

### SFT

```bash
uv run --no-sync python scripts/train.py \
  --preset sft_3b --gpus 0 --output outputs/sft
```

SFT 从 `Qwen/Qwen2.5-VL-3B-Instruct` 开始训练。本地基础模型可用 `--model /path/to/base-model` 指定，训练 7B 时使用 `--preset sft_7b`。

### GRPO

将 `/path/to/sft/checkpoint` 替换为保存的 SFT checkpoint，合并后启动训练：

```bash
uv run --no-sync swift export \
  --adapters /path/to/sft/checkpoint \
  --merge_lora true --use_hf true --output_dir outputs/sft-merged
uv run --no-sync python scripts/train.py \
  --preset grpo --model outputs/sft-merged \
  --gpus 0 --output outputs/grpo
```

GRPO 训练后，用 `swift export` 合并其 adapter，再将合并目录传给 `scripts/infer.py --model`。

## 项目结构

```text
data/A2Seek/             数据集和恢复后的图片
models/
  A2Seek-R1-3B/           默认推理模型
  A2Seek-R1-7B/           可选的 7B 模型
outputs/                 推理结果、评估指标和训练 checkpoint
scripts/
  train.py               训练命令入口
  infer.py               推理命令入口
  evaluate.py            评估命令入口
  data/                  图片恢复、训练数据准备
src/a2seek/              核心实现和训练配置
docs/
  index.html             项目网页（GitHub Pages）
  static/                网页图片、视频、样式和脚本
```

保留下载仓库的完整目录结构。`data/`、`models/` 和 `outputs/` 均由 Git 忽略。

推理和数据准备使用其他存放位置时，按需用 `--dataset-root` 和 `--model` 指定，并同步修改推理命令的 `--data` 路径。

GitHub Pages 发布来源设为 `main` 分支的 `/docs` 目录。

## 引用与许可

如果 A2Seek 对你有帮助，欢迎给本仓库点个 ⭐ Star，或引用我们的论文。感谢你的支持！

```bibtex
@inproceedings{NEURIPS2025_de02de51,
 author = {Mo, Mengjingcheng and Tong, Xinyang and Tan, Mingpi and Leng, Jiaxu and Zheng, JianKang and Liu, Yiran and Chen, Haosheng and Gan, Ji and Li, Weisheng and Gao, Xinbo},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/085713-5057},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 publisher = {Curran Associates, Inc.},
 title = {A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/de02de513503962e1d21035ab50ce661-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {38, Main Conference},
 year = {2025}
}
```

主要依赖的许可：

- **Apache-2.0**：ms-swift、Transformers、PEFT、TRL、qwen-vl-utils、Datasets、PyArrow 和 huggingface_hub。
- **BSD-3-Clause**：PyTorch 和 torchvision。

模型权重和加载文件遵循发布目录中的 `LICENSE` 和 `NOTICE`。
