# Don't Stop Long-Context Continual Pre-Training Too Early 🚀

📄 [论文 (arXiv:2610.29352)](https://arxiv.org/abs/2610.29352) | 🤗 [Hugging Face Collection](https://hf.co/collections/EdenQiao/dont-stop-lccp)

论文 *Don't Stop Long-Context Continual Pre-Training Too Early: An Empirical Scaling Law* 的官方仓库。本仓库提供 scaling law 拟合代码、LongPPL/ShortPPL 评测代码和下游 benchmark 评测脚本；模型权重与评测数据发布在 Hugging Face，统一汇总于上面的 Collection。

## 🔬 模型权重

我们开源论文中 S5 模型（总参数 1.22B，非 embedding 参数 N = 821M）沿 32K LCCP 轨迹的 checkpoint，以及从这些 checkpoint 出发进行后续训练得到的模型。所有模型的短文本预训练量均为 D<sub>s</sub> = 450N。希望这些权重能帮助社区分析 LCCP 规模扩大过程中模型长文本能力的变化。

- 行：32K LCCP 的训练 token 数 D<sub>l</sub>（120N ≈ 100B tokens）；后三列的模型均从该行对应的 32K LCCP checkpoint 出发训练。
- 两张表分别对应论文中的两种学习率策略：**Constant-Last4** 对最近 4 个连续 checkpoint（间隔 1N）做参数平均；**Constant-Raw** 是恒定学习率下的原始 checkpoint。
- 后训练 token 数：128K Extension 2.10B，Long-context SFT 2.10B，Agentic SFT 4.20B。

**Constant-Last4（对最近 4 个连续 checkpoint 做参数平均）**

| D<sub>l</sub> | 32K LCCP | 128K Extension | Long-context SFT | Agentic SFT |
|---|---|---|---|---|
| 10N | Coming soon | Coming soon | Coming soon | Coming soon |
| 20N | Coming soon | Coming soon | Coming soon | Coming soon |
| 40N | Coming soon | Coming soon | Coming soon | Coming soon |
| 60N | Coming soon | Coming soon | Coming soon | Coming soon |
| 80N | Coming soon | Coming soon | Coming soon | Coming soon |
| 100N | Coming soon | Coming soon | Coming soon | Coming soon |
| 120N | Coming soon | Coming soon | Coming soon | Coming soon |

**Constant-Raw（恒定学习率下的原始 checkpoint）**

| D<sub>l</sub> | 32K LCCP | 128K Extension | Long-context SFT | Agentic SFT |
|---|---|---|---|---|
| 10N | Coming soon | Coming soon | Coming soon | Coming soon |
| 20N | Coming soon | Coming soon | Coming soon | Coming soon |
| 40N | Coming soon | Coming soon | Coming soon | Coming soon |
| 60N | Coming soon | Coming soon | Coming soon | Coming soon |
| 80N | Coming soon | Coming soon | Coming soon | Coming soon |
| 100N | Coming soon | Coming soon | Coming soon | Coming soon |
| 120N | Coming soon | Coming soon | Coming soon | Coming soon |

<!-- 上传后替换为：[模型名](https://huggingface.co/EdenQiao/xxx) -->

## 📊 评测数据

| 数据集 | 内容 | 用途 |
|---|---|---|
| [EdenQiao/dont-stop-lccp-longppl-32k-govreport](https://huggingface.co/datasets/EdenQiao/dont-stop-lccp-longppl-32k-govreport) | 10,000 篇 GovReport 文档（32K）及预先计算好的 long/short 字符 span 标注 | LongPPL / ShortPPL |
| [EdenQiao/dont-stop-lccp-ruler-32k-500](https://huggingface.co/datasets/EdenQiao/dont-stop-lccp-ruler-32k-500) | RULER 13 个子任务 × 500 条（32K） | RULER-32K |

```bash
huggingface-cli download EdenQiao/dont-stop-lccp-longppl-32k-govreport --repo-type dataset --local-dir data/govreport
huggingface-cli download EdenQiao/dont-stop-lccp-ruler-32k-500 --repo-type dataset --local-dir data/ruler
```

## ⚙️ 安装

测试环境为 Python 3.12。

```bash
pip install -r requirements.txt                 # scaling law 拟合
pip install -r requirements/inference.txt       # LongPPL 评测
pip install -r requirements/benchmarks.txt      # 下游 benchmark
```

模型使用 attention sink。以 `--attention flash_attention_2` 运行时，需要支持 `learnable_sink` 参数的 FlashAttention 版本，否则脚本会直接报错退出。

## 📖 使用

### 1. Scaling law 拟合

$$\log(\mathrm{LongPPL})=\frac{A}{N^{\alpha}}+\frac{B_s}{D_s^{\beta_s}}+\frac{B_l}{D_l^{\beta_l}}+C$$

输入 CSV 包含列 `scale,N,D_short,D_long,longppl`，其中参数量和 token 数用绝对值，`longppl` 不取对数。训练集中每个 token 轴需至少有 4 个不同取值（另两个变量固定），用于切片估计指数。

```bash
python -m lccp.scaling --data YOUR_DATA.csv \
  --train-scales S1 S2 S3 --holdout-scale S4 \
  --restarts 64 --output outputs/fit.json
```

目标函数为 log 空间的 Huber loss（f_scale = 0.05）加上 β<sub>s</sub>、β<sub>l</sub> 的单边二次正则（强度 `--lambda-s` 和 `--lambda-l` 默认 1.0）。`--threshold-mode slice` 使用切片拟合的中位数作为阈值；默认 `fixed` 模式下阈值为 0.2。

### 2. LongPPL / ShortPPL 评测

```bash
python -m lccp.longppl --model MODEL_DIR --data data/govreport \
  --trust-remote-code --device cuda --dtype bfloat16 \
  --attention flash_attention_2 --loss-backend minicpm-head \
  --max-length 32768 --output outputs/longppl.json
```

span 标注是基于参考模型 Llama-3.1-8B 预先计算的字符区间，评测时会映射到被测模型的 tokenizer 上，因此不需要加载参考模型。输出包含 PPL、LongPPL 和 ShortPPL。选中 token 少于 `--min-key-tokens`（默认 10）的文档不计入。

### 3. RULER-32K

```bash
RULER_ROOT=PATH_TO_RULER MODEL=MODEL_DIR RULER_DATA=data/ruler/32k \
  OUTPUT=outputs/ruler bash benchmarks/ruler/run.sh
```

需要先 clone [RULER](https://github.com/NVIDIA/RULER)。`OUTPUT` 不能与 `RULER_DATA` 相同。

### 4. 长文本下游任务（SFT 模型）

LongBench、GraphWalks、LongReason、BABILong 通过 OpenCompass 评测，使用 non-thinking 模式和 greedy decoding。

```bash
export OPENCOMPASS_ROOT=PATH_TO_OPENCOMPASS
export LCCP_MODEL=MODEL_DIR
export LCCP_BENCHMARK=longreason          # longbench | graphwalks | longreason | babilong
export LONGREASON_DATA=PATH_TO_LONGREASON # 目录下含 32k.parquet
bash benchmarks/opencompass/run.sh
```

| `LCCP_BENCHMARK` | 数据路径 |
|---|---|
| `longbench` | 使用 OpenCompass 自带配置 |
| `graphwalks` | `GRAPHWALKS_DATA`：含 `graphwalks_128k_and_shorter.parquet` 的目录 |
| `longreason` | `LONGREASON_DATA`：含 `32k.parquet` 的目录 |
| `babilong` | `BABILONG_DATA`：含 32K split 的本地数据集 |

四项分数（0–100）的等权平均：`python benchmarks/aggregate.py SCORES.json`，JSON 的键为 `longbench`、`graphwalks`、`longreason`、`babilong`。

## 📝 引用

```bibtex
@misc{qiao2026dontstoplongcontextcontinual,
      title={Don't Stop Long-Context Continual Pre-Training Too Early: An Empirical Scaling Law},
      author={Ziqing Qiao and Xiaoyue Xu and Sikui Zhang and Yinuo Xu and Bingxiang He and Chaojun Xiao and Xu Han},
      year={2026},
      eprint={2610.29352},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.29352},
}
```
