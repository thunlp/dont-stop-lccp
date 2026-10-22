# Don't Stop Long-Context Continual Pre-Training Too Early 🚀

📄 [Paper (arXiv:2610.29352)](https://arxiv.org/abs/2610.29352) | 🤗 [Hugging Face Collection](https://hf.co/collections/EdenQiao/dont-stop-lccp)

Official repository for the paper *Don't Stop Long-Context Continual Pre-Training Too Early: An Empirical Scaling Law*. This repository provides code for scaling law fitting, LongPPL/ShortPPL evaluation, and downstream benchmark evaluation. Model weights and evaluation datasets are hosted on Hugging Face and collected in the link above.

## 🔬 Model Weights

We release checkpoints from the S5 model (1.22B total parameters, N = 821M non-embedding parameters) along its 32K LCCP trajectory, as well as models trained from these checkpoints through subsequent post-training stages. All models use D<sub>s</sub> = 450N for short-text pre-training. These weights enable the community to analyze how long-context capabilities evolve as LCCP scales.

- Rows: D<sub>l</sub> training tokens for 32K LCCP (120N ≈ 100B tokens); the last three columns show models initialized from the corresponding 32K LCCP checkpoint in that row.
- The two tables correspond to the two learning rate strategies in the paper: **Constant-Last4** averages the most recent 4 consecutive checkpoints (1N apart); **Constant-Raw** uses the original checkpoint under a constant learning rate.
- Post-training token counts: 128K Extension 2.10B, Long-context SFT 2.10B, Agentic SFT 4.20B.

**Constant-Last4 (averaging the most recent 4 consecutive checkpoints)**

| D<sub>l</sub> | 32K LCCP | 128K Extension | Long-context SFT | Agentic SFT |
|---|---|---|---|---|
| 10N | [dont-stop-lccp-1.22b-32k-10n](https://huggingface.co/EdenQiao/dont-stop-lccp-1.22b-32k-10n) | Coming soon | Coming soon | Coming soon |
| 20N | Coming soon | Coming soon | Coming soon | Coming soon |
| 40N | [dont-stop-lccp-1.22b-32k-40n](https://huggingface.co/EdenQiao/dont-stop-lccp-1.22b-32k-40n) | Coming soon | Coming soon | Coming soon |
| 60N | Coming soon | Coming soon | Coming soon | Coming soon |
| 80N | [dont-stop-lccp-1.22b-32k-80n](https://huggingface.co/EdenQiao/dont-stop-lccp-1.22b-32k-80n) | Coming soon | Coming soon | Coming soon |
| 100N | Coming soon | Coming soon | Coming soon | Coming soon |
| 120N | [dont-stop-lccp-1.22b-32k-120n](https://huggingface.co/EdenQiao/dont-stop-lccp-1.22b-32k-120n) | Coming soon | Coming soon | Coming soon |

**Constant-Raw (original checkpoint under constant learning rate)**

| D<sub>l</sub> | 32K LCCP | 128K Extension | Long-context SFT | Agentic SFT |
|---|---|---|---|---|
| 10N | Coming soon | Coming soon | Coming soon | Coming soon |
| 20N | Coming soon | Coming soon | Coming soon | Coming soon |
| 40N | Coming soon | Coming soon | Coming soon | Coming soon |
| 60N | Coming soon | Coming soon | Coming soon | Coming soon |
| 80N | Coming soon | Coming soon | Coming soon | Coming soon |
| 100N | Coming soon | Coming soon | Coming soon | Coming soon |
| 120N | Coming soon | Coming soon | Coming soon | Coming soon |

<!-- After uploading, replace with: [model_name](https://huggingface.co/EdenQiao/xxx) -->

## 📊 Evaluation Datasets

| Dataset | Content | Purpose |
|---|---|---|
| [EdenQiao/dont-stop-lccp-longppl-32k-govreport](https://huggingface.co/datasets/EdenQiao/dont-stop-lccp-longppl-32k-govreport) | 10,000 GovReport documents (32K) with pre-computed long/short character span annotations | LongPPL / ShortPPL |
| [EdenQiao/dont-stop-lccp-ruler-32k-500](https://huggingface.co/datasets/EdenQiao/dont-stop-lccp-ruler-32k-500) | RULER 13 subtasks × 500 samples (32K) | RULER-32K |

```bash
huggingface-cli download EdenQiao/dont-stop-lccp-longppl-32k-govreport --repo-type dataset --local-dir data/govreport
huggingface-cli download EdenQiao/dont-stop-lccp-ruler-32k-500 --repo-type dataset --local-dir data/ruler
```

## ⚙️ Installation

Tested with Python 3.12.

```bash
pip install -r requirements.txt                 # Scaling law fitting
pip install -r requirements/inference.txt       # LongPPL evaluation
pip install -r requirements/benchmarks.txt      # Downstream benchmarks
```

Models use attention sink. When running with `--attention flash_attention_2`, a FlashAttention build supporting the `learnable_sink` parameter is required; otherwise the script will abort with an error.

## 📖 Usage

### 1. Scaling Law Fitting

$$\log(\mathrm{LongPPL})=\frac{A}{N^{\alpha}}+\frac{B_s}{D_s^{\beta_s}}+\frac{B_l}{D_l^{\beta_l}}+C$$

Input CSV must contain columns `scale,N,D_short,D_long,longppl`, where parameter counts and token counts use absolute values and `longppl` is not log-transformed. The training set must contain at least 4 distinct values along each token axis (with the other two variables fixed) for slice-based exponent estimation.

```bash
python -m lccp.scaling --data YOUR_DATA.csv \
  --train-scales S1 S2 S3 --holdout-scale S4 \
  --restarts 64 --output outputs/fit.json
```

The objective is Huber loss (f_scale = 0.05) in log space plus one-sided quadratic regularization on β<sub>s</sub> and β<sub>l</sub> (penalty strengths `--lambda-s` and `--lambda-l` default to 1.0). `--threshold-mode slice` uses the median from slice fits as the threshold; the default `fixed` mode uses a threshold of 0.2.

### 2. LongPPL / ShortPPL Evaluation

```bash
python -m lccp.longppl --model MODEL_DIR --data data/govreport \
  --trust-remote-code --device cuda --dtype bfloat16 \
  --attention flash_attention_2 --loss-backend minicpm-head \
  --max-length 32768 --output outputs/longppl.json
```

Span annotations are character intervals pre-computed based on the reference model Llama-3.1-8B. At evaluation time, they are mapped to the tested model's tokenizer, so loading the reference model is unnecessary. Output includes PPL, LongPPL, and ShortPPL. Documents with fewer than `--min-key-tokens` (default 10) selected tokens are excluded.

### 3. RULER-32K

```bash
RULER_ROOT=PATH_TO_RULER MODEL=MODEL_DIR RULER_DATA=data/ruler/32k \
  OUTPUT=outputs/ruler bash benchmarks/ruler/run.sh
```

Requires cloning [RULER](https://github.com/NVIDIA/RULER) first. `OUTPUT` must differ from `RULER_DATA`.

### 4. Long-Context Downstream Tasks (SFT Models)

LongBench, GraphWalks, LongReason, BABILong are evaluated via OpenCompass using non-thinking mode and greedy decoding.

```bash
export OPENCOMPASS_ROOT=PATH_TO_OPENCOMPASS
export LCCP_MODEL=MODEL_DIR
export LCCP_BENCHMARK=longreason          # longbench | graphwalks | longreason | babilong
export LONGREASON_DATA=PATH_TO_LONGREASON # Directory containing 32k.parquet
bash benchmarks/opencompass/run.sh
```

| `LCCP_BENCHMARK` | Data Path |
|---|---|
| `longbench` | Uses OpenCompass built-in configuration |
| `graphwalks` | `GRAPHWALKS_DATA`: directory containing `graphwalks_128k_and_shorter.parquet` |
| `longreason` | `LONGREASON_DATA`: directory containing `32k.parquet` |
| `babilong` | `BABILONG_DATA`: local dataset containing the 32K split |

Equal-weight average of the four scores (0–100): `python benchmarks/aggregate.py SCORES.json`, where JSON keys are `longbench`, `graphwalks`, `longreason`, `babilong`.

## 📝 Citation

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
