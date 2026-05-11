# LongMemEval-V2

**LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues**

[Di Wu](https://xiaowu0162.github.io/),
[Zixiang Ji](https://www.linkedin.com/in/zixiang-ji-56902624b/),
[Asmi Kawatkar](https://www.linkedin.com/in/asmi-kawatkar),
[Bryan Kwan](https://www.linkedin.com/in/kwan-bryan),
[Jia-Chen Gu](https://jasonforjoy.github.io/index.html),
[Nanyun Peng](https://vnpeng.net/), and
[Kai-Wei Chang](https://kwchang.net/)

[Website](https://xiaowu0162.github.io/longmemeval-v2/) |
[Paper (temporary)](https://github.com/xiaowu0162/LongMemEval) |
[Data](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2) |
[Leaderboard](https://xiaowu0162.github.io/longmemeval-v2/#leaderboard)

This is the official LongMemEval-V2 repository. It contains the public
evaluation harness, data preparation tools, leaderboard packaging utilities,
and the memory baselines reported with the benchmark.

## Overview

LongMemEval-V2 evaluates whether memory systems can help agents acquire the
experience needed to become knowledgeable colleagues in customized
environments. The benchmark pairs manually curated questions with long
histories of multimodal web-agent trajectories. A memory system consumes the
trajectory history and returns compact evidence for downstream question
answering; evaluation targets both answer accuracy and query latency.

LongMemEval-V2 contains:

- 451 manually curated questions.
- 5 memory abilities.
- Up to 500 trajectories per haystack.
- Up to 115M tokens in the largest haystacks.
- Two domains: web and enterprise.
- Two public leaderboard tiers: small and medium.

The benchmark tests five core memory abilities:

- **Static state recall**: remembers important landmarks, page layouts, module
  affordances, and subtle state differences.
- **Dynamic state tracking**: understands how states and actions change the
  environment over time.
- **Workflow knowledge**: knows the steps needed to complete recurring tasks in
  customized environments.
- **Environment gotchas**: recognizes recurring local failure modes and avoids
  environment-specific traps.
- **Premise awareness**: detects assumptions that are valid elsewhere but wrong
  in the current deployment.

## Repository Layout

```text
data/                 download, preparation, and validation scripts
evaluation/           evaluation runner, scoring code, configs, and shell wrappers
leaderboard/          metric merging, LAFS scoring, and submission packaging
memory_modules/       memory backend implementations
```

The repository includes the following evaluation methods:

- `no_retrieval`: no memory context.
- `rag_query_to_slice`: RAG query to raw state slices.
- `rag_query_to_slice_notes`: RAG query to raw state slices plus trajectory
  notes.
- `agentrunbook_r`: AgentRunbook-R.
- `codex`: vanilla Codex coding-agent memory baseline.
- `agentrunbook_c`: AgentRunbook-C.

Benchmark curation scripts, annotation tools, ablations, and exploratory
experiments are not part of this public release.

## Setup

LongMemEval-V2 uses Python 3.11. The default conda environment installs
PyTorch through `requirements-torch.txt`. For CUDA 12.4 machines, the torch
install command is:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Create the environment and install the package:

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate lme-v2-release
pip install -e .
```

Researchers using a different CUDA or CPU setup should install the appropriate
PyTorch build first, either with a direct `pip install` command or by editing
`requirements-torch.txt` before creating the environment.

The environment does not include vLLM. Start or forward your own
OpenAI-compatible model servers, then point the scripts to them. The paper runs
use Qwen3.5-9B as the fixed reader and Qwen3-Embedding-8B for embedding-based
methods. For `codex` and `agentrunbook_c`, install Codex v0.117.0 separately
and set `CODEX_BINARY`.

## Data

Download the released dataset from Hugging Face and prepare the runtime
screenshots layout:

```bash
python data/download_data.py --data-root data/longmemeval-v2
export DATA_ROOT="$(pwd)/data/longmemeval-v2"
python data/prepare_data.py --data-root "$DATA_ROOT" --mode symlink
python data/validate_data.py --data-root "$DATA_ROOT" --tier small
```

The default dataset repository is
`xiaowu0162/longmemeval-v2`. Screenshot bundles are stored as `.tar.gz`
archives under `trajectory_screenshots/`; `prepare_data.py` extracts them when
needed and links the resulting directories into:

```text
screenshots/<trajectory_id>/<step>.png
```

## Model Endpoints

Example endpoint settings:

```bash
export READER_BASE_URL=http://localhost:8023/v1
export READER_MODEL=Qwen/Qwen3.5-9B
export LME_CONTROLLER_BASE_URL=http://localhost:8023/v1
export LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_BASE_URL=http://localhost:8114/v1
export LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

For LLM-judged abstention and gotchas questions:

```bash
export OPENAI_API_KEY=...
```

The default judge is `gpt-5.2` with medium reasoning effort.

For Codex-based runs:

```bash
export CODEX_BINARY=/path/to/codex-binary
export CODEX_MODEL=gpt-5.4-mini
export CODEX_REASONING_EFFORT=xhigh
```

Codex also expects common command-line tools such as `rg` and `find`.

## Quick Smoke Test

Reader-only smoke run:

```bash
python evaluation/run_eval.py \
  --data-root "$DATA_ROOT" \
  --domain web \
  --tier small \
  --method no_retrieval \
  --limit 1 \
  --output-dir runs/smoke_no_retrieval_web_small
```

RAG smoke run:

```bash
python evaluation/run_eval.py \
  --data-root "$DATA_ROOT" \
  --domain web \
  --tier small \
  --method rag_query_to_slice \
  --limit 1 \
  --output-dir runs/smoke_rag_web_small
```

Expected outputs include:

```text
run_args.json
prompt_rows.jsonl
per_question.jsonl
aggregated_metrics.json
runtime_inputs/
memory_workspace/
```

## Reproducing Baselines

Each shell script accepts extra argparse flags after the environment variables:

```bash
export DATA_ROOT=/path/to/longmemeval-v2
export OUTPUT_ROOT=runs
export DOMAIN=web
export TIER=small

evaluation/scripts/run_no_retrieval.sh
evaluation/scripts/run_rag_query_to_slice.sh
evaluation/scripts/run_rag_query_to_slice_notes.sh
evaluation/scripts/run_agentrunbook_r.sh
evaluation/scripts/run_codex.sh
evaluation/scripts/run_agentrunbook_c.sh
```

Change `DOMAIN=enterprise` and `TIER=medium` to run the other paper settings.
The direct argparse form is:

```bash
python evaluation/run_eval.py \
  --data-root "$DATA_ROOT" \
  --domain web \
  --tier small \
  --method agentrunbook_r \
  --output-dir runs/agentrunbook_r_web_small
```

## Metrics

Each run writes `aggregated_metrics.json`. Category results are reported
separately for regular and abstention questions, and
`combined_abstention_by_category` reports the paper-facing average of
`(static, static-abs)`, `(dynamic, dynamic-abs)`, and
`(procedure, procedure-abs)`.

To combine matching enterprise and web runs for the same method and tier:

```bash
python leaderboard/combine_aggregated_metrics.py \
  runs/agentrunbook_r_enterprise_small/aggregated_metrics.json \
  runs/agentrunbook_r_web_small/aggregated_metrics.json \
  -o runs/agentrunbook_r_small_combined_metrics.json
```

The combined file uses example-count-weighted scores, token means, and mean
timing. Timing percentiles are not merged from aggregate-only files.

## Leaderboard Submissions

Leaderboard entries measure how much a memory system improves the released
baseline + AgentRunbook accuracy-latency frontier. The score is LAFS gain over
the fixed reference frontier, and a submission may include multiple latency
operating points for the same method and tier.

See [leaderboard/README.md](leaderboard/README.md) for the full packaging
instructions.

Submissions will be collected through a Google form. Please do not submit
leaderboard entries as GitHub issues; informal submission issues will be
closed or deleted.

## Troubleshooting

- Missing trajectory screenshots: run
  `python data/prepare_data.py --data-root "$DATA_ROOT" --mode symlink`.
- Endpoint connection failure: check `READER_BASE_URL`,
  `LME_CONTROLLER_BASE_URL`, and `LME_EMBEDDING_BASE_URL`.
- `top_k` rejected by the server: use a vLLM-compatible OpenAI endpoint or set
  `--reader-top-k` / `--controller-top-k` according to your server support.
- Codex binary not found: set `CODEX_BINARY` to the Codex v0.117.0 executable.
- Existing output directory errors: use a new `--output-dir`; the harness
  refuses to overwrite memory workspaces.

## Citation

Please use the following placeholder citation until the final preprint metadata
is available:

```bibtex
@article{wu2026longmemevalv2,
  title = {LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues},
  author = {Di Wu and Zixiang Ji and Asmi Kawatkar and Bryan Kwan and Jia-Chen Gu and Nanyun Peng and Kai-Wei Chang},
  year = {2026},
  note = {Preprint forthcoming},
  url = {https://xiaowu0162.github.io/longmemeval-v2/}
}
```
