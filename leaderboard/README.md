# LongMemEval-V2 Leaderboard Submissions

This folder contains utilities for preparing LongMemEval-V2 leaderboard
packages. A leaderboard submission represents one memory method at one tier
(`small` or `medium`) and may contain multiple latency operating points.

Leaderboard entries are submitted through a Google form. Please do not submit
leaderboard entries as GitHub issues; informal submission issues will be
closed or deleted.

## What Gets Scored

Each operating point is evaluated from a pair of completed runs:

- one `web` run
- one `enterprise` run

The helper combines the two domains with example-count-weighted averages and
extracts:

- `overall_full_set`
- `gotchas_accuracy`
- `static_accuracy`
- `dynamic_accuracy`
- `procedure_accuracy`
- `memory_query_avg_seconds`

The final package computes LAFS gain against the fixed reference frontier for
the selected tier. LAFS uses:

- accuracy: `overall_full_set * 100`
- latency: `memory_query_avg_seconds`

## Step 1: Build One Operating Point

Run step 1 once for each latency operating point:

```bash
python leaderboard/build_submission_step_1_single_operating_point.py \
  runs/agentrunbook_c_web_small \
  runs/agentrunbook_c_enterprise_small \
  submission_1 \
  fast \
  small
```

Arguments:

- `web_output_dir`: completed web run folder.
- `enterprise_output_dir`: completed enterprise run folder.
- `submission_name`: final submission folder name.
- `operating_point_name`: name for this latency operating point. Use a short
  filesystem-safe label such as `fast`, `balanced`, or `accurate`.
- `tier`: `small` or `medium`.

If your run folder names include latency labels, pass `--method <method_name>`
so all operating points record the same method:

```bash
python leaderboard/build_submission_step_1_single_operating_point.py \
  runs/my_method_fast_web_small \
  runs/my_method_fast_enterprise_small \
  submission_1 \
  fast \
  small \
  --method my_method
```

Step 1 writes:

```text
leaderboard/submissions/submission_1/
  operating_points/
    fast/
      metric_overview.json
      operating_point_metadata.json
      web/
        aggregated_metrics.json
        per_question.jsonl
        run_args.json
        runtime_inputs/
      enterprise/
        aggregated_metrics.json
        per_question.jsonl
        run_args.json
        runtime_inputs/
```

Step 1 validates:

- both run folders are complete
- domains are `web` and `enterprise`
- `per_question.jsonl` covers every question in `runtime_inputs/questions.json`
- question ids are unique
- question-type counts match runtime inputs
- `aggregated_metrics.json` question counts match the logs
- reader model contains `qwen3.5-9b`
- judge model contains `gpt-5.2`
- both domains use the same method and tier

Use `--force` to rebuild an existing operating point folder. If a final tarball
already exists for the submission, `--force` removes it because it would become
stale.

## Step 2: Build the Final Package

After creating all operating points, build the final package:

```bash
python leaderboard/build_submission_step_2_build_package.py \
  submission_1 \
  SYSTEM_DESCRIPTION.md \
  path/to/code_file.py \
  leaderboard/submissions/submission_1/operating_points/fast \
  leaderboard/submissions/submission_1/operating_points/balanced \
  leaderboard/submissions/submission_1/operating_points/accurate
```

Arguments:

- `submission_name`: must match the step-1 submission name.
- `system_description_path`: path to `SYSTEM_DESCRIPTION.md`.
- `code_file_path`: path to one code artifact file. Directories are rejected.
- `operating_point_dirs`: one or more step-1 operating point folders.

Step 2 writes:

```text
leaderboard/submissions/submission_1/
  SYSTEM_DESCRIPTION.md
  code_file.py
  submission_overview.json
  operating_points/
    fast/
    balanced/
    accurate/

leaderboard/submissions/submission_1.tar.gz
```

Step 2 validates that all operating points use:

- the same method
- the same tier
- a supported tier: `small` or `medium`
- the same web question ids
- the same enterprise question ids
- the same web haystack
- the same enterprise haystack

The top-level `submission_overview.json` records the method, tier, operating
point accuracy/latency values, the LAFS summary, and paths to each operating
point metric overview.

Use `--force` to replace existing root package files, remove stale operating
point folders not passed to step 2, and rebuild the tarball.

## Package Checklist

Before submitting, inspect the final folder:

```bash
tar -tzf leaderboard/submissions/submission_1.tar.gz | head
python -m json.tool leaderboard/submissions/submission_1/submission_overview.json
```

The package should contain one `operating_points/<name>/` folder per latency
operating point, plus `SYSTEM_DESCRIPTION.md`, the code file, and
`submission_overview.json` at the root.

## Troubleshooting

- `Missing aggregated_metrics.json`: the run has not finished evaluation.
- `model must contain 'qwen3.5-9b'`: rerun with the fixed reader model expected
  by the leaderboard.
- `evaluator_model must contain 'gpt-5.2'`: rerun with the fixed judge model.
- `different question ids` or `different haystack`: rebuild the operating
  points from the same tier and data split.
- `Operating points use different methods`: pass the same `--method` value to
  step 1 for all operating points.
