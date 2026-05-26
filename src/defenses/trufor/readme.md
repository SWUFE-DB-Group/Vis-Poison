# TruFor

This folder contains the scripts used to run TruFor as an image-forensics defense baseline.

## Files

- `run_imgs_180_trufor.sh`: run TruFor on the prepared image groups
- `summarize_imgs_180_trufor.py`: summarize `.npz` outputs into CSV tables

We do not vendor the TruFor repository or model weights here. The scripts assume
that TruFor is already available as a Docker image named `trufor:server`.

## Input Layout

The default workspace is:

```text
dataset/trufor_workspace
```

Expected layout:

```text
dataset/trufor_workspace/
  sample_180.json
  poison/
  trufor_outputs/
  trufor_logs/
```

`sample_180.json` stores the sample metadata, including
`counterfactual_edit.category`.

## Run TruFor

```bash
bash src/defenses/trufor/run_imgs_180_trufor.sh
```

Outputs are written to:

```text
dataset/trufor_workspace/trufor_outputs/<group>/*.npz
dataset/trufor_workspace/trufor_logs/<group>.log
dataset/trufor_workspace/trufor_logs/<group>.status
```

We use `score > 0.5` as the binary detection rule.

## Summarize Results

```bash
uv run src/defenses/trufor/summarize_imgs_180_trufor.py
```

Summary files are written to:

```text
dataset/trufor_workspace/trufor_results/trufor_all_scores.csv
dataset/trufor_workspace/trufor_results/trufor_group_summary.csv
dataset/trufor_workspace/trufor_results/trufor_category_summary.csv
```
