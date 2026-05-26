# TruFor

This folder contains the minimal scripts used to run TruFor as an image-forensics safeguard in the defense experiments.

We do not vendor the full TruFor repository or model weights here. The scripts assume that TruFor has already been packaged as a Docker image named `trufor:server`.

## Input layout

The default workspace is resolved from the repository root:

```text
dataset/trufor_workspace
```

Expected files and folders:

```text
dataset/trufor_workspace/
  sample_180.json
  poison/
  clip_white_data_2/
  siglip_white_data_2/
  trufor_outputs/
  trufor_logs/
```

Each image folder contains 180 poisoned images. `sample_180.json` stores the metadata, including the edit category under `counterfactual_edit.category`.

## Run TruFor

```bash
bash run_imgs_180_trufor.sh
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
docker run --rm --entrypoint python \
  -v dataset/trufor_workspace:/work \
  trufor:server /work/summarize_imgs_180_trufor.py
```
