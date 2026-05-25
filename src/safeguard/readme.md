# TruFor safeguard

This folder contains the minimal scripts used to run TruFor as an image-forensics safeguard in our experiments.

We do not vendor the full TruFor repository or model weights here. The scripts assume that TruFor has already been packaged as a Docker image named `trufor:server`.

## Input layout

The server-side input directory used in our experiment was:

```text
/home/ubuntu/liangrj/imgs_180
```

Expected files and folders:

```text
imgs_180/
  sample_180.json
  poison/
  clip_white_data_2/
  siglip_white_data_2/
```

Each image folder contains 180 poisoned images. `sample_180.json` contains the metadata, including the edit category under:

```text
counterfactual_edit.category
```

## Run TruFor

Run all three groups with TruFor in CPU mode:

```bash
bash run_imgs_180_trufor.sh
```

The script runs:

```bash
docker run --rm \
  -v "$in_dir:/data:ro" \
  -v "$out_dir:/data_out" \
  trufor:server -gpu -1 -in /data -out /data_out
```

Outputs are written to:

```text
/home/ubuntu/liangrj/imgs_180/trufor_outputs/<group>/*.npz
/home/ubuntu/liangrj/imgs_180/trufor_logs/<group>.log
/home/ubuntu/liangrj/imgs_180/trufor_logs/<group>.status
```

Each `.npz` contains TruFor's standard outputs, including:

```text
map    anomaly localization map
conf   confidence map, if produced
score  image-level forgery score in [0, 1]
imgsize input image size
```

We use `score > 0.5` as the binary detection rule.

## Summarize Results

The summarizer needs `numpy`, so we run it inside the same Docker image:

```bash
docker run --rm --entrypoint python \
  -v /home/ubuntu/liangrj/imgs_180:/work \
  trufor:server /work/summarize_imgs_180_trufor.py
```

Summary files are written to:

```text
/home/ubuntu/liangrj/imgs_180/trufor_results/trufor_group_summary.csv
/home/ubuntu/liangrj/imgs_180/trufor_results/trufor_category_summary.csv
/home/ubuntu/liangrj/imgs_180/trufor_results/trufor_all_scores.csv
```

`trufor_group_summary.csv` reports detection counts and score statistics for each group.

`trufor_category_summary.csv` reports the same metrics grouped by `counterfactual_edit.category`.

`trufor_all_scores.csv` stores per-image scores and localization-map statistics.

## Notes

TruFor's `score` is an image-level forensic manipulation score, not a semantic poisoning score. Higher values indicate stronger image-forensics evidence of manipulation. In our experiments, low TruFor scores indicate that the poisoned images do not introduce obvious low-level forensic traces detectable by TruFor.
