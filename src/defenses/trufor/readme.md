# TruFor

This folder contains the minimal scripts used to run TruFor as an image-forensics safeguard in the defense experiments.

We do not vendor the full TruFor repository or model weights here. The scripts assume that TruFor has already been packaged as a Docker image named `trufor:server`.

## Input layout

Expected input directory:

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

Each image folder contains 180 poisoned images. `sample_180.json` stores the metadata, including the edit category under `counterfactual_edit.category`.

## Run TruFor

```bash
bash run_imgs_180_trufor.sh
```

Outputs are written to:

```text
/home/ubuntu/liangrj/imgs_180/trufor_outputs/<group>/*.npz
/home/ubuntu/liangrj/imgs_180/trufor_logs/<group>.log
/home/ubuntu/liangrj/imgs_180/trufor_logs/<group>.status
```

We use `score > 0.5` as the binary detection rule.

## Summarize Results

```bash
docker run --rm --entrypoint python \
  -v /home/ubuntu/liangrj/imgs_180:/work \
  trufor:server /work/summarize_imgs_180_trufor.py
```
