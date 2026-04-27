# CloudWorker Dataset Prep

This directory contains scripts and metadata for converting the existing
CloudWorker COCO annotations into pair-level relation samples for
`HOI backbone + simple classifier head` training.

## Task definition

We are not training standard HICO-style HOI detection here. Instead, each
training sample is a worker-scaffolding pair with a behavior label:

- `climbing`
- `leaning_out`
- `step_on_railing`
- `throwing_material`
- `holding_material`

The supervision unit is:

`(image, worker instance, scaffolding instance) -> behavior label`

## Source annotations

The source dataset is expected to use COCO-style annotations with:

- worker behavior categories such as `worker with helmet_climbing`
- scaffolding categories such as `scaffolding` / `Scaffolding`

## Output format

The conversion script emits one JSON file per split. Each file is a list of
image-level samples. Every image entry contains:

- image metadata
- worker instances
- scaffolding instances
- worker-scaffolding relation pairs

Each pair record contains:

- `human_box`
- `object_box`
- `human_mask_annotation_id`
- `object_mask_annotation_id`
- `label`
- `label_name`
- `pair_source`

## Pairing rule

For every worker instance, the converter selects one scaffolding instance in
the same image. The default strategy is:

- choose the scaffolding instance with the nearest box center

If there is no scaffolding instance in the image, the worker instance is
skipped and the image is recorded with zero valid pairs.

## Example

```powershell
python upt/cloudworker/prepare_cloudworker.py `
  --data-root E:\MasterDegree\Experiment\scene-behavior\data\cloudworker.v11i.coco `
  --output-dir E:\MasterDegree\Experiment\scene-behavior\data\cloudworker.v11i.coco\prepared
```

## Next step

After conversion, the training dataloader should read the prepared pair-level
JSON files instead of the original raw COCO annotations.

