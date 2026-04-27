import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BEHAVIOR_CATEGORY_NAME_TO_LABEL = {
    "worker with helmet_climbing": 0,
    "worker with helmet_leaning out": 1,
    "worker with helmet_step on railing": 2,
    "worker with helmet_throwing_material": 3,
    "worker_with_helmet_holding_material": 4,
}

LABEL_TO_BEHAVIOR_NAME = {
    0: "climbing",
    1: "leaning_out",
    2: "step_on_railing",
    3: "throwing_material",
    4: "holding_material",
}

SCAFFOLDING_CATEGORY_NAMES = {
    "scaffolding",
    "scaffolding".lower(),
}


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def bbox_xywh_to_xyxy(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def bbox_center_xyxy(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def squared_center_distance(box_a: List[float], box_b: List[float]) -> float:
    ax, ay = bbox_center_xyxy(box_a)
    bx, by = bbox_center_xyxy(box_b)
    return (ax - bx) ** 2 + (ay - by) ** 2


def load_coco_annotations(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_maps(categories: List[Dict]) -> Tuple[Dict[int, int], set]:
    behavior_category_id_to_label: Dict[int, int] = {}
    scaffolding_category_ids = set()

    for category in categories:
        category_id = int(category["id"])
        raw_name = category["name"]
        normalized = normalize_name(raw_name)

        for behavior_name, label in BEHAVIOR_CATEGORY_NAME_TO_LABEL.items():
            if normalized == normalize_name(behavior_name):
                behavior_category_id_to_label[category_id] = label
                break

        if normalized == "scaffolding":
            scaffolding_category_ids.add(category_id)

    return behavior_category_id_to_label, scaffolding_category_ids


def choose_scaffolding(
    worker_box: List[float],
    scaffolds: List[Dict]
) -> Optional[Dict]:
    if not scaffolds:
        return None
    return min(
        scaffolds,
        key=lambda scaffold: squared_center_distance(worker_box, scaffold["bbox_xyxy"])
    )


def convert_split(annotation_path: Path) -> Dict:
    coco = load_coco_annotations(annotation_path)
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    behavior_category_id_to_label, scaffolding_category_ids = build_category_maps(categories)

    images_by_id = {int(image["id"]): image for image in images}
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[int(ann["image_id"])].append(ann)

    converted_images = []
    total_pairs = 0
    skipped_workers_without_scaffold = 0
    skipped_unknown_categories = 0

    for image_id, image in images_by_id.items():
        worker_instances = []
        scaffolding_instances = []

        for ann in annotations_by_image.get(image_id, []):
            ann_id = int(ann["id"])
            category_id = int(ann["category_id"])
            bbox_xyxy = bbox_xywh_to_xyxy(ann["bbox"])

            if category_id in scaffolding_category_ids:
                scaffolding_instances.append({
                    "annotation_id": ann_id,
                    "category_id": category_id,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_xywh": ann["bbox"],
                    "segmentation": ann.get("segmentation"),
                    "area": ann.get("area"),
                    "iscrowd": ann.get("iscrowd", 0),
                })
            elif category_id in behavior_category_id_to_label:
                label = behavior_category_id_to_label[category_id]
                worker_instances.append({
                    "annotation_id": ann_id,
                    "category_id": category_id,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_xywh": ann["bbox"],
                    "segmentation": ann.get("segmentation"),
                    "area": ann.get("area"),
                    "iscrowd": ann.get("iscrowd", 0),
                    "label": label,
                    "label_name": LABEL_TO_BEHAVIOR_NAME[label],
                })
            else:
                skipped_unknown_categories += 1

        pairs = []
        for worker in worker_instances:
            scaffold = choose_scaffolding(worker["bbox_xyxy"], scaffolding_instances)
            if scaffold is None:
                skipped_workers_without_scaffold += 1
                continue

            pairs.append({
                "human_box": worker["bbox_xyxy"],
                "object_box": scaffold["bbox_xyxy"],
                "human_mask_annotation_id": worker["annotation_id"],
                "object_mask_annotation_id": scaffold["annotation_id"],
                "label": worker["label"],
                "label_name": worker["label_name"],
                "pair_source": "nearest_scaffolding_center",
            })

        total_pairs += len(pairs)
        converted_images.append({
            "image_id": image_id,
            "file_name": image["file_name"],
            "width": int(image["width"]),
            "height": int(image["height"]),
            "workers": worker_instances,
            "scaffolds": scaffolding_instances,
            "pairs": pairs,
        })

    return {
        "meta": {
            "source_annotation": str(annotation_path),
            "behavior_classes": LABEL_TO_BEHAVIOR_NAME,
            "num_images": len(converted_images),
            "num_pairs": total_pairs,
            "skipped_workers_without_scaffold": skipped_workers_without_scaffold,
            "skipped_unknown_categories": skipped_unknown_categories,
        },
        "images": converted_images,
    }


def infer_annotation_path(data_root: Path, split: str) -> Path:
    return data_root / split / "_annotations.coco.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to convert."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        annotation_path = infer_annotation_path(args.data_root, split)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

        converted = convert_split(annotation_path)
        output_path = args.output_dir / f"{split}_pair_annotations.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False)

        meta = converted["meta"]
        print(
            f"[{split}] images={meta['num_images']} "
            f"pairs={meta['num_pairs']} "
            f"skipped_no_scaffold={meta['skipped_workers_without_scaffold']} "
            f"skipped_unknown={meta['skipped_unknown_categories']}"
        )
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
