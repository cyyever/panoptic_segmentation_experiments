# dataset settings

import json
import os

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco_panoptic import CocoPanopticDataset


@DATASETS.register_module()
class MaplliaryPanopticDataset(CocoPanopticDataset):
    CLASSES: list = []
    THING_CLASSES: list = []
    STUFF_CLASSES: list = []


def list_files(dirname: str) -> list:
    return [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
    ]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadPanopticAnnotations", with_bbox=True, with_mask=True, with_seg=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SegRescale", scale_factor=1 / 4),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks", "gt_semantic_seg"],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data_root = "/home/cyy/mapillary"
dataset_type = "MaplliaryPanopticDataset"

training_dir = os.path.join(data_root, "training")
validation_dir = os.path.join(data_root, "validation")


def generate_things() -> None:
    classes = set()
    things = set()
    annotation_file = os.path.join(
        training_dir, "v2.0", "panoptic", "panoptic_2020.json"
    )
    with open(annotation_file, "rt", encoding="utf8") as f:
        config = json.load(f)
        print(config["categories"])
        for category in config["categories"]:
            classes.add(category["name"])
            if category["isthing"]:
                things.add(category["name"])
    MaplliaryPanopticDataset.CLASSES = list(sorted(classes))
    MaplliaryPanopticDataset.THING_CLASSES = list(sorted(things))
    MaplliaryPanopticDataset.STUFF_CLASSES = list(sorted(classes - things))
    # gfdgd
    # print("things size is", len(MaplliaryPanopticDataset.THING_CLASSES))
    # print("stuff size is", len(MaplliaryPanopticDataset.STUFF_CLASSES))
    # fdsfdsfe
    # fdsf


generate_things()

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=os.path.join(training_dir, "v2.0", "panoptic", "panoptic_2020.json"),
        img_prefix=os.path.join(training_dir, "images"),
        seg_prefix=os.path.join(training_dir, "v2.0", "panoptic"),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=os.path.join(validation_dir, "v2.0", "panoptic", "panoptic_2020.json"),
        img_prefix=os.path.join(validation_dir, "images"),
        seg_prefix=os.path.join(validation_dir, "annotations"),
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=os.path.join(validation_dir, "annotations.json"),
        img_prefix=os.path.join(validation_dir, "images"),
        seg_prefix=os.path.join(validation_dir, "annotations"),
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=5, metric=["PQ"])
