# dataset settings

import json
import os

from cyy_naive_lib.algorithm.mapping_op import change_mapping_values
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco_panoptic import CocoPanopticDataset


@DATASETS.register_module()
class KITTIPanopticDataset(CocoPanopticDataset):
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
    # dict(type="SegRescale", scale_factor=1 / 4),
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

data_root = "/ssd2/kitti_panoptic"
dataset_type = "KITTIPanopticDataset"

training_dir = os.path.join(data_root, "training")
validation_dir = os.path.join(data_root, "validation")


def generate_kitti_things() -> None:
    things = set()
    classes: dict = {}
    annotation_file = os.path.join(training_dir, "annotations.json")
    class_name_to_id = {}
    with open(annotation_file, "rt", encoding="utf8") as f:
        annotations = json.load(f)
        for category in annotations["categories"]:
            category_id = int(category["id"])
            assert category_id not in classes
            classes[category_id] = category["name"]
            assert category["name"] not in class_name_to_id
            class_name_to_id[category["name"]] = category_id
            if category["isthing"]:
                things.add(category["name"])
    print(classes)

    KITTIPanopticDataset.THING_CLASSES = list(sorted(things))
    KITTIPanopticDataset.STUFF_CLASSES = list(sorted(set(classes.values()) - things))
    print(len(KITTIPanopticDataset.THING_CLASSES))
    print(len(KITTIPanopticDataset.STUFF_CLASSES))
    KITTIPanopticDataset.CLASSES = (
        KITTIPanopticDataset.THING_CLASSES + KITTIPanopticDataset.STUFF_CLASSES
    )
    category_id_map = {}
    next_id = 1
    for class_name in KITTIPanopticDataset.CLASSES:
        category_id_map[class_name_to_id[class_name]] = next_id
        next_id += 1
    print("category_id_map is", category_id_map)
    assert len(category_id_map) == len(classes)

    def get_mapped_category_id(v):
        return category_id_map[v]

    with open(os.path.join(training_dir, "annotations.json"), "rt") as f:
        training_annotations = json.load(f)
        print(training_annotations["annotations"][0])
        new_training_annotations = change_mapping_values(
            d=training_annotations,
            key="category_id",
            f=get_mapped_category_id,
        )
        print("new", new_training_annotations["annotations"][0])
        new_training_annotations["categories"] = change_mapping_values(
            d=new_training_annotations["categories"],
            key="id",
            f=get_mapped_category_id,
        )
        with open(os.path.join(training_dir, "new_annotations.json"), "wt") as f:
            json.dump(new_training_annotations, f)

    with open(os.path.join(validation_dir, "annotations.json"), "rt") as f:
        validation_annotations = json.load(f)
        print(validation_annotations["annotations"][0])
        new_validation_annotations = change_mapping_values(
            d=validation_annotations,
            key="category_id",
            f=get_mapped_category_id,
        )
        print("new", new_validation_annotations["annotations"][0])
        new_validation_annotations["categories"] = change_mapping_values(
            d=new_validation_annotations["categories"],
            key="id",
            f=get_mapped_category_id,
        )
        with open(os.path.join(validation_dir, "new_annotations.json"), "wt") as f:
            json.dump(new_validation_annotations, f)


generate_kitti_things()


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=os.path.join(training_dir, "new_annotations.json"),
        img_prefix=os.path.join(training_dir, "images"),
        seg_prefix=os.path.join(training_dir, "annotations"),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=os.path.join(validation_dir, "new_annotations.json"),
        img_prefix=os.path.join(validation_dir, "images"),
        seg_prefix=os.path.join(validation_dir, "annotations"),
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=os.path.join(validation_dir, "new_annotations.json"),
        img_prefix=os.path.join(validation_dir, "images"),
        seg_prefix=os.path.join(validation_dir, "annotations"),
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric=["PQ"])
