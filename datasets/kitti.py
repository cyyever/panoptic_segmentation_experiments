import json
import os
import re

# import cv2
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco_panoptic import CocoPanopticDataset


@DATASETS.register_module()
class KITTIPanopticDataset(CocoPanopticDataset):
    CLASSES: list = []
    THING_CLASSES: list = []


def list_files(dirname: str) -> list:
    return [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
    ]


def generate_kitti_things(kitti_root: str) -> None:
    things = set()
    classes = set()
    annotation_file = os.path.join(kitti_root, "training", "annotations.json")
    with open(annotation_file, "rt", encoding="utf8") as f:
        annotations = json.load(f)
        for category in annotations["categories"]:
            classes.add(category["name"])
            if category["isthing"]:
                things.add(category["name"])
    KITTIPanopticDataset.CLASSES = list(sorted(classes))
    KITTIPanopticDataset.THING_CLASSES = list(sorted(things))


def generate_kitti_panoptic_annotations(path_list: list) -> dict:
    assert path_list
    annotations = {}
    annotations["categories"] = [
        {str(idx): label} for idx, label in enumerate(KITTIPanopticDataset.CLASSES)
    ]
    annotations["images"] = []
    annotations["annotations"] = []
    for image_path in path_list:
        basename = os.path.basename(image_path)
        # basename is XXXXX_10.png
        match_res = re.match("^([0-9]+)[_]10[.]png$", basename)
        if match_res is None:
            raise RuntimeError(f"can't parse file {basename}")
        image_id: str = match_res.group(1)
        img = mmcv.imread(image_path)
        annotations["images"].append(
            {
                "file_name": image_path,
                "height": img.shape[0],
                "width": img.shape[1],
                "id": int(image_id),
            },
        )
        annotation: dict = {
            "file_name": os.path.join(
                os.path.dirname(image_path), "..", "instance", basename
            ),
            "id": int(image_id),
        }
        annotations["annotations"].append(annotation)
    return annotations


def prepare_kitti_panoptiic_dataset(
    kitti_root: str = "/home/cyy/kitti2/kitti_panoptic",
) -> dict:
    generate_kitti_things(kitti_root)

    training_dir = os.path.join(kitti_root, "training")
    validation_dir = os.path.join(kitti_root, "validation")

    dataset_type = "KITTIPanopticDataset"
    return dict(
        train=dict(
            type=dataset_type,
            ann_file=os.path.join(training_dir, "annotations.json"),
            img_prefix=os.path.join(training_dir, "images"),
            seg_prefix=os.path.join(training_dir, "annotations"),
        ),
        val=dict(
            type=dataset_type,
            ann_file=os.path.join(validation_dir, "annotations.json"),
            img_prefix=os.path.join(validation_dir, "images"),
            seg_prefix=os.path.join(validation_dir, "annotations"),
        ),
        test=dict(
            type=dataset_type,
            ann_file=os.path.join(validation_dir, "annotations.json"),
            img_prefix=os.path.join(validation_dir, "images"),
            seg_prefix=os.path.join(validation_dir, "annotations"),
        ),
    )


if __name__ == "__main__":
    res = prepare_kitti_panoptiic_dataset("/home/cyy/kitti2/kitti_panoptic")
    print(res)
