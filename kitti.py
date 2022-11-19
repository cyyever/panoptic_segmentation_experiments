import json
import os
import re

import cv2
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco_panoptic import CocoPanopticDataset


@DATASETS.register_module()
class KITTIPanopticDataset(CocoPanopticDataset):
    CLASSES = [
        "unlabeled",
        "ego vehicle",
        "rectification border",
        "out of roi",
        "static",
        "dynamic",
        "ground",
        "road",
        "sidewalk",
        "parking",
        "rail track",
        "building",
        "wall",
        "fence",
        "guard rail",
        "bridge",
        "tunnel",
        "pole",
        "polegroup",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "caravan",
        "trailer",
        "train",
        "motorcycle",
        "bicycle",
    ]
    THING_CLASSES: list = []


def list_files(dirname: str) -> list:
    return [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
    ]


def generate_kitti_things(kitti_root: str) -> list:
    thing_ids = set()
    training_annotation_dir = os.path.join(kitti_root, "training", "instance")
    annotation_paths = list_files(training_annotation_dir)
    for annotation_path in annotation_paths:
        img = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
        instances = img % 256
        thing_ids |= set(instances.reshape(-1).tolist())
    thing_ids.remove(0)
    return [KITTIPanopticDataset.CLASSES[thing_id] for thing_id in thing_ids]


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
        match_res = re.match("^(0-9]+)_10[.]png$", basename)
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


def prepare_kitti_panoptiic_dataset(kitti_root: str = "/home/cyy/kitti") -> dict:
    if not KITTIPanopticDataset.THING_CLASSES:
        KITTIPanopticDataset.THING_CLASSES = generate_kitti_things(kitti_root)

    training_dir = os.path.join(kitti_root, "training", "image_2")
    training_json = os.path.join(training_dir, "training_annotation.json")
    if not os.path.exists(training_json):
        image_paths = list_files(training_dir)
        with open(training_json, "wt", encoding="utf8") as f:
            json.dump(generate_kitti_panoptic_annotations(image_paths), f)
    test_dir = os.path.join(kitti_root, "testing", "image_2")

    dataset_type = "KITTIPanopticDataset"
    return dict(
        train=dict(
            type=dataset_type,
            ann_file="training_json",
            img_prefix="training_dir",
        ),
        val=dict(
            type=dataset_type,
            ann_file="training_json",
            img_prefix="training_dir",
        ),
        test=dict(
            type=dataset_type,
            img_prefix=test_dir,
        ),
    )


if __name__ == "__main__":
    res = prepare_kitti_panoptiic_dataset("/home/cyy/kitti")
    print(res)
