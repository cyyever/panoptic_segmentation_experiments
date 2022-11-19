import os

import mmcv
from mmcv import Config
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import CocoPanopticDataset, build_dataset
from mmdet.models import build_detector
# from mmseg.apis import set_random_seed, train_segmentor
from mmseg.utils import get_device

from datasets import kitti

if __name__ == "__main__":
    cfg = Config.fromfile("configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py")
    # modify num classes of the model in decode/auxiliary head
    cfg.data = kitti.prepare_kitti_panoptiic_dataset("/home/cyy/kitti2/kitti_panoptic")

    # Modify dataset type and path

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu = 8

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        # dict(type="Resize", img_scale=(320, 240), ratio_range=(0.5, 2.0)),
        # dict(type="RandomCrop", crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type="RandomFlip", flip_ratio=0.5),
        # dict(type="PhotoMetricDistortion"),
        dict(type="Normalize", **cfg.img_norm_cfg),
        # dict(type="Pad", size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type="DefaultFormatBundle"),
        # dict(type="Collect", keys=["img", "gt_semantic_seg"]),
    ]
    cfg.data.train["pipeline"] = cfg.train_pipeline

    cfg.test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        # dict(type="Resize", img_scale=(320, 240), ratio_range=(0.5, 2.0)),
        # dict(type="RandomCrop", crop_size=cfg.crop_size, cat_max_ratio=0.75),
        # dict(type="RandomFlip", flip_ratio=0.5),
        # dict(type="PhotoMetricDistortion"),
        dict(type="Normalize", **cfg.img_norm_cfg),
        # dict(type="Pad", size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type="DefaultFormatBundle"),
        # dict(type="Collect", keys=["img", "gt_semantic_seg"]),
    ]
    cfg.data.val["pipeline"] = cfg.test_pipeline
    cfg.data.test["pipeline"] = cfg.test_pipeline
    print(cfg.data)
    # cfg.test_pipeline = [
    #     dict(type="LoadImageFromFile"),
    #     dict(
    #         type="MultiScaleFlipAug",
    #         img_scale=(320, 240),
    #         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #         flip=False,
    #         transforms=[
    #             dict(type="Resize", keep_ratio=True),
    #             dict(type="RandomFlip"),
    #             dict(type="Normalize", **cfg.img_norm_cfg),
    #             dict(type="ImageToTensor", keys=["img"]),
    #             dict(type="Collect", keys=["img"]),
    #         ],
    #     ),
    # ]

    # Set up working dir to save files and logs.
    cfg.work_dir = "./work_dirs/tutorial"

    cfg.runner.max_epochs = 200
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 200
    cfg.checkpoint_config.interval = 200

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(2)
    cfg.device = get_device()

    # Let's have a look at the final config used for training
    # print(f"Config:\n{cfg.pretty_text}")

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)

    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True, meta=dict())
