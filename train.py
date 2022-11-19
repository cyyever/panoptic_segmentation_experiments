import os

import mmcv
from mmcv import Config
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import CocoPanopticDataset, build_dataset
from mmdet.models import build_detector
# from mmseg.apis import set_random_seed, train_segmentor
from mmseg.utils import get_device

from datasets.coco import get_coco_panoptiic_dataset

# from datasets.kitti import prepare_kitti_panoptiic_dataset

if __name__ == "__main__":
    cfg = Config.fromfile("configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py")

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu = 8

    # Set up working dir to save files and logs.
    cfg.work_dir = "./work_dirs/tutorial"

    # cfg.runner.max_epochs = 200
    cfg.runner.max_iters = 200

    # # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
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
