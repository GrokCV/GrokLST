# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmagic.utils import print_colored_log
import sys

sys.path.append(os.getcwd())
print(os.getcwd())
from groklst.models.editors import *
from groklst.datasets import *
from groklst.datasets.transforms import *
from groklst.evaluation import *
from groklst.models.data_preprocessors import *

from groklst.models.losses import *
from groklst.visualization import *


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="Test (and eval) a model")
    # TODO ==================================suft
    parser.add_argument(
        "--config",
        default="configs/1_suft/black_river_suft_x4-512-128_1xb16-20k_lst.py",
        help="test config file path",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/black_river_suft_x4-512-128_1xb16-20k_lst/train_20240130_111731/best_LST_MAE_iter_10000.pth",
        help="checkpoint file",
    )
    # TODO ==================================
    parser.add_argument("--out", help="the file to save metric results.")
    parser.add_argument("--work-dir", help="the directory to save the file containing evaluation metrics")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
    # we save log and checkpoint at 'work_dirs/cfg_file/train_timestamp/' dir.
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    timestamp = "test_" + timestamp
    cfg.work_dir = osp.join(cfg.work_dir, timestamp)

    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    print_colored_log(f"Working directory: {cfg.work_dir}")
    print_colored_log(f"Log directory: {runner._log_dir}")

    if args.out:

        class SaveMetricHook(Hook):
            def after_test_epoch(self, _, metrics=None):
                if metrics is not None:
                    mmengine.dump(metrics, args.out)

        runner.register_hook(SaveMetricHook(), "LOWEST")

    # start testing
    runner.test()


if __name__ == "__main__":
    main()
