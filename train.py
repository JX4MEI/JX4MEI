import os
import time
import random
import argparse
import numpy as np

import torch
from datetime import datetime
import torch.backends.cudnn as cudnn

import xmeiqwen.tasks as tasks
from xmeiqwen.common.config import Config
from xmeiqwen.common.dist_utils import get_rank, init_distributed_mode
from xmeiqwen.common.logger import setup_logger
from xmeiqwen.common.registry import registry
from xmeiqwen.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from xmeiqwen.tasks import *
from xmeiqwen.models import *
from xmeiqwen.runners import *
from xmeiqwen.processors import *
from xmeiqwen.datasets.builders import *

def setup_seeds(config): 
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="overwrite params in xxx.config (only for run and model). Example: --options 'ckpt=aaa' 'ckpt_2=bbb'")
    args = parser.parse_args()
    return args

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base")) # 'affectgpt.runners.runner_base.RunnerBase'
    return runner_cls

def main():

    args = parse_args()
    cfg = Config(args)


    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d%H%M')[:-1]}" # zhuofan

    print (job_id)

    # print logging files
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger() 
    cfg.pretty_print()

    # load task and start training
    task = tasks.setup_task(cfg) # image_text_pretrain
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id, 
        task=task, 
        model=model, 
        datasets=datasets
    )
    runner.train()

if __name__ == "__main__":
    main()
