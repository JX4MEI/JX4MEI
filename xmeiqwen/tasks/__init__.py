from xmeiqwen.common.registry import registry
from xmeiqwen.tasks.base_task import BaseTask
from xmeiqwen.tasks.video_text_pretrain import ImageTextPretrainTask

def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."
    task_name = cfg.run_cfg.task 
    task = registry.get_task_class(task_name).setup_task(cfg=cfg) 
    # task = xmeiqwen.tasks.Image_text_pretrain.ImageTextPretrainTask
    assert task is not None, "Task {} not properly registered.".format(task_name)
    return task

__all__ = [
    "BaseTask",
    "ImageTextPretrainTask"
]
