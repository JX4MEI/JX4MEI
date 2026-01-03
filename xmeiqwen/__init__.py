import os
import sys
from omegaconf import OmegaConf

from xmeiqwen.tasks import *
from xmeiqwen.models import *
from xmeiqwen.processors import *
from xmeiqwen.datasets.builders import *
from xmeiqwen.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("library_root", root_dir)

repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)

cache_root = os.path.join(repo_root, "cache")
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)

registry.register("SPLIT_NAMES", ["train", "val", "test"])
