import os
from pathlib import Path
from typing import Optional

from absl import flags, logging
from ml_collections import ConfigDict



def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg



# def load_cfgs(
#     _TASK_FILE,
#     _NEF_FILE,
#     _SCHEDULER_FILE: Optional[flags.FlagHolder] = None,
#     _OPTIMIZER_FILE: Optional[flags.FlagHolder] = None,
# ):
#     cfg = _TASK_FILE.value
#     nef_cfg = _NEF_FILE.value

#     # TODO find a way to not have to do this
#     nef_cfg.unlock()
#     nef_cfg.params.output_dim = cfg.dataset.get("out_channels", 1)
#     nef_cfg.lock()

#     if _SCHEDULER_FILE is not None:
#         scheduler_cfg = _SCHEDULER_FILE.value
#         cfg.unlock()
#         cfg["scheduler"] = scheduler_cfg
#         cfg.lock()

#     if _OPTIMIZER_FILE is not None:
#         optimizer_cfg = _OPTIMIZER_FILE.value
#         cfg.unlock()
#         cfg["optimizer"] = optimizer_cfg
#         cfg.lock()

#     return cfg, nef_cfg
