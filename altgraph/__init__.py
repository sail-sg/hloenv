__version__ = "0.0.1"
from .python.py_hlo_env import HloEnv  # noqa F401,F403
from .python.py_hlo_env import HloModule  # noqa F401,F403
from .python.py_hlo_env import AltPipeline  # noqa F401,F403
from .python.py_hlo_env import Pipeline  # noqa F401,F403
from .python.py_hlo_env import Pass  # noqa F401,F403
from .python.py_hlo_env import GpuBackend  # noqa F401,F403
from .python.py_hlo_env import EvaluationResult  # noqa F401,F403
from .python.py_hlo_env import hlo_pass as HloPass

# HloEnv.__module__ = "altgraph"
# HloModule.__module__ = "altgraph"
# AltPipeline.__module__ = "altgraph"
# Pipeline.__module__ = "altgraph"
# Pass.__module__ = "altgraph"
# GpuBackend.__module__ = "altgraph"
# EvaluationResult.__module__ = "altgraph"

__all__ = [
  "HloEnv",
  "HloModule",
  "AltPipeline",
  "Pipeline",
  "Pass",
  "HloPass",
  "GpuBackend",
  "EvaluationResult",
]
