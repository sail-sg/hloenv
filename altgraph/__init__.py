# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
