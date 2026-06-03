# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import importlib.util
import sys
from pathlib import Path


def _bootstrap_hyphen_policy_package(package_name: str, directory_name: str) -> None:
    qualified_name = f"{__name__}.{package_name}"
    if qualified_name in sys.modules:
        return

    package_root = Path(__file__).resolve().parent / directory_name
    init_file = package_root / "__init__.py"
    if not init_file.exists():
        return

    spec = importlib.util.spec_from_file_location(
        qualified_name,
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)


_bootstrap_hyphen_policy_package("pi0_v3", "pi0-v3")

from .act.configuration_act import ACTConfig as ACTConfig
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
from .groot.configuration_groot import GrootConfig as GrootConfig
from .multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig as MultiTaskDiTConfig
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .pi0_fast.configuration_pi0_fast import PI0FastConfig as PI0FastConfig
from .pi05.configuration_pi05 import PI05Config as PI05Config
try:
    from .pi0_v3.configuration_pi0_v3 import PI0V3Config as PI0V3Config
except ImportError:
    PI0V3Config = None
try:
    from .pi05_v1.configuration_pi05_v1 import PI05V1Config as PI05V1Config
except ImportError:
    PI05V1Config = None
from .pi05_v2_deepseek.configuration_pi05_v2_deepseek import PI05V2DeepseekConfig as PI05V2DeepseekConfig
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .smolvla.processor_smolvla import SmolVLANewLineProcessor
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig
from .wall_x.configuration_wall_x import WallXConfig as WallXConfig
from .xvla.configuration_xvla import XVLAConfig as XVLAConfig

__all__ = [
    "ACTConfig",
    "DiffusionConfig",
    "MultiTaskDiTConfig",
    "PI0Config",
    "PI0V3Config",
    "PI05Config",
    "PI05V2DeepseekConfig",
    "PI0FastConfig",
    "SmolVLAConfig",
    "SARMConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "GrootConfig",
    "XVLAConfig",
    "WallXConfig",
]

if PI0V3Config is None:
    __all__.remove("PI0V3Config")

if PI05V1Config is not None:
    __all__.append("PI05V1Config")
