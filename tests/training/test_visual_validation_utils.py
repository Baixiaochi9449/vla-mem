#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#测试文件，可以删除。
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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.utils import validate_visual_features_consistency


def test_validate_visual_features_consistency_accepts_current_signature_with_rename_map():
    dataset_image_keys = ["observation.images.top", "observation.images.side"]
    policy_image_features = {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    rename_map = {
        "observation.images.top": "observation.images.camera1",
        "observation.images.side": "observation.images.camera2",
    }

    validate_visual_features_consistency(dataset_image_keys, policy_image_features, rename_map)


def test_validate_visual_features_consistency_accepts_legacy_config_signature():
    config = PI05Config(max_action_dim=4, max_state_dim=4, dtype="float32")
    config.input_features = {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }

    dataset_features = {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }

    validate_visual_features_consistency(config, dataset_features)