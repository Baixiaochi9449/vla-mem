#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#新增加的文件。
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

import pytest

pytest.importorskip("metaworld")

from lerobot.envs.metaworld import _configure_mujoco_gl_backend, _should_use_headless_egl


def test_should_use_headless_egl_without_display_on_linux():
    environ = {}

    assert _should_use_headless_egl(environ=environ, platform="linux") is True


def test_should_not_override_existing_backend():
    environ = {"MUJOCO_GL": "osmesa"}

    assert _should_use_headless_egl(environ=environ, platform="linux") is False
    assert _configure_mujoco_gl_backend(environ=environ, platform="linux") == "osmesa"


def test_should_not_enable_egl_when_display_is_available():
    environ = {"DISPLAY": ":0"}

    assert _should_use_headless_egl(environ=environ, platform="linux") is False
    assert _configure_mujoco_gl_backend(environ=environ, platform="linux") is None


def test_configure_sets_egl_in_headless_linux():
    environ = {}

    assert _configure_mujoco_gl_backend(environ=environ, platform="linux") == "egl"
    assert environ["MUJOCO_GL"] == "egl"