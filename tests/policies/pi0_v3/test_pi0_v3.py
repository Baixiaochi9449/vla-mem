#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.policies.pi0_v3.configuration_pi0_v3 import PI0V3Config
from lerobot.policies.pi0_v3.memory_pi0_v3 import HybridSlotMemoryBank
from lerobot.policies.pi0_v3.modeling_pi0_v3 import PI0V3Policy
from lerobot.policies.pi0_v3.processor_pi0_v3 import (
    MEMORY_ACTION_PAD_KEY,
    MEMORY_ACTION_WINDOW_KEY,
    MEMORY_IMAGE_WINDOW_PREFIX,
    MEMORY_STATE_PAD_KEY,
    MEMORY_STATE_WINDOW_KEY,
    Pi0V3PrepareMemoryProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def test_pi0_v3_config_factory_and_history_indices():
    config = make_policy_config(
        policy_type="pi0_v3",
        memory_sequence_lookback=4,
        memory_slots=6,
        chunk_size=8,
        max_state_dim=14,
        max_action_dim=7,
    )

    assert isinstance(config, PI0V3Config)
    assert config.observation_delta_indices == [-3, -2, -1, 0]
    assert config.action_delta_indices[:5] == [-3, -2, -1, 0, 1]


def test_pi0_v3_prepare_memory_processor_step_splits_history_windows():
    step = Pi0V3PrepareMemoryProcessorStep(max_state_dim=6, history_action_length=2, chunk_size=4)
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor(
                [
                    [[-1.0, -0.5, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.5, 1.0]],
                    [[0.2, 0.1, 0.0], [0.3, 0.2, 0.1], [0.4, 0.3, 0.2]],
                ],
                dtype=torch.float32,
            ),
            OBS_IMAGE: torch.rand(2, 3, 3, 8, 8),
        },
        TransitionKey.ACTION: torch.rand(2, 6, 4),
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick the cube", "open the drawer"],
            f"{OBS_STATE}_is_pad": torch.tensor([[True, False, False], [False, False, False]]),
        },
    }

    processed = step(transition)
    observation = processed[TransitionKey.OBSERVATION]
    complementary_data = processed[TransitionKey.COMPLEMENTARY_DATA]

    assert observation[OBS_STATE].shape == (2, 3)
    assert complementary_data[MEMORY_STATE_WINDOW_KEY].shape == (2, 3, 3)
    assert complementary_data[MEMORY_STATE_PAD_KEY].shape == (2, 3)
    assert complementary_data[f"{MEMORY_IMAGE_WINDOW_PREFIX}{OBS_IMAGE}"].shape == (2, 3, 3, 8, 8)
    assert complementary_data[MEMORY_ACTION_WINDOW_KEY].shape == (2, 2, 4)
    assert complementary_data[MEMORY_ACTION_PAD_KEY].shape == (2, 2)
    assert processed[TransitionKey.ACTION].shape == (2, 4, 4)
    assert complementary_data["task"][0].startswith("Task: pick the cube")


def test_pi0_v3_slot_memory_rollout_and_online_update():
    config = PI0V3Config(
        max_state_dim=6,
        max_action_dim=4,
        memory_slots=4,
        memory_topk=2,
        memory_token_count=2,
        memory_key_dim=8,
        memory_value_dim=12,
        memory_phase_embed_dim=4,
        memory_phase_codebook_size=8,
        memory_sequence_lookback=3,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    config.validate_features()

    memory_bank = HybridSlotMemoryBank(
        config=config,
        prefix_dim=16,
        expert_dim=12,
        state_dim=6,
        action_dim=4,
    )

    batch_size = 2
    seq_len = 3
    write_summaries = torch.randn(batch_size, seq_len, 16)
    state_window = torch.randn(batch_size, seq_len, 6)
    pad_mask = torch.tensor([[True, False, False], [False, False, False]])
    prefix_summary = torch.randn(batch_size, 16)
    current_state = torch.randn(batch_size, 6)

    state, rollout_info = memory_bank.rollout_window(write_summaries, state_window, pad_mask)
    query = memory_bank.build_query(prefix_summary, current_state)
    read_result = memory_bank.read_from_state(state, query)

    assert state.keys.shape == (batch_size, 4, 8)
    assert rollout_info["last_write_key"].shape == (batch_size, 8)
    assert read_result.tokens.shape == (batch_size, 2, 16)
    assert read_result.context.shape == (batch_size, 12)
    assert read_result.future_action.shape == (batch_size, 4)

    memory_bank.update_online(prefix_summary, current_state, torch.randn(batch_size, 4))
    online_read = memory_bank.read_online(query)
    assert online_read.tokens.shape == (batch_size, 2, 16)


def test_pi0_v3_policy_tiny_forward_and_action_chunk():
    config = PI0V3Config(
        paligemma_variant="gemma_tiny",
        action_expert_variant="gemma_tiny",
        image_resolution=(32, 32),
        tokenizer_max_length=16,
        chunk_size=4,
        n_action_steps=2,
        memory_sequence_lookback=3,
        memory_recent_tokens=2,
        memory_temporal_layers=1,
        memory_temporal_heads=4,
        max_state_dim=4,
        max_action_dim=4,
        device="cpu",
    )
    config.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    config.validate_features()

    policy = PI0V3Policy(config)
    batch_size = 1
    batch = {
        OBS_IMAGE: torch.rand(batch_size, 3, 32, 32),
        OBS_STATE: torch.rand(batch_size, 4),
        MEMORY_STATE_WINDOW_KEY: torch.rand(batch_size, 3, 4),
        MEMORY_STATE_PAD_KEY: torch.tensor([[True, False, False]]),
        f"{MEMORY_IMAGE_WINDOW_PREFIX}{OBS_IMAGE}": torch.rand(batch_size, 3, 3, 32, 32),
        OBS_LANGUAGE_TOKENS: torch.zeros(batch_size, 16, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 16, dtype=torch.bool),
        ACTION: torch.rand(batch_size, 4, 4),
    }
    loss, info = policy.forward(batch)
    chunk = policy.predict_action_chunk(batch)

    assert torch.isfinite(loss)
    assert "memory_aux_loss" in info
    assert chunk.shape == (batch_size, 4, 4)