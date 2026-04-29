#可以删掉
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.pi05_v2_deepseek.configuration_pi05_v2_deepseek import PI05V2DeepseekConfig
from lerobot.policies.pi05_v2_deepseek.dynamic_lora_pi05_v2_deepseek import (
    DynamicLoRALinear,
    DynamicLoRAParams,
)
from lerobot.policies.pi05_v2_deepseek.memory_pi05_v2_deepseek import Pi05V2DeepseekMemoryEncoder
from lerobot.policies.pi05_v2_deepseek.processor_pi05_v2_deepseek import (
    MEMORY_DELTA_INDEX_KEY,
    MEMORY_IMAGES_WINDOW_KEY,
    MEMORY_STATE_WINDOW_KEY,
    Pi05V2DeepseekPrepareMemoryProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def test_factory_registration_for_pi05_v2_deepseek():
    policy_cls = get_policy_class("pi05_v2_deepseek")
    policy_cfg = make_policy_config("pi05_v2_deepseek")

    assert policy_cls.name == "pi05_v2_deepseek"
    assert isinstance(policy_cfg, PI05V2DeepseekConfig)


def test_dynamic_lora_linear_matches_base_without_context():
    base = torch.nn.Linear(6, 4, bias=True)
    wrapper = DynamicLoRALinear(base, target_key="unit.test")
    inputs = torch.randn(2, 3, 6)

    torch.testing.assert_close(wrapper(inputs), base(inputs))

    wrapper.set_dynamic_lora(
        DynamicLoRAParams(
            A=torch.randn(2, 2, 6),
            B=torch.randn(2, 4, 2),
            scale=torch.zeros(2),
        )
    )
    torch.testing.assert_close(wrapper(inputs), base(inputs))


def test_memory_processor_splits_history_and_current():
    step = Pi05V2DeepseekPrepareMemoryProcessorStep(history_deltas=(-3, -1), max_cameras=3)
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.randn(3, 4),
            f"{OBS_IMAGES}.cam0": torch.rand(3, 3, 16, 16),
            f"{OBS_IMAGES}.cam1": torch.rand(3, 3, 16, 16),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["push puck"],
        },
    }

    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    comp = out[TransitionKey.COMPLEMENTARY_DATA]

    assert obs[OBS_STATE].shape == (1, 4)
    assert comp[MEMORY_STATE_WINDOW_KEY].shape == (1, 2, 4)
    assert comp[MEMORY_IMAGES_WINDOW_KEY].shape == (1, 2, 3, 3, 16, 16)
    assert comp[MEMORY_DELTA_INDEX_KEY].shape == (1, 2)


def test_memory_encoder_output_shape():
    config = PI05V2DeepseekConfig(device="cpu")
    encoder = Pi05V2DeepseekMemoryEncoder(config=config, vision_dim=2048, expert_dim=1024)

    batch_size = 2
    num_steps = len(config.memory_history_deltas)
    num_cameras = config.memory_max_cameras
    vision_tokens = torch.randn(batch_size * num_steps * num_cameras, 8, 2048)
    image_mask = torch.ones(batch_size, num_steps, num_cameras, dtype=torch.bool)
    state_window = torch.randn(batch_size, num_steps, 4)
    state_pad = torch.zeros(batch_size, num_steps, dtype=torch.bool)
    delta_indices = torch.tensor(config.memory_history_deltas, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    latent = encoder(
        vision_tokens=vision_tokens,
        image_mask=image_mask,
        state_window=state_window,
        state_pad=state_pad,
        delta_indices=delta_indices,
    )

    assert latent.shape == (batch_size, 1024)


def test_make_pre_post_processors_for_pi05_v2_deepseek():
    config = PI05V2DeepseekConfig(device="cpu")
    config.input_features = {
        f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}

    preprocessor, postprocessor = make_pre_post_processors(config, None)

    assert preprocessor is not None
    assert postprocessor is not None