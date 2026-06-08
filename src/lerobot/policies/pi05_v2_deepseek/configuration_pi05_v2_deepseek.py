#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PI05Config.register_subclass("pi05_v2_deepseek")
@dataclass
class PI05V2DeepseekConfig(PI05Config):
    memory_enabled: bool = True
    memory_history_deltas: list[int] = field(default_factory=lambda: [-12, -4, -1])
    memory_max_cameras: int = 3
    memory_expert_dim: int = 1024
    memory_ff_dim: int = 2048
    memory_view_encoder_layers: int = 1
    memory_temporal_encoder_layers: int = 2
    memory_encoder_heads: int = 8
    memory_detach_vision_features: bool = True

    lora_rank: int = 8
    lora_basis_count: int = 4
    lora_hidden_dim: int = 2048
    lora_target_layers: list[int] = field(default_factory=lambda: [17])
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_scale_init: float = 0.0

    compile_model: bool = False

    # Path to a pretrained pi05 checkpoint directory (pretrained_model/).
    # When set, the pi05 backbone weights (paligemma + action expert) are loaded from this
    # checkpoint, while the new memory encoder and dynamic LoRA hypernet are randomly
    # initialized.  This is intentionally a separate field from `pretrained_path` so that
    # `--policy.type=pi05_v2_deepseek` and this path can coexist on the CLI without
    # triggering the parser's path-vs-type conflict check.
    #
    # Three training scenarios:
    #   1. pi05 backbone init (recommended):
    #        --policy.type=pi05_v2_deepseek --policy.pi05_base_path=<pi05_checkpoint>
    #   2. fully random init (ablation):
    #        --policy.type=pi05_v2_deepseek   (omit pi05_base_path)
    #   3. resume from a saved pi05_v2_deepseek checkpoint:
    #        --policy.pretrained_path=<pi05_v2_deepseek_checkpoint>  (omit type)
    pi05_base_path: str | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.memory_max_cameras <= 0:
            raise ValueError("memory_max_cameras must be positive")
        if len(self.memory_history_deltas) == 0:
            raise ValueError("memory_history_deltas must not be empty")
        if any(delta >= 0 for delta in self.memory_history_deltas):
            raise ValueError("memory_history_deltas must be strictly negative")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_basis_count <= 0:
            raise ValueError("lora_basis_count must be positive")
        if len(self.lora_target_layers) == 0:
            raise ValueError("lora_target_layers must not be empty")
        if len(self.lora_target_modules) == 0:
            raise ValueError("lora_target_modules must not be empty")

        self.memory_history_deltas = sorted(set(self.memory_history_deltas))
        self.lora_target_layers = sorted(set(self.lora_target_layers))
        self.lora_target_modules = list(dict.fromkeys(self.lora_target_modules))

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if not self.memory_enabled:
            return None
        return [*self.memory_history_deltas, 0]