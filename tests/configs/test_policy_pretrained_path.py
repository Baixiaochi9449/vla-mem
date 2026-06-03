from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.train import TrainPipelineConfig


@dataclass
class _FakePolicy:
    type: str = "pi05"
    pretrained_path: Path | None = None
    push_to_hub: bool = False
    repo_id: str | None = None


@dataclass
class _FakeEnv:
    type: str = "robomme"


def test_eval_pipeline_uses_policy_pretrained_path_when_policy_path_is_missing(monkeypatch):
    fake_policy = _FakePolicy()
    captured: dict[str, object] = {}

    monkeypatch.setattr("lerobot.configs.eval.parser.get_path_arg", lambda name: None)
    monkeypatch.setattr(
        "lerobot.configs.eval.parser.parse_arg",
        lambda name, args=None: "lerobot/pi05_base" if name == "policy.pretrained_path" else None,
    )
    monkeypatch.setattr("lerobot.configs.eval.parser.get_cli_overrides", lambda name: ["--device=cpu"])

    def _from_pretrained(path, cli_overrides=None):
        captured["path"] = path
        captured["cli_overrides"] = cli_overrides
        return fake_policy

    monkeypatch.setattr("lerobot.configs.eval.PreTrainedConfig.from_pretrained", _from_pretrained)

    cfg = EvalPipelineConfig(env=_FakeEnv())

    assert captured == {"path": "lerobot/pi05_base", "cli_overrides": ["--device=cpu"]}
    assert cfg.policy is fake_policy
    assert cfg.policy.pretrained_path == Path("lerobot/pi05_base")


def test_filter_path_args_removes_policy_args_when_pretrained_path_is_present():
    args = [
        "--policy.pretrained_path=lerobot/pi05_base",
        "--policy.push_to_hub=false",
        "--steps=1",
    ]

    filtered = parser.filter_path_args("policy", args)

    assert filtered == ["--steps=1"]


def test_get_cli_overrides_excludes_pretrained_path():
    args = [
        "--policy.pretrained_path=lerobot/pi05_base",
        "--policy.push_to_hub=false",
        "--policy.device=cpu",
    ]

    overrides = parser.get_cli_overrides("policy", args)

    assert overrides == ["--push_to_hub=false", "--device=cpu"]


def test_train_pipeline_uses_policy_pretrained_path_when_policy_path_is_missing(monkeypatch):
    fake_policy = _FakePolicy()
    captured: dict[str, object] = {}

    monkeypatch.setattr("lerobot.configs.train.parser.get_path_arg", lambda name: None)
    monkeypatch.setattr(
        "lerobot.configs.train.parser.parse_arg",
        lambda name, args=None: "lerobot/pi05_base" if name == "policy.pretrained_path" else None,
    )
    monkeypatch.setattr("lerobot.configs.train.parser.get_cli_overrides", lambda name: ["--device=cpu"])

    def _from_pretrained(path, cli_overrides=None):
        captured["path"] = path
        captured["cli_overrides"] = cli_overrides
        return fake_policy

    monkeypatch.setattr("lerobot.configs.train.PreTrainedConfig.from_pretrained", _from_pretrained)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/robomme_ee_pose_train"),
        use_policy_training_preset=False,
        optimizer=object(),
        scheduler=object(),
    )
    cfg.validate()

    assert captured == {"path": "lerobot/pi05_base", "cli_overrides": ["--device=cpu"]}
    assert cfg.policy is fake_policy
    assert cfg.policy.pretrained_path == Path("lerobot/pi05_base")