from pathlib import Path

import pytest
import yaml

from experiments.run_exp import step3_feature_analysis


DEFAULT_CONFIG = Path("configs/step3_defaults.yaml")
MINIMAL_STEP3_ARGV = ["--model", "gptoss20b"]


def test_step3_defaults_config_is_noop():
    no_config = step3_feature_analysis.parse_args(MINIMAL_STEP3_ARGV)
    with_defaults_config = step3_feature_analysis.parse_args(
        ["--config", str(DEFAULT_CONFIG), *MINIMAL_STEP3_ARGV]
    )

    assert "config" not in vars(no_config)
    assert vars(with_defaults_config) == vars(no_config)


def test_step3_config_precedence(tmp_path):
    config_path = tmp_path / "step3_override.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 123, "alpha": 0.2}), encoding="utf-8")

    from_config = step3_feature_analysis.parse_args(
        ["--config", str(config_path), *MINIMAL_STEP3_ARGV]
    )
    assert from_config.seed == 123
    assert from_config.alpha == 0.2

    cli_override = step3_feature_analysis.parse_args(
        [
            "--config",
            str(config_path),
            *MINIMAL_STEP3_ARGV,
            "--seed",
            "999",
            "--alpha",
            "0.01",
        ]
    )
    assert cli_override.seed == 999
    assert cli_override.alpha == 0.01


def test_step3_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "bad_step3.yaml"
    config_path.write_text(yaml.safe_dump({"not_a_real_step3_arg": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        step3_feature_analysis.parse_args(["--config", str(config_path), *MINIMAL_STEP3_ARGV])

from experiments.run_exp import step4_stratified_analysis, step5_commonality_analysis


@pytest.mark.parametrize(
    ("module", "default_config", "minimal_argv"),
    [
        (step4_stratified_analysis, Path("configs/step4_defaults.yaml"), ["--model", "gptoss20b"]),
        (step5_commonality_analysis, Path("configs/step5_defaults.yaml"), ["--model", "gptoss20b"]),
    ],
)
def test_step4_step5_defaults_config_is_noop(module, default_config, minimal_argv):
    no_config = module.parse_args(minimal_argv)
    with_defaults_config = module.parse_args(["--config", str(default_config), *minimal_argv])

    assert "config" not in vars(no_config)
    assert vars(with_defaults_config) == vars(no_config)


@pytest.mark.parametrize(
    ("module", "minimal_argv"),
    [
        (step4_stratified_analysis, ["--model", "gptoss20b"]),
        (step5_commonality_analysis, ["--model", "gptoss20b"]),
    ],
)
def test_step4_step5_config_precedence(tmp_path, module, minimal_argv):
    config_path = tmp_path / "override.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 123, "alpha": 0.2}), encoding="utf-8")

    from_config = module.parse_args(["--config", str(config_path), *minimal_argv])
    assert from_config.seed == 123
    assert from_config.alpha == 0.2

    cli_override = module.parse_args(
        [
            "--config",
            str(config_path),
            *minimal_argv,
            "--seed",
            "999",
            "--alpha",
            "0.01",
        ]
    )
    assert cli_override.seed == 999
    assert cli_override.alpha == 0.01


@pytest.mark.parametrize(
    ("module", "minimal_argv"),
    [
        (step4_stratified_analysis, ["--model", "gptoss20b"]),
        (step5_commonality_analysis, ["--model", "gptoss20b"]),
    ],
)
def test_step4_step5_config_rejects_unknown_keys(tmp_path, module, minimal_argv):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump({"not_a_real_arg": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        module.parse_args(["--config", str(config_path), *minimal_argv])

from experiments.run_exp import step6_cross_comparison


def test_step6_defaults_config_is_noop():
    no_config = step6_cross_comparison.parse_args([])
    with_defaults_config = step6_cross_comparison.parse_args(
        ["--config", "configs/step6_defaults.yaml"]
    )

    assert "config" not in vars(no_config)
    assert vars(with_defaults_config) == vars(no_config)


def test_step6_config_precedence(tmp_path):
    config_path = tmp_path / "step6_override.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 123, "alpha": 0.2}), encoding="utf-8")

    from_config = step6_cross_comparison.parse_args(["--config", str(config_path)])
    assert from_config.seed == 123
    assert from_config.alpha == 0.2

    cli_override = step6_cross_comparison.parse_args(
        [
            "--config",
            str(config_path),
            "--seed",
            "999",
            "--alpha",
            "0.01",
        ]
    )
    assert cli_override.seed == 999
    assert cli_override.alpha == 0.01


def test_step6_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "bad_step6.yaml"
    config_path.write_text(yaml.safe_dump({"not_a_real_step6_arg": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        step6_cross_comparison.parse_args(["--config", str(config_path)])
