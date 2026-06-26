# Optional YAML Configs

Steps 3, 4, 5, and 6 can optionally read a flat YAML file with argparse destination names as keys:

```bash
PYTHONPATH=. conda run -n alag python experiments/run_exp/step3_feature_analysis.py \
  --config configs/step3_defaults.yaml \
  --model gptoss20b \
  --method self_reported
```

The same pattern works for Step 4, Step 5, and Step 6 with `configs/step4_defaults.yaml`, `configs/step5_defaults.yaml`, and `configs/step6_defaults.yaml`.

Precedence is:

1. argparse hardcoded defaults
2. YAML values from `--config`
3. explicit CLI flags

Argparse remains the source of truth for defaults. Do not move defaults into YAML.
The `*_defaults.yaml` files are generated documentation snapshots of the current parser defaults; loading one should produce the same parsed Namespace as omitting `--config`.

Config files must be flat mappings. Keys are validated against the target step's argparse destination names, so typos fail fast instead of silently changing a run.
