"""Review-only contracts for the six Occlusion 460-image eval YAML files."""

import json
from pathlib import Path

import pytest

from experiments import run_unit


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("occlusion_*.yaml"))
EVAL_CONFIGS = [path for path in CONFIGS if path.stem in {
    f"occlusion_{model}_{dataset}"
    for model in ("resnet50", "densenet121", "vgg16")
    for dataset in ("imagenet", "voc")
}]
assert len(EVAL_CONFIGS) == 6


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda path: path.stem)
def test_eval_config_matches_debug_gate_without_sharing_output(path):
    config = run_unit._load_config(path)
    model = config["model"]["name"]
    dataset = config["dataset"]["name"]
    gate = run_unit._load_config(ROOT / "configs" / f"occlusion_gate_{model}_{dataset}.yaml")
    manifest = json.loads((ROOT / "docs" / "VOC20_CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))

    assert config["method"] == "occlusion"
    assert config["dataset"] == {"name": dataset, "split": "eval"}
    assert config["model"] == gate["model"]
    assert config["attribution"] == gate["attribution"]
    assert config["metrics"] == gate["metrics"]
    assert config["metrics"]["names"] == ["efficiency_time_ms", "faithfulness_morf_auc_raw"]
    assert config["metrics"]["deletion_fractions"] == [index / 20 for index in range(21)]
    assert config["runtime"] == {
        "device": "cuda", "seed": 42, "max_images": None, "warmup_runs": 1, "resume": True
    }
    if dataset == "voc":
        entry = next(item for item in manifest["checkpoints"] if item["model"] == model)
        assert config["model"]["checkpoint"] == entry["path"]
        assert config["model"]["num_classes"] == 20
        assert config["model"]["output_activation"] == "sigmoid"
    else:
        assert config["model"]["checkpoint"] is None
        assert config["model"]["num_classes"] == 1000
        assert config["model"]["output_activation"] == "softmax"

    prefix = f"results/occlusion_eval/{model}_{dataset}/"
    output = config["output"]
    for key in (
        "per_image_csv", "units_csv", "predictions_csv", "config_dir", "run_log",
        "state_dir", "maps_dir", "float_maps_dir",
    ):
        assert output[key].startswith(prefix)
        assert output[key] != gate["output"].get(key)
    assert output["save_maps"] is True
    assert output["save_float_maps"] is True
    assert run_unit._config_hash(config) != run_unit._config_hash(gate)


def test_eval_output_paths_are_unique_and_not_existing_formal_result_paths():
    configs = [run_unit._load_config(path) for path in EVAL_CONFIGS]
    for field in ("per_image_csv", "units_csv", "predictions_csv", "state_dir", "maps_dir", "float_maps_dir"):
        paths = [config["output"][field] for config in configs]
        assert len(paths) == len(set(paths)) == 6
        assert all(path != "results/per_image.csv" and not path.startswith("results/occlusion_gates/") for path in paths)
