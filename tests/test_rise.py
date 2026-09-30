from pathlib import Path

import torch
import yaml

from experiments.attribution import RISE


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs.mean(dim=(1, 2, 3))
        return torch.stack((-score, score), dim=1)


def test_rise_shape_range_and_determinism():
    model = TinyClassifier().eval()
    image = torch.linspace(0, 1, 3 * 12 * 10).reshape(1, 3, 12, 10)
    rise = RISE(model, num_masks=32, mask_size=4, batch_size=8, seed=7)

    first = rise.attribute(image, target=1)
    second = rise.attribute(image, target=1)

    assert first.shape == (12, 10)
    assert torch.equal(first, second)
    assert 0.0 <= first.min() <= first.max() <= 1.0


def test_rise_rejects_invalid_target():
    rise = RISE(TinyClassifier(), num_masks=4, mask_size=2, batch_size=2)
    image = torch.ones(1, 3, 4, 4)

    try:
        rise.attribute(image, target=2)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("invalid target should fail")


def test_rise_supports_multilabel_sigmoid_scoring():
    model = TinyClassifier().eval()
    image = torch.linspace(0, 1, 3 * 12 * 10).reshape(1, 3, 12, 10)

    softmax_map = RISE(
        model, num_masks=32, mask_size=4, batch_size=8, seed=7,
        output_activation="softmax",
    ).attribute(image, target=1)
    sigmoid_map = RISE(
        model, num_masks=32, mask_size=4, batch_size=8, seed=7,
        output_activation="sigmoid",
    ).attribute(image, target=1)

    assert not torch.equal(softmax_map, sigmoid_map)


def test_six_formal_rise_configs_match_the_frozen_contract():
    deletion_fractions = [index / 20 for index in range(21)]
    units = set()
    for model in ("vgg16", "resnet50", "densenet121"):
        for dataset in ("imagenet", "voc"):
            path = Path("configs") / f"rise_{model}_{dataset}.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            units.add((config["method"], config["model"]["name"], config["dataset"]["name"]))

            assert config["dataset"]["split"] == "eval"
            assert config["attribution"]["num_masks"] >= 4000
            assert config["metrics"]["names"] == [
                "efficiency_time_ms",
                "faithfulness_morf_auc_raw",
            ]
            assert config["metrics"]["deletion_fractions"] == deletion_fractions
            assert config["runtime"]["max_images"] is None
            assert config["runtime"]["warmup_runs"] == 1
            assert config["output"]["save_float_maps"] is True
            assert config["model"]["output_activation"] == (
                "softmax" if dataset == "imagenet" else "sigmoid"
            )
            assert config["attribution"]["output_activation"] == (
                "softmax" if dataset == "imagenet" else "sigmoid"
            )
            if dataset == "voc":
                assert config["model"]["checkpoint"] == (
                    f"models/checkpoints/{model}_voc20.pt"
                )

    assert len(units) == 6
