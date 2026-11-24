"""Issue #28: fixed-protocol, group-held-out attribution-method selection.

Metrics generate offline labels/evaluation ONLY. Features are known conditions.
Two fitted depth-3 CART subtrees sit below a fixed interface-eligibility gate.
Raw per-image exports/split assignments stay in ignored results/, never Git.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from .validate_dcu_team_handoff import EXPECTED_SHA, METHODS, ROOT, validate


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "analysis/d_method_selection_20261122/protocol.json"
FEATURES = ("dataset_is_voc", "model_is_densenet121", "model_is_vgg16",
            "objective_is_stability", "objective_is_efficiency")
OBJECTIVES = ("faithfulness", "stability", "efficiency")
PROFILES = {"full": METHODS, "no_grad_logits": METHODS[2:]}
METRIC = {"faithfulness": "faithfulness_morf_auc_raw", "stability": "stability_spearman",
          "efficiency": "efficiency_time_ms"}
TOL = 1e-12
SEED = 20261122


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def features(rows):
    """Fixed schema: identifiers/outcomes cannot silently become predictors."""
    require(set(rows.dataset) <= {"imagenet", "voc"}, "unknown dataset")
    require(set(rows.model) <= {"resnet50", "densenet121", "vgg16"}, "unknown model")
    require(set(rows.objective) <= set(OBJECTIVES), "unknown objective")
    return np.column_stack([
        rows.dataset.eq("voc"), rows.model.eq("densenet121"), rows.model.eq("vgg16"),
        rows.objective.eq("stability"), rows.objective.eq("efficiency")
    ]).astype(np.float64)


def loss_table(matrix, objective):
    require(objective in OBJECTIVES, "unknown objective")
    a = np.asarray(matrix, dtype=float)
    require(a.ndim == 2 and a.shape[1] >= 2 and np.isfinite(a).all(), "invalid paired matrix")
    return -a if objective == "stability" else a.copy()


def oracle(loss, names):
    loss = np.asarray(loss, float)
    require(loss.ndim == 2 and loss.shape[1] == len(names) and np.isfinite(loss).all(), "oracle shape/values")
    best = loss <= loss.min(axis=1, keepdims=True) + TOL
    # argmax on bool picks the first tie; evaluation keeps the complete tie set.
    return np.asarray(names)[best.argmax(axis=1)], best


def selection_scores(loss, predictions, names):
    """Tie-aware hit, rank regret in [0,1], and native-unit oracle gap."""
    loss = np.asarray(loss, float)
    require(loss.ndim == 2 and loss.shape == (len(predictions), len(names))
            and len(names) >= 2 and np.isfinite(loss).all(), "invalid score matrix")
    require(set(predictions) <= set(names), "recommended method unavailable")
    indices = np.array([names.index(p) for p in predictions])
    chosen = loss[np.arange(len(loss)), indices]
    gap = np.maximum(0., chosen - loss.min(axis=1))
    hits = gap <= TOL
    ranks = (loss < chosen[:, None] - TOL).sum(axis=1) / (len(names) - 1)
    return dict(hit=hits.astype(float), rank_regret=ranks, native_gap=np.where(hits, 0., gap),
                selected_loss=chosen)


def grouped_split(rows, test_size=.25, seed=SEED):
    require("group" in rows and rows.group.nunique() >= 4, "too few original-image groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(rows, groups=rows.group))
    train_groups, test_groups = set(rows.iloc[train].group), set(rows.iloc[test].group)
    require(not train_groups & test_groups, "original-image leakage")
    return train, test


def prepare_cases(per):
    keys = ["dataset", "model", "image_id", "metric", "method"]
    require(not per.duplicated(keys).any(), "duplicate paired observation")
    require(set(per.method) == set(METHODS), "six methods required")
    blocks = []
    for objective in OBJECTIVES:
        paired = per[per.metric.eq(METRIC[objective])].pivot(
            index=["dataset", "model", "image_id"], columns="method", values="value"
        ).reindex(columns=METHODS).sort_index()
        require(not paired.isna().any().any(), "incomplete six-method pairs")
        for profile, names in PROFILES.items():
            part = paired[list(names)].reset_index()
            part["objective"], part["profile"] = objective, profile
            part["group"] = part.dataset + "::" + part.image_id.astype(str)
            part["label"], _ = oracle(loss_table(part[list(names)], objective), names)
            blocks.append(part)
    result = pd.concat(blocks, ignore_index=True).sort_values(
        ["profile", "objective", "dataset", "model", "image_id"]
    ).reset_index(drop=True)
    require(result.groupby(["dataset", "model", "image_id"]).size().eq(6).all(), "objective/profile pair coverage")
    return result


def fit_policy(train, profile, protocol):
    part = train[train.profile.eq(profile)]
    fit = protocol["fit"]
    tree = DecisionTreeClassifier(criterion=fit["criterion"], max_depth=fit["max_depth_per_profile"],
        min_samples_leaf=fit["min_samples_leaf"], random_state=fit["random_state"], class_weight=None)
    tree.fit(features(part), part.label)
    return tree


def baseline_choices(train, names):
    """No dataset/model inputs; every choice learned from training rows only."""
    modal, rank = {}, {}
    for objective in OBJECTIVES:
        p = train[train.objective.eq(objective)]
        counts = p.label.value_counts()
        modal[objective] = max(names, key=lambda n: counts.get(n, 0))
        loss = loss_table(p[list(names)], objective)
        regrets = [selection_scores(loss, [name] * len(p), names)["rank_regret"].mean() for name in names]
        rank[objective] = names[int(np.argmin(regrets))]
    return modal, rank


def export_tree(tree):
    t = tree.tree_
    def node(i):
        info = dict(node=int(i), train_scenarios=int(t.n_node_samples[i]),
                    recommended=str(tree.classes_[t.value[i][0].argmax()]))
        if t.children_left[i] != t.children_right[i]:
            info.update(feature=FEATURES[t.feature[i]], threshold=float(t.threshold[i]),
                        left=node(t.children_left[i]), right=node(t.children_right[i]))
        return info
    return node(0)


def predict_export(node, x):
    result = []
    for row in x:
        leaf = node
        while "feature" in leaf:
            leaf = leaf["left"] if row[FEATURES.index(leaf["feature"])] <= leaf["threshold"] else leaf["right"]
        result.append(leaf["recommended"])
    return np.array(result)


def bootstrap_difference(evaluation, resamples):
    """Stratified original-image cluster bootstrap of paired policy differences."""
    rows = []
    for (profile, objective), part in evaluation.groupby(["profile", "objective"]):
        for baseline in ("goal_mode", "goal_rank"):
            left = part[part.policy.eq("tree")].set_index("case_id")
            right = part[part.policy.eq(baseline)].set_index("case_id")
            require(left.index.is_unique and right.index.is_unique and set(left.index) == set(right.index),
                    "paired-policy identity")
            right = right.reindex(left.index)
            require(left[["dataset", "group"]].equals(right[["dataset", "group"]]), "policy group mismatch")
            frame = left[["dataset", "group"]].copy()
            frame["count"] = 1
            for metric in ("hit", "rank_regret"):
                frame[metric] = left[metric] - right[metric]
            sums = frame.groupby(["dataset", "group"])[["count", "hit", "rank_regret"]].sum()
            rng = np.random.default_rng(SEED + len(rows))
            total = np.zeros((resamples, 3))
            for dataset in sorted(frame.dataset.unique()):
                a = sums.loc[dataset].to_numpy(float)
                weights = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=resamples)
                total += weights @ a
            for j, metric in enumerate(("hit", "rank_regret"), 1):
                low, high = np.quantile(total[:, j] / total[:, 0], [.025, .975])
                rows.append(dict(profile=profile, objective=objective, baseline=baseline, metric=metric,
                    tree_minus_baseline=float(frame[metric].mean()), ci_low=float(low), ci_high=float(high),
                    n_original_images=int(frame.group.nunique()), resamples=resamples))
    return pd.DataFrame(rows)


def leaf_rows(tree, train, test, profile, protocol):
    names = PROFILES[profile]
    a, b = train[train.profile.eq(profile)], test[test.profile.eq(profile)]
    train_ids, test_ids = tree.apply(features(a)), tree.apply(features(b))
    paths = {}
    def visit(i, path):
        t = tree.tree_
        if t.children_left[i] == t.children_right[i]:
            paths[i] = " AND ".join(path) or "all conditions"
        else:
            feature = FEATURES[t.feature[i]]
            visit(t.children_left[i], path + [feature + " = no"])
            visit(t.children_right[i], path + [feature + " = yes"])
    visit(0, [])
    result = []
    for leaf, rule in paths.items():
        part = a.iloc[np.flatnonzero(train_ids == leaf)]
        held = b.iloc[np.flatnonzero(test_ids == leaf)]
        counts = part.label.value_counts()
        recommendation = str(tree.predict(features(part.iloc[:1]))[0])
        shares = {name: float(counts.get(name, 0) / len(part)) for name in names}
        candidates = [name for name in names if shares[name] >= protocol["candidate_threshold"]]
        if recommendation not in candidates:
            candidates.append(recommendation)
        result.append(dict(profile=profile, leaf=int(leaf), rule=rule, recommended=recommendation,
            train_scenarios=len(part), train_original_images=int(part.group.nunique()),
            test_scenarios=len(held), test_original_images=int(held.group.nunique()),
            train_modal_share=shares[recommendation], retain_multiple=(len(candidates) > 1 or shares[recommendation] < protocol["single_candidate_minimum_share"]),
            candidate_methods=" / ".join(candidates), **{f"train_share_{n}": shares.get(n, 0.) for n in METHODS}))
    return pd.DataFrame(result)


def plot_outputs(trees, leaves, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "d-method-tree-20261122"})
    readable = ["Task is VOC", "Model is DenseNet121", "Model is VGG16",
                "Priority is stability", "Priority is efficiency"]
    fig, axes = plt.subplots(2, 1, figsize=(21, 14), constrained_layout=True)
    for ax, (profile, tree) in zip(axes, trees.items()):
        artists = plot_tree(tree, feature_names=readable, class_names=list(tree.classes_),
                  filled=True, rounded=True, impurity=False, node_ids=True, fontsize=10, ax=ax)
        import re
        for artist in artists:
            match = re.search(r"node #(\d+)", artist.get_text())
            if not match:
                # sklearn labels the original <= .5 predicate True/False;
                # our displayed binary question has the opposite wording.
                if artist.get_text() in ("True", "False"):
                    artist.set_text("no" if artist.get_text() == "True" else "yes")
                continue
            node = int(match.group(1))
            if tree.tree_.children_left[node] != tree.tree_.children_right[node]:
                query = readable[tree.tree_.feature[node]] + "?"
                artist.set_text(query + f"\ntrain scenarios = {tree.tree_.n_node_samples[node]}")
            else:
                row = leaves[leaves.profile.eq(profile) & leaves.leaf.eq(node)].iloc[0]
                artist.set_text(f"Recommend: {row.recommended}\n"
                    f"Train scenarios: {row.train_scenarios} ({row.train_original_images} images)\n"
                    f"Holdout scenarios: {row.test_scenarios}\n"
                    f"Training winner share: {row.train_modal_share:.1%}\n"
                    f"Candidates: {row.candidate_methods}")
        # Edge annotations are not included in plot_tree's returned node list
        # in all sklearn versions. Relabel every Axes text, including edges.
        for artist in ax.texts:
            label = artist.get_text().strip()
            if label in ("True", "False"):
                artist.set_text("no" if label == "True" else "yes")
        ax.set_title(f"{'Full access: six methods' if profile == 'full' else 'No gradients + logits: four perturbation methods'}"
                     f" | fitted CART depth {tree.get_depth()} | left = no, right = yes", fontsize=15)
    fig.suptitle("Method selection: fixed interface gate, then a fitted shallow tree\n"
                 "Known inputs only; samples = image/model/priority scenarios, not independent image count", fontsize=18)
    fig.savefig(output / "method_selection_tree.png", dpi=160)
    fig.savefig(output / "method_selection_tree.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = ["#236B8E", "#D08A29", "#738B50"]
    for row, profile in enumerate(PROFILES):
        part = metrics[metrics.profile.eq(profile)]
        for col, metric in enumerate(("hit_rate", "mean_rank_regret")):
            ax = axes[row, col]
            for j, policy in enumerate(("tree", "goal_mode", "goal_rank")):
                values = part[part.policy.eq(policy)].set_index("objective").reindex(OBJECTIVES)[metric]
                ax.bar(np.arange(3) + (j - 1) * .24, values, .23, color=colors[j], label=policy)
            ax.set_xticks(range(3), OBJECTIVES)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Oracle hit rate (higher better)" if col == 0 else "Mean rank regret (lower better)")
            ax.set_title(profile)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncols=3, fontsize=9)
    fig.suptitle("Held-out original images: tree versus training-only goal baselines", fontsize=16)
    fig.savefig(output / "heldout_policy_comparison.png", dpi=160)
    plt.close(fig)


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame[columns].itertuples(index=False, name=None):
        cells = [f"{v:.6f}" if isinstance(v, float) else str(v).replace("|", "/") for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(archive, output, scratch, project=PROJECT):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require(protocol["input_sha256"] == EXPECTED_SHA and tuple(protocol["features"]) == FEATURES,
            "protocol/schema drift")
    require(protocol["profiles"] == {p: list(n) for p, n in PROFILES.items()}, "eligibility protocol drift")
    require(protocol["tie_absolute_tolerance"] == TOL, "tie protocol drift")
    frames, validation, _ = validate(Path(archive), Path(project))
    cases = prepare_cases(frames["model_correct"])
    train_i, test_i = grouped_split(cases, protocol["split"]["test_size"], protocol["split"]["random_state"])
    train, test = cases.iloc[train_i].copy(), cases.iloc[test_i].copy()
    require(train.group.nunique() + test.group.nunique() == cases.group.nunique(), "split partition")
    output, scratch = Path(output), Path(scratch)
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    split_manifest = cases[["dataset", "image_id", "group"]].drop_duplicates().sort_values(["dataset", "image_id"])
    split_manifest["split"] = np.where(split_manifest.group.isin(set(train.group)), "train", "test")
    split_bytes = split_manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (scratch / "split_assignments.csv").write_bytes(split_bytes)
    trees, exported, baselines = {}, {}, {}
    leaves, evaluations = [], []
    for profile, names in PROFILES.items():
        tree = fit_policy(train, profile, protocol)
        trees[profile], exported[profile] = tree, export_tree(tree)
        a, b = train[train.profile.eq(profile)], test[test.profile.eq(profile)]
        mode, rank = baseline_choices(a, names)
        baselines[profile] = dict(goal_mode=mode, goal_rank=rank)
        require(np.array_equal(tree.predict(features(b)), predict_export(exported[profile], features(b))),
                "portable JSON prediction mismatch")
        leaves.append(leaf_rows(tree, train, test, profile, protocol))
        for split, part in (("train", a), ("test", b)):
            predictions = {
                "tree": tree.predict(features(part)),
                "goal_mode": part.objective.map(mode).to_numpy(),
                "goal_rank": part.objective.map(rank).to_numpy(),
            }
            for objective in OBJECTIVES:
                mask = part.objective.eq(objective).to_numpy()
                selected = part.loc[mask]
                loss = loss_table(selected[list(names)], objective)
                for policy, predicted in predictions.items():
                    frame = selected[["dataset", "model", "image_id", "group", "objective", "profile", "label"]].copy()
                    frame["case_id"], frame["split"], frame["policy"] = frame.index, split, policy
                    frame["recommended"] = predicted[mask]
                    for key, value in selection_scores(loss, predicted[mask], names).items():
                        frame[key] = value
                    evaluations.append(frame)
    evaluation = pd.concat(evaluations, ignore_index=True)
    evaluation.to_csv(scratch / "per_scenario_evaluation.csv", index=False, lineterminator="\n")
    heldout = evaluation[evaluation.split.eq("test")]
    keys = ["profile", "objective", "policy"]
    def summarize(frame, extra):
        return frame.groupby(keys + extra, sort=True).agg(
            scenarios=("hit", "size"), original_images=("group", "nunique"),
            hit_rate=("hit", "mean"), mean_rank_regret=("rank_regret", "mean"),
            mean_native_gap=("native_gap", "mean"), mean_selected_loss=("selected_loss", "mean")
        ).reset_index()
    metrics = summarize(heldout, []).drop(columns=["mean_native_gap", "mean_selected_loss"])
    # Native gaps are interpreted within a dataset/model/objective, never ranked across tasks.
    cells = summarize(heldout, ["dataset", "model"])
    split_metrics = evaluation.groupby(["split"] + keys).agg(
        scenarios=("hit", "size"), hit_rate=("hit", "mean"), mean_rank_regret=("rank_regret", "mean")
    ).reset_index()
    leaf_table = pd.concat(leaves, ignore_index=True)
    ci = bootstrap_difference(heldout, protocol["bootstrap"]["resamples"])
    distributions = cases.groupby(["profile", "objective", "dataset", "model", "label"]).size().rename("scenarios").reset_index()
    tie_rows = []
    for profile, names in PROFILES.items():
        for objective in OBJECTIVES:
            p = cases[cases.profile.eq(profile) & cases.objective.eq(objective)]
            _, ties = oracle(loss_table(p[list(names)], objective), names)
            tie_rows.append(dict(profile=profile, objective=objective, scenarios=len(p), tied_oracle_rows=int((ties.sum(1) > 1).sum())))
    tables = dict(heldout_metrics=metrics, heldout_cells=cells, train_test_metrics=split_metrics,
                  leaves=leaf_table, paired_policy_intervals=ci, oracle_label_counts=distributions,
                  oracle_ties=pd.DataFrame(tie_rows))
    exception_rows = []
    for split, source in (("all", cases), ("train", train), ("test", test)):
        p = source[source.profile.eq("full") & source.objective.eq("faithfulness")
                   & source.dataset.eq("imagenet") & source.model.eq("densenet121")]
        exception_rows.append(dict(split=split, n=len(p), mean_auc_ig=float(p.ig.mean()),
            mean_auc_kernelshap=float(p.kernelshap.mean()), ks_minus_ig_mean=float((p.kernelshap - p.ig).mean()),
            ig_pairwise_wins=int((p.ig < p.kernelshap - TOL).sum()),
            kernelshap_pairwise_wins=int((p.kernelshap < p.ig - TOL).sum()),
            tree_recommendations=" / ".join(sorted(set(trees["full"].predict(features(p)))))))
    tables["ig_kernelshap_exception"] = pd.DataFrame(exception_rows)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g", lineterminator="\n")
    # Freeze all 36 known condition patterns for direct A/report use.
    recommendations = []
    for profile, tree in trees.items():
        part = cases[cases.profile.eq(profile)][["dataset", "model", "objective"]].drop_duplicates().sort_values(["dataset", "model", "objective"])
        part["profile"] = profile
        part["leaf"] = tree.apply(features(part))
        part["recommended"] = tree.predict(features(part))
        recommendations.append(part.merge(leaf_table[["profile", "leaf", "candidate_methods", "retain_multiple"]], on=["profile", "leaf"], validate="many_to_one"))
    conditions = pd.concat(recommendations, ignore_index=True)
    conditions.to_csv(output / "condition_recommendations.csv", index=False, lineterminator="\n")
    policy = dict(features=list(FEATURES), interface_gate=protocol["interface_gate"], eligible={k: list(v) for k, v in PROFILES.items()},
                  subtrees=exported, baselines=baselines)
    (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8", newline="\n")
    plot_outputs(trees, leaf_table, metrics, output)
    # Matplotlib's SVG writer uses platform newlines. Hash canonical LF bytes,
    # matching .gitattributes, so Windows checkout/staging preserves the receipt.
    svg = output / "method_selection_tree.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    text = "# Fixed-split method selection results\n\nNative gap columns belong to within-task cells only.\n\n"
    text += "## Held-out comparison\n\n" + markdown_table(metrics, keys + ["scenarios", "hit_rate", "mean_rank_regret"]) + "\n\n"
    text += "## Conditional recommendations\n\n" + markdown_table(conditions, ["profile", "dataset", "model", "objective", "recommended", "candidate_methods", "retain_multiple"]) + "\n\n"
    text += "## Leaf support\n\n" + markdown_table(leaf_table, ["profile", "leaf", "recommended", "train_scenarios", "train_original_images", "test_scenarios", "train_modal_share", "candidate_methods"]) + "\n"
    (output / "result_tables.md").write_text(text, encoding="utf-8", newline="\n")
    receipt = dict(date="2026-11-22", status="passed", integration_base=protocol["integration_base"],
        input_sha256=validation["zip_sha256"], input_validation=validation, cohort="model_correct",
        source_per_metric_rows=len(frames["model_correct"]), paired_image_model_rows=len(cases) // 6,
        scenario_rows=len(cases), train_scenarios=len(train), test_scenarios=len(test),
        train_original_images=int(train.group.nunique()), test_original_images=int(test.group.nunique()),
        group_overlap=0, feature_names=list(FEATURES), split_assignments_sha256=sha(split_bytes),
        tree_depths={p: int(t.get_depth()) for p, t in trees.items()},
        tree_leaves={p: int(t.get_n_leaves()) for p, t in trees.items()},
        condition_patterns=len(conditions), bootstrap_resamples=protocol["bootstrap"]["resamples"],
        portable_policy_predictions_match=True, raw_exports_tracked=False,
        versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
        python=platform.python_version(), code_sha256=sha(Path(__file__).read_bytes()),
        protocol_sha256=sha(PROTOCOL_PATH.read_bytes()),
        outputs={p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file() and p.name not in ("receipt.json", "protocol.json", "verification.json")},
        scope=protocol["limits"])
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: receipt[k] for k in ("status", "paired_image_model_rows", "train_original_images", "test_original_images", "tree_depths", "tree_leaves")}, ensure_ascii=False))
    print(metrics[keys + ["hit_rate", "mean_rank_regret"]].to_string(index=False))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PROJECT / "analysis/d_method_selection_20261122")
    parser.add_argument("--scratch", type=Path, default=PROJECT / "results/d_method_selection_20261122")
    args = parser.parse_args()
    run(args.zip, args.out, args.scratch)


if __name__ == "__main__":
    main()
