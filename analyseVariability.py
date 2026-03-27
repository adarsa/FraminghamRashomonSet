import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

from utils import compute_calibration_metrics, extract_decision_path, count_contradictions, tree_stats


def load_config():
    """Rashomon output root — must match generateSet.py (data/rashomonSet)."""
    return os.path.join("data", "rashomonSet")


def load_eval_data():
    df = pd.read_csv("data/framingham_preproc.csv", index_col=0)
    y = df["TenYearCHD"]
    X = df.drop(columns=["TenYearCHD"]).fillna(df.drop(columns=["TenYearCHD"]).median())
    feature_names = X.columns.tolist()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, y_test, feature_names


def get_eps_dirs(output_root):
    if not os.path.exists(output_root):
        raise FileNotFoundError(f"Output directory not found: {output_root}")
    return sorted(
        d for d in os.listdir(output_root)
        if d.startswith("eps_") and os.path.isdir(os.path.join(output_root, d))
    )


def parse_epsilon(eps_dir_name):
    return float(eps_dir_name.split("_", 1)[1])


def load_trees_from_dir(eps_path):
    tree_files = sorted(
        f for f in os.listdir(eps_path)
        if f.startswith("tree_") and f.endswith(".pkl")
    )
    trees = []
    for file_name in tree_files:
        with open(os.path.join(eps_path, file_name), "rb") as f:
            trees.append(pickle.load(f))
    return trees


def analyse_epsilon(trees, X_test, y_test, feature_names):
    if not trees:
        return {
            "feature_importances_df": pd.DataFrame(columns=feature_names),
            "tree_metrics_df": pd.DataFrame(
                columns=["tree_index", "auc", "brier", "ece", "mce", "n_nodes", "n_leaves"]
            ),
            "contradiction_details": [],
            "instance_rules": [],
            "result": {
                "rashomon_size": 0,
                "auc_mean": np.nan,
                "auc_std": np.nan,
                "brier_mean": np.nan,
                "ece_mean": np.nan,
                "mce_mean": np.nan,
                "tree_nodes_mean": np.nan,
                "tree_nodes_std": np.nan,
                "leaf_count_mean": np.nan,
                "leaf_count_std": np.nan,
                "contradictions_mean": np.nan,
                "feature_importance_variance_mean": np.nan,
            },
        }

    aucs, briers, eces, mces = [], [], [], []
    n_nodes_list, n_leaves_list = [], []
    feature_importances = []
    tree_rows = []

    for idx, tree in enumerate(trees):
        probs = tree.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        ece, mce = compute_calibration_metrics(y_test, probs)
        n_nodes, n_leaves, _ = tree_stats(tree)

        aucs.append(auc)
        briers.append(brier)
        eces.append(ece)
        mces.append(mce)
        n_nodes_list.append(n_nodes)
        n_leaves_list.append(n_leaves)
        feature_importances.append(tree.feature_importances_)

        tree_rows.append(
            {
                "tree_index": idx,
                "auc": auc,
                "brier": brier,
                "ece": ece,
                "mce": mce,
                "n_nodes": n_nodes,
                "n_leaves": n_leaves,
            }
        )

    importances = np.array(feature_importances)
    importances_var = importances.var(axis=0)

    contradictions_per_instance = []
    contradiction_details = []
    instance_rules = []

    for i in range(len(X_test)):
        x_i = X_test.iloc[i:i + 1].values
        rules_dicts = [
            dict((f, d) for f, d, _ in extract_decision_path(t, x_i, feature_names))
            for t in trees
        ]
        contradictions_per_instance.append(
            count_contradictions(rules_dicts, feature_names=feature_names)
        )
        contradiction_details.append(rules_dicts)
        instance_rules.append(
            [extract_decision_path(t, x_i, feature_names) for t in trees]
        )

    return {
        "feature_importances_df": pd.DataFrame(importances, columns=feature_names),
        "tree_metrics_df": pd.DataFrame(tree_rows),
        "contradiction_details": contradiction_details,
        "instance_rules": instance_rules,
        "result": {
            "rashomon_size": len(trees),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "brier_mean": float(np.mean(briers)),
            "ece_mean": float(np.mean(eces)),
            "mce_mean": float(np.mean(mces)),
            "tree_nodes_mean": float(np.mean(n_nodes_list)),
            "tree_nodes_std": float(np.std(n_nodes_list)),
            "leaf_count_mean": float(np.mean(n_leaves_list)),
            "leaf_count_std": float(np.std(n_leaves_list)),
            "contradictions_mean": float(np.mean(contradictions_per_instance)),
            "feature_importance_variance_mean": float(np.mean(importances_var)),
            "feature_importance_variance": importances_var.tolist(),
        },
    }


def main():
    output_root = load_config()
    X_test, y_test, feature_names = load_eval_data()
    eps_dirs = get_eps_dirs(output_root)

    summary_rows = []
    for eps_dir in eps_dirs:
        eps_path = os.path.join(output_root, eps_dir)
        eps = parse_epsilon(eps_dir)
        trees = load_trees_from_dir(eps_path)

        analysis = analyse_epsilon(trees, X_test, y_test, feature_names)

        analysis["feature_importances_df"].to_csv(
            os.path.join(eps_path, "feature_importances.csv"),
            index=False,
        )
        analysis["tree_metrics_df"].to_csv(
            os.path.join(eps_path, "tree_metrics.csv"),
            index=False,
        )
        with open(os.path.join(eps_path, "contradictions.pkl"), "wb") as f:
            pickle.dump(analysis["contradiction_details"], f)
        with open(os.path.join(eps_path, "instance_rules.pkl"), "wb") as f:
            pickle.dump(analysis["instance_rules"], f)

        result_row = {"epsilon": eps}
        result_row.update(analysis["result"])
        summary_rows.append(result_row)
        print(f"Analysed {eps_dir}: {analysis['result']['rashomon_size']} trees")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_root, "variability_summary.csv"),
        index=False,
    )
    print(f"Saved variability summary to {os.path.join(output_root, 'variability_summary.csv')}")


if __name__ == "__main__":
    main()
