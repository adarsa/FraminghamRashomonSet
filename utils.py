import numpy as np
from sklearn.tree import _tree

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve


def compute_calibration_metrics(y_true, probs, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins)
    ece = np.mean(np.abs(prob_true - prob_pred))
    mce = np.max(np.abs(prob_true - prob_pred))
    return ece, mce


def extract_decision_path(tree, x, feature_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    node_indicator = tree.decision_path(x)
    node_index = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    rules = []

    for node_id in node_index:
        if feature[node_id] == _tree.TREE_UNDEFINED:
            continue

        feat_idx = feature[node_id]
        feat_name = feature_names[feat_idx]

        direction = "le" if x[0, feat_idx] <= threshold[node_id] else "gt"

        rules.append((feat_name, direction, node_id))

    return rules


def count_contradictions(rules_list, feature_names=None):
    contradictory_features = set()

    if feature_names is None:
        feature_names = sorted(
            {
                feat
                for rules in rules_list
                for feat in rules.keys()
            }
        )

    for feat in feature_names:
        dirs = [r.get(feat) for r in rules_list if feat in r]
        if len(set(dirs)) > 1:
            contradictory_features.add(feat)

    return len(contradictory_features)


def tree_stats(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    is_leaf = (children_left == _tree.TREE_LEAF)
    n_leaves = np.sum(is_leaf)
    gini_nodes = tree.tree_.impurity.tolist()

    return n_nodes, n_leaves, gini_nodes
