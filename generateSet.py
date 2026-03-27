import os
import pickle
import json
import numpy as np
import pandas as pd


# Load all candidate trees from pickle
with open(os.path.join("data", "candidate_trees.pkl"), "rb") as f:
    candidate_trees = pickle.load(f)

# Load candidate AUCs corresponding to candidate trees
with open(os.path.join("data", "candidate_aucs.pkl"), "rb") as f:
    candidate_aucs = pickle.load(f)

best_auc = max(candidate_aucs)

with open("config/threshold.json", "r", encoding="utf-8") as f:
    eps_config = json.load(f)
    epsilons = np.array(eps_config.get("epsilons", []), dtype=float)

# Always store generated Rashomon sets under data/rashomonSet
output_root = os.path.join("data", "rashomonSet")

if len(candidate_trees) != len(candidate_aucs):
    raise ValueError(
        "Mismatch between number of candidate trees and candidate AUCs."
    )

all_results = []


os.makedirs(output_root, exist_ok=True)

for eps in epsilons:

    selected_pairs = [
        (idx, tree, auc)
        for idx, (tree, auc) in enumerate(zip(candidate_trees, candidate_aucs))
        if auc >= (best_auc - eps)
    ]

    print(f"Epsilon {eps:.3f} -> size {len(selected_pairs)}")

    eps_dir = os.path.join(output_root, f"eps_{eps:.3f}")
    os.makedirs(eps_dir, exist_ok=True)

    manifest_rows = []
    for saved_idx, (orig_idx, tree, auc) in enumerate(selected_pairs):
        with open(os.path.join(eps_dir, f"tree_{saved_idx}.pkl"), "wb") as f:
            pickle.dump(tree, f)

        manifest_rows.append(
            {
                "saved_tree_index": saved_idx,
                "candidate_tree_index": orig_idx,
                "candidate_auc": auc,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(os.path.join(eps_dir, "tree_manifest.csv"), index=False)

    auc_values = manifest_df["candidate_auc"].to_numpy() if not manifest_df.empty else np.array([])

    result = {
        "epsilon": float(eps),
        "rashomon_size": int(len(selected_pairs)),
        "candidate_auc_mean": float(np.mean(auc_values)) if auc_values.size else np.nan,
        "candidate_auc_min": float(np.min(auc_values)) if auc_values.size else np.nan,
        "candidate_auc_max": float(np.max(auc_values)) if auc_values.size else np.nan,
    }

    all_results.append(result)

pd.DataFrame(all_results).to_csv(os.path.join(output_root, "set_summary.csv"), index=False)
print(f"Saved Rashomon sets to {output_root}")


