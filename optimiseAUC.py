## Determine the best AUC for framingham dataset
import numpy as np
import pandas as pd
import os
import pickle   
import hashlib
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

np.random.seed(42)


df = pd.read_csv("data/framingham_preproc.csv", index_col=0)

y = df["TenYearCHD"]
X = df.drop(columns=["TenYearCHD"])
X = X.fillna(X.median())
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def tree_hash(tree):
    t = tree.tree_
    payload = np.concatenate([
        t.feature.astype(np.int64),
        t.threshold.astype(np.float64).view(np.int64),
        t.children_left.astype(np.int64),
        t.children_right.astype(np.int64),
        t.value.ravel().astype(np.float64).view(np.int64),
    ])
    return hashlib.sha256(payload.tobytes()).hexdigest()


# Train 1000candidate trees
n_trees = 10000
candidate_trees = []
candidate_aucs = []
tree_hashlist = []

for seed in range(n_trees):
    tree = DecisionTreeClassifier(
        max_depth=np.random.randint(2, 7),
        min_samples_leaf=np.random.randint(5, 40),
        random_state=seed
    )

    tree.fit(X_train, y_train)

    probs = tree.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    hash = tree_hash(tree)
    if hash in tree_hashlist:
        print(f"Duplicate tree skipped: {seed}")
    else:
        tree_hashlist.append(hash)
        candidate_trees.append(tree)
        candidate_aucs.append(auc)

# Store all candidate trees in data folder
os.makedirs("data", exist_ok=True)
trees_output_path = os.path.join("data", "candidate_trees.pkl")
with open(trees_output_path, "wb") as f:
    pickle.dump(candidate_trees, f)

aucs_output_path = os.path.join("data", "candidate_aucs.pkl")
with open(aucs_output_path, "wb") as f:
    pickle.dump(candidate_aucs, f)

best_auc = max(candidate_aucs)
print(f"Best AUC: {best_auc:.3f}")
print(f"Saved {len(candidate_trees)} candidate trees to {trees_output_path}")
print(f"Saved {len(candidate_aucs)} candidate AUCs to {aucs_output_path}")

#random forest AUC: 0.693