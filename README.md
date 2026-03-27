# Framingham Rashomon Set

This project builds a set of near-optimal decision trees (a Rashomon set) on the Framingham dataset, analyses variability across that set, and provides user-based tree matching for human-readable explanations.

## Project Files

- `optimiseAUC.py`  
  Trains candidate decision trees, computes AUC for each tree, and stores:
  - `data/candidate_trees.pkl`
  - `data/candidate_aucs.pkl`

- `generateSet.py`  
  Reads candidate trees/AUCs and creates epsilon-specific Rashomon folders (`eps_*`) under `data/rashomonSet`.

- `analyseVariability.py`  
  Reads each `eps_*` folder under `data/rashomonSet`, computes variability metrics, and writes:
  - per-epsilon files (`feature_importances.csv`, `tree_metrics.csv`, `contradictions.pkl`, `instance_rules.pkl`)
  - root summary (`data/rashomonSet/variability_summary.csv`)

- `UserMatchedTrees.ipynb`  
  Reads `config/userconfig.json` and prints the best-matched tree per user.

## Config

- `config/threshold.json`
  - `epsilons`: list of epsilon thresholds (used by `generateSet.py`)

- `config/userconfig.json`
  - user-to-feature mapping used in user-based matching


## Order of Running

Run the pipeline in this order from the project root:

1. Generate candidate trees and AUCs
   - `python optimiseAUC.py`
2. Build epsilon-wise Rashomon sets
   - `python generateSet.py`
3. Compute variability metrics from generated `eps_*` folders
   - `python analyseVariability.py`
4. User-oriented model notebook
   - `UserMatchedTrees.ipynb`
   - set `EPSILON_TO_USE` and run all cells

## Notes

- Input dataset expected at `data/framingham_preproc.csv`.
- `UserMatchedTrees.ipynb` reads from `data/rashomonSet/eps_<epsilon>/` and expects `feature_importances.csv` there (created by `analyseVariability.py`).
- Use the same Python environment for all steps to avoid package/version mismatch.
