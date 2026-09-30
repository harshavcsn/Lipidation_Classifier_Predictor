# Lipidation Classifier Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harshavcsn/Lipidation_Classifier_Predictor/blob/main/predict_colab.ipynb)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predicts whether a **tripeptide** forms a `fiber`, `droplet` or `MetaStable (MS)`
assembly. Each residue is described by AAindex properties compressed with PCA,
and three models (SVM, elastic-net logistic regression, random forest) vote on
the final class.

Sequences are written with a leading `G` placeholder followed by the
tripeptide, e.g. `GAAC` for the tripeptide `AAC`. The `G` is not used by the
models.

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/harshavcsn/Lipidation_Classifier_Predictor.git
cd Lipidation_Classifier_Predictor
pip install -r requirements.txt          # prediction only
pip install -r requirements-train.txt    # also needed to train your own model
```

---

## Predict

**Google Colab (no installation):** click **Open in Colab** above and run all cells.

**Command line:**

```bash
python predict.py GAAC GKLV GGWV
python predict.py --input new_predict.csv --output predictions.csv
```

```
sequence SVM_RBF_pred ElasticNetLogistic_pred RandomForest_pred consensus_pred consensus_agreement
    GAAC        fiber                   fiber             fiber          fiber                 3/3
    GKLV      droplet                 droplet           droplet        droplet                 3/3
    GGWV      droplet                 droplet           droplet        droplet                 3/3
```

`consensus_agreement` shows how many models agree. Treat `2/3` as a
low-confidence call.

**Notebook:** open [`predict.ipynb`](predict.ipynb) in Jupyter or VS Code.

**Python:**

```python
from predict import predict
predict(["GAAC", "GKLV"])
```

Input files can list one sequence per line, or be a CSV with a `sequence` column.

---

## Model performance

Cross-validated on 64 labeled sequences (4-fold × 20 repeats).

| Model | Macro-F1 | Balanced accuracy | Accuracy |
|---|---|---|---|
| **SVM (RBF kernel)** | **0.536** | **0.562** | **0.773** |
| Elastic-net logistic regression | 0.506 | 0.532 | 0.733 |
| Random forest | 0.506 | 0.527 | 0.724 |

---

## Train your own model

1. Prepare a CSV with **no header** and two columns, `sequence,label`, like
   [`list3.csv`](list3.csv).
2. Run the pipeline. This quick run tests 20 settings per model on 8 CPUs; set
   `LIPML_MAX_COMBOS=none` for the full search.

   ```bash
   LIPML_DATA=my_data.csv LIPML_OUT=results_my_run LIPML_MAX_COMBOS=20 ./run_all.sh 8
   ```

3. Predict with your new models:

   ```bash
   python predict.py GAAC --models-dir results_my_run/final \
                          --pca-table results_my_run/features/pca_table.csv
   ```

All settings are in [`config.py`](config.py). Naive Bayes can be switched on
in [`models.py`](models.py).

---

## Files

| File | Purpose |
|---|---|
| `predict.py` | Predict sequences with the trained models |
| `predict.ipynb`, `predict_colab.ipynb` | Prediction notebooks (Jupyter / VS Code, Google Colab) |
| `list3.csv` | Labeled training data |
| `new_predict.csv` | Example sequences to predict |
| `run_all.sh` | Runs the full training pipeline (steps 01–06) |
| `01_build_features.py` | Builds AAindex PCA features |
| `02_make_jobs.py` | Lists every hyperparameter combination to test |
| `run_search.sh`, `run_search.slurm`, `run_shard.py` | Runs the search in parallel (local or SLURM) |
| `03_aggregate.py` | Collects search results and picks the best settings |
| `04_evaluate_best.py` | Confusion matrices and per-class reports |
| `05_ablation.py` | Feature importance and feature-block ablation |
| `06_final_train_predict.py` | Trains the final models on all data |
| `07_export_for_origin.py` | Optional: exports figure data for Origin |
| `config.py`, `features.py`, `models.py`, `cv.py` | Settings and shared code |
| `results_4block_pca/` | Trained models, results and figures from this study |

---

## Citation

This work is part of a manuscript currently under review. Citation details will
be added upon publication.

## License

[MIT](LICENSE)
