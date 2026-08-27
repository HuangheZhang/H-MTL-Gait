# H-MTL: A Multi-Task Learning Experimental Framework for Gait Assessment

## Summary

H-MTL is a hierarchical multi-task learning framework for comprehensive gait assessment from wearable inertial signals. It jointly addresses ten heterogeneous tasks spanning pathological screening, multi-granularity condition classification, clinical score regression, and demographic estimation within a unified training and evaluation workflow. The framework provides hierarchy-aware supervision and multiple MTL configurations for systematic comparison.

This release accompanies the unpublished manuscript **“Hierarchical Multi-Task Learning for Comprehensive Gait Assessment Using Wearable Inertial Sensors.”**

**Research use only. This software is not intended for clinical decision-making and must not be used to make clinical decisions.**

## Methods included from the manuscript

Stage 2 of the manuscript names Hard Sharing, MMoE, PLE, PCGrad, CAGrad, Uncertainty Weighting, and DWA. The public method map is machine-readable in `configs/paper_mtl_methods.yaml`. Its `model_type` and `gradient_method` fields keep the two selection dimensions explicit.

| Public experiment ID | Base model | Optimization strategy | Kind | Exact CLI selection |
|---|---|---|---|---|
| `hard_sharing` | Hard Sharing | none | model family | `--experiment hard_sharing` |
| `mmoe` | MMoE | none | model family | `--experiment mmoe` |
| `ple` | PLE | none | model family | `--experiment ple` |
| `pcgrad` | Hard Sharing | PCGrad | optimization strategy | `--experiment pcgrad` |
| `cagrad` | Hard Sharing | CAGrad | optimization strategy | `--experiment cagrad` |
| `uncertainty_weighting` | Hard Sharing | Uncertainty Weighting | optimization strategy | `--experiment uncertainty_weighting` |
| `dwa` | Hard Sharing | DWA | optimization strategy | `--experiment dwa` |
| `mmoe_dwa` | MMoE | DWA | combination | `--experiment mmoe_dwa` |
| `mmoe_cagrad` | MMoE | CAGrad | combination | `--experiment mmoe_cagrad` |

All configurations use the same data interface, task definitions, training protocol, evaluation metrics, and result format, enabling direct comparison under a consistent experimental workflow.

## Repository tree

```text
.
├── README.md
├── LICENSE
├── NOTICE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── CHECKSUMS.sha256
├── checkpoints/
│   └── hmtl_mmoe_dwa_example.pt
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── paper_mtl_methods.yaml
│   └── example_checkpoint.yaml
├── scripts/
│   ├── run_mtl_experiments.py
│   ├── inspect_checkpoint.py
│   └── infer.py
├── src/
│   ├── data/{__init__.py,dataset.py,augmentation.py,normalization.py}
│   ├── models/{__init__.py,deep_models.py,mtl_models.py,gradient_methods.py,losses.py,hierarchical_mtl.py}
│   ├── trainers/{__init__.py,mtl_trainer_v2.py}
│   ├── evaluators/{__init__.py,metrics.py}
│   └── utils/{__init__.py,helpers.py}
└── tests/
    ├── test_fold_normalization.py
    └── test_paper_mtl_methods.py
```

## Execution flow

```text
CLI experiment ID
  -> experiment registry: model_type + gradient_method + method kwargs
  -> independent model and optimization factories
  -> raw window loading and subject-level splits
  -> training-fold-only standardization
  -> training, validation, checkpointing, and evaluation
  -> per-fold results and aggregate summaries
```

`inspect_checkpoint.py` reports the file hash and performs strict loading. `infer.py` runs all task outputs with zero, seeded random, or local NumPy input.

## Installation and tests

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_mtl_experiments.py --help
python scripts/inspect_checkpoint.py
python scripts/infer.py
python scripts/infer.py --random --batch-size 2 --seed 42
```

## Data source

Dataset files are not distributed in this repository. The framework uses the open clinical gait dataset described by the following resources:

- Dataset article: https://doi.org/10.1038/s41597-025-05959-w
- Dataset download DOI: https://doi.org/10.6084/m9.figshare.28806086
- Dataset code repository: https://github.com/CyrilVoisard/dataset_gait_1

The dataset article is authored by **Cyril Voisard, Rémi Barrois, Nicolas de l’Escalopier, Nicolas Vayatis, Pierre-Paul Vidal, Alain Yelnik, Damien Ricard, and Laurent Oudre**. The dataset contains **260 participants, four sensors, and signals sampled at 100 Hz**.

## Acknowledgements

We prominently thank Cyril Voisard, Rémi Barrois, Nicolas de l’Escalopier, Nicolas Vayatis, Pierre-Paul Vidal, Alain Yelnik, Damien Ricard, and Laurent Oudre for creating and openly sharing **“A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts,” Scientific Data 12, 1674 (2025)**. Their open sharing made this study possible.

We also thank all study participants and the maintainers of PyTorch, scikit-learn, NumPy, and pandas. The dataset article is available at https://doi.org/10.1038/s41597-025-05959-w.

## Expected data layout and channel order

Place separately obtained processed files under the configured root:

```text
data/
└── {healthy,neuro,ortho}/
    └── PATHOLOGY/
        └── SUBJECT/
            └── TRIAL/
                ├── TRIAL_processed_data.txt
                └── TRIAL_meta.json
```

The default input uses sensor positions `HE`, `LB`, `LF`, `RF` in that order. Within each sensor, signals are ordered `acc`, `freeacc`, `gyr`; each signal uses axes `X`, `Y`, `Z`. The resulting 36-channel sequence is:

```text
HE_Acc_X, HE_Acc_Y, HE_Acc_Z, HE_FreeAcc_X, ..., HE_Gyr_Z,
LB_Acc_X, ..., LB_Gyr_Z,
LF_Acc_X, ..., LF_Gyr_Z,
RF_Acc_X, ..., RF_Gyr_Z
```

Windows contain 200 samples at 100 Hz with a 100-sample stride.

## Fold-local preprocessing

`GaitDataLoader` returns raw windows. Normalization occurs only after split membership is fixed:

1. Outer train/test subjects are selected with pathology stratification.
2. Outer-training subjects are split deterministically into training and validation subjects.
3. `FoldStandardizer` replaces non-finite values with zero and fits channel-wise statistics on training windows only.
4. Validation and test windows reuse the unchanged training statistics.
5. Newly saved MTL checkpoints include a serializable preprocessing state with `scope: training_fold_only`.

Subject-wise, trial-wise, STL, and the public `create_dataloaders` path follow the same train-only fitting rule. Passing `normalize=True` to `prepare_dataset` raises an error to prevent pre-split fitting.

## Training examples

A complete recorded-protocol example is:

```bash
python scripts/run_mtl_experiments.py \
  --data_path /path/to/data \
  --experiment mmoe_dwa \
  --folds 10 \
  --split_mode subject_wise \
  --use_mse_loss \
  --seed 42 \
  --epochs 80 \
  --lr 0.0003 \
  --batch_size 32 \
  --dropout 0.35 \
  --weight_decay 0.0005 \
  --warmup_epochs 8 \
  --patience 20 \
  --label_smoothing 0.1 \
  --aug_p 0.5 \
  --scheduler CosineAnnealing \
  --reg_weight 2.0 \
  --head_hidden_dim 64 \
  --monitor_metric val_binary_accuracy \
  --monitor_mode max \
  --output_dir results/mmoe_dwa_seed42
```

Switch only the experiment ID to run another published configuration:

```bash
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment hard_sharing
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment mmoe
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment ple
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment pcgrad
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment cagrad
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment uncertainty_weighting
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment dwa
python scripts/run_mtl_experiments.py --data_path /path/to/data --experiment mmoe_cagrad
```

Alternatively, set `HMTL_DATA_PATH=/path/to/data`. Aggregate reporting should use all requested outer folds.

## Task outputs

| Key | Shape | Meaning |
|---|---:|---|
| `binary` | `(B,2)` | logits: healthy / pathological |
| `coarse` | `(B,3)` | logits: healthy / neurological / orthopaedic |
| `fine` | `(B,8)` | logits: HS / CVA / PD / CIPN / RIL / KOA / HOA / ACL |
| `regression` | `(B,1)` | normalized pathology-specific clinical score |
| `vga_class` | `(B,5)` | logits for visual gait assessment classes 0–4 |
| `vga_regression` | `(B,1)` | visual gait assessment score divided by 4 |
| `gender` | `(B,2)` | logits under the M/F training encoding |
| `age` | `(B,1)` | age divided by 100 |
| `tug` | `(B,1)` | Timed Up and Go value divided by 100 |
| `neuro_fine` | `(B,4)` | logits: CVA / PD / CIPN / RIL |

Classification outputs are logits, and regression outputs are normalized raw values.

## Example pretrained weight

The repository includes one example pretrained weight for loading and inference demonstrations:

```text
path: checkpoints/hmtl_mmoe_dwa_example.pt
method: mmoe_dwa
SHA-256: 1eca97d7390988a7c96c411fed446ed2fd8038a981b51d8383cb1b008f7bd89a
```

Use `scripts/inspect_checkpoint.py` to read its metadata and `scripts/infer.py` to run example inputs.

## Release scope

This public repository contains the H-MTL framework, manuscript method map, tests, configuration, and one example pretrained weight. It does not distribute dataset files or generated experiment outputs. The Apache-2.0 software license does not grant rights to the dataset; dataset access and terms remain separate.

## Citation

Citation metadata for the accompanying unpublished manuscript is in `CITATION.cff`:

```bibtex
@article{wu_hmtl_gait_2026,
  title  = {Hierarchical Multi-Task Learning for Comprehensive Gait Assessment Using Wearable Inertial Sensors},
  author = {Wu, Peng and Dong, Huashuo and Ding, Ziyun and Gao, Tianyuan and Kong, Weixin and Liu, Yitian and Song, Rui and Zhang, Huanghe},
  year   = {2026},
  note   = {Unpublished manuscript}
}
```

## License

The software is licensed under the Apache License 2.0; see `LICENSE` and `NOTICE`. Dataset files are not distributed and remain under their separate terms.
