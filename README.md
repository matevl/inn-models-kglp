# INN-MODELS-KGLP: Interval Neural Networks for Knowledge Graph Link Prediction

A implementation of Interval Neural Networks (INN) for link prediction in knowledge graphs.

## Overview & Project Goal

INN-MODELS-KGLP is a knowledge graph link prediction framework that learns entity and relation representations as **intervals** rather than points. This approach provides more expressive representations by capturing uncertainty and relation-specific variations in embedding space.

The explicit goal of this project is to **compare different approaches of INN** based on recent literature, and to evaluate their performance against state-of-the-art graph models (like LightGCN) across standard reference datasets.

**Link prediction** is the task of predicting missing relations between entities in a knowledge graph. Given a triple `(head, relation, ?)`, the model predicts the tail entity.

### Key Features

- **Interval-based embeddings**: Entities and relations are represented as `(center, radius)` pairs
- **Differentiable interval arithmetic**: Forward propagation through interval operations
- **Multiple model types**: Train and compare interval MLPs (`inn_ours_mlp`) against baseline baselines (`inn_lightgcn`)
- **Modular Configuration**: Driven by Hydra, allowing easy hyperparameter sweeps without parsing huge CLI vectors
- **TensorBoard integration**: Track experimental results (Loss, Radius sizes, etc.) automatically

## Installation & Setup

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- Optional: [uv](https://github.com/astral-sh/uv) (for faster dependency installation)

```bash
# Clone the repository
git clone <repository_url>
cd inn-models-kglp

# Install environment and download datasets (FB15k-237 and WN18RR)
make install
make setup
```

## Configuration & Usage (Hydra)

This project has migrated to `hydra-core` for robust experiment tracking. Instead of modifying argparse flags, you launch `src/main.py` with overrides or by creating new YAML files.

### Configuration Structure
All configurations live inside the `configs/` directory:
- `configs/config.yaml`: The main configuration. Contains defaults for training, evaluation, and I/O logic.
- `configs/dataset/`: Definitions for datasets (`fb15k237.yaml`, `wn18rr.yaml`).
- `configs/model/`: Model architectures and hyperparams (`inn_ours_mlp.yaml`, `inn_lightgcn.yaml`).
- `configs/experiment/`: Pre-configured environments that combine models, datasets, and parameters (`inn_ours_mlp_v1.yaml`).

This allows you to create an experiment YAML, tweak a specific property like `hidden_layers: [500, 2000, 500]`, and run it without touching the CLI every time.

### Quickstart CLI Commands

```bash
# Initialize and train INN_Ours_MLP model with standard config
python src/main.py +experiment=inn_ours_mlp_v1

# Change Dataset to WN18RR and Override Batch Size on the fly
python src/main.py +experiment=inn_ours_mlp_v1 dataset=wn18rr training.batch_size=512

# Run LightGCN baseline instead
python src/main.py mode=train model=inn_lightgcn

# Evaluation Mode
python src/main.py mode=test model=inn_ours_mlp evaluation.split=test
```

## Datasets

The framework uses the Hugging Face CLI to securely download and symlink standard KGLP benchmarks:
- **FB15k-237**: Heavily used knowledge graph benchmark (14,541 Entities, 237 Relations).
- **WN18RR**: Standard WordNet subset designed to test link prediction.

## Evaluation Metrics

- **MRR** (Mean Reciprocal Rank): Average rank of the correct answer
- **Hits@K**: Percentage of queries where the correct answer is in the top-K predictions

## Reproducibility

All experiments use TensorBoard for tracking. Metrics such as `Radius/Max`, `Radius/Mean`, and standard loss are logged automatically in the `runs/` folder.

```bash
# View results:
tensorboard --logdir runs
```

## Cleanup

To reset everything (cache, temporary files):
```bash
make clean
```

To remove the virtual environment and datasets entirely:
```bash
make uninstall
```

## Citation

This project is built to test and compare Interval Network structures. If you use this work, please cite:

```bibtex
@inproceedings{diop:hal-05511161,
  TITLE = {{Extraction diff{\'e}rentiable d'ensemble de motifs d'intervalles}},
  AUTHOR = {Diop, Lamine and Plantevit, Marc},
  URL = {https://hal.science/hal-05511161},
  BOOKTITLE = {{Extraction et Gestion de Connaissances (EGC'26)}},
  ADDRESS = {Anglet (France), France},
  YEAR = {2026},
  MONTH = Jan,
  PDF = {https://hal.science/hal-05511161v1/file/egc26%20%286%29.pdf},
  HAL_ID = {hal-05511161},
  HAL_VERSION = {v1},
}
```

This work builds upon [Schlichtkrull et al. (2018)](https://arxiv.org/abs/1703.06103) for knowledge graph representation learning:

```bibtex
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Berg, Rianne van den and Titov, Ivan and Welling, Max},
  booktitle={European semantic web conference},
  pages={593--607},
  year={2018},
  organization={Springer}
}
```

This project also integrates and implements components from [CompGCN](https://github.com/malllabiisc/CompGCN):
```bibtex
@article{vashishth2019composition,
  title={Composition-based multi-relational graph convolutional networks},
  author={Vashishth, Shikhar and Sanyal, Soumya and Nitin, Vikram and Talukdar, Partha},
  journal={arXiv preprint arXiv:1911.03082},
  year={2019}
}
```
