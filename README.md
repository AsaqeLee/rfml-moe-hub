# rfml-moe-hub (fork)

> **Fork notice:** This repository is a fork of [`r4d10n/rfml-moe-hub`](https://github.com/r4d10n/rfml-moe-hub). Upstream authors own the original research code, experiments, and narrative. Do not treat AsaqeLee as the original author of this hub.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research hub for **RF-based drone detection and classification** using classical features, deep learning, and mixture-of-experts (MoE) style architectures. The upstream project consolidates experiments across multiple datasets and model families (statistical features, spectrogram CNNs/transformers, raw-IQ models, and MoE routing).

## Overview

Radio-frequency (RF) emissions from drone links can be used for passive detection and classification. Upstream documentation compares hand-crafted statistical features against deep models and discusses sample-size regimes where each approach tends to dominate. Quantitative tables and phase-by-phase experiment notes in this tree come from the upstream research write-up; reproduce results locally before citing them.

For the full experimental narrative, architecture diagrams, and result tables, see the preserved upstream README content in git history and accompanying files such as [`EXPERIMENTS.md`](EXPERIMENTS.md) and [`INSTALL.md`](INSTALL.md).

## Upstream

- Parent repository: https://github.com/r4d10n/rfml-moe-hub
- This fork: https://github.com/AsaqeLee/rfml-moe-hub

## Requirements

Typical stack (see `INSTALL.md` for authoritative steps):

- Python 3.10+
- PyTorch 2.x (CUDA or ROCm builds as available on your machine)
- Dataset downloads and preprocessing scripts under `datasets/` / `preprocessing/`

## Getting started

Follow [`INSTALL.md`](INSTALL.md) for environment setup, then consult [`EXPERIMENTS.md`](EXPERIMENTS.md) for phase-specific training and evaluation commands. High-level layout:

```text
models/
preprocessing/
datasets/
experiments/
checkpoints/
results/
visualizations/
docs/
```

## Status / limitations

Fork for personal study and experimentation. Claims of accuracy, parameter counts, and “novel architecture” status belong to the upstream authors’ documentation. This fork does not assert original authorship of those results.

## License

MIT (as shipped with the upstream project). See [LICENSE](LICENSE) and respect upstream copyright notices.
