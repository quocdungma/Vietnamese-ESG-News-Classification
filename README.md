# Vietnamese ESG News Classification

This repository provides research code for evaluating six article-level ESG classification tasks on Vietnamese financial news.

## Core scripts

1. `scripts/01_expert_agreement.py` — expert agreement.
2. `scripts/02_expert_adjudication.py` — expert adjudication assessment.
3. `scripts/03_train_phobert.py` — PhoBERT and lexical baselines.
4. `scripts/04_train_xlmr.py` — XLM-RoBERTa comparison.
5. `scripts/05_random_seed_evaluation.py` — repeated seed experiments.

The scripts are exported from Colab notebooks. They still require environment/path adaptation and have not been verified as independent command-line programs.

## Data

Original restricted inputs in Google Drive: `returned_expert_A_annotation.xlsx` and `returned_expert_B_annotation.xlsx`. Their full workbooks contain publisher news text and expert notes and are **not distributed publicly**. The release contains only minimal label-only CSV derivatives under `data/annotations/`; these are expert-return records, not adjudicated final gold labels.

The model experiments use the broader labeled Vietstock/CafeF corpus (2,800 articles). The expert-return data alone are **not sufficient for model training**.

## Scope

Training: Vietstock. Target-source validation and held-out testing: CafeF. The research does not claim zero-shot adaptation.

The collection/crawling and initial annotation preparation notebooks are intentionally omitted.
