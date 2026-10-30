# Vietnamese ESG News Classification

Reproducibility materials for article-level classification of Vietnamese financial news from Vietstock and CafeF.

The study evaluates six tasks: ESG relevance; sentiment for all eligible articles; sentiment within ESG-relevant articles; and Environmental, Social and Governance presence. Model comparisons include TF-IDF baselines, PhoBERT and XLM-RoBERTa.

## Repository contents

- `data/annotations/`: **label-only** expert-return datasets. Article body, title, description, evidence quotations and expert notes are intentionally excluded.
- `scripts/`: core processing, model training and evaluation code extracted from research notebooks without notebook outputs.
- `docs/`: documentation of study scope and reproduction limitations.

## Important data and reproduction limitations

This is a public repository. Full copyrighted news articles and unredacted expert workbooks are **not distributed**. Labels alone are insufficient for retraining text classifiers: authorized access to the original article texts is needed. Do not interpret the label-only datasets as the final adjudicated benchmark; the expert-return workbooks are pre-adjudication review material.

The study trains on Vietstock, selects configurations on CafeF validation, and reports results on held-out CafeF testing. As publisher and time period shift together, the protocol does not isolate publisher effects.

The scripts were adapted from Colab notebooks; paths, installed dependencies, checkpoints, and environment must be configured before running. Do not assume end-to-end reproducibility without the restricted inputs.

## Citation

Manuscript citation and archival DOI will be added when available.
