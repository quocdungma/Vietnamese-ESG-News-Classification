# Vietnamese ESG News Classification

Research code and selected publication figures for classifying Vietnamese financial news by ESG relevance, sentiment, and ESG pillars.

## Study scope

Six article-level tasks:
1. ESG relevance
2. Sentiment on all eligible articles
3. Sentiment within ESG-relevant articles
4. Environmental presence
5. Social presence
6. Governance presence

Models include TF-IDF-based lexical baselines, PhoBERT, and XLM-RoBERTa. Training uses Vietstock articles; CafeF is used for target-source validation and held-out testing. Publisher and time-period differences are not separately identified, so the results should not be described as zero-shot adaptation or a pure publisher-only effect.

## Code

- `scripts/01_expert_agreement.py`: expert agreement
- `scripts/02_expert_adjudication.py`: expert adjudication assessment
- `scripts/03_train_phobert.py`: PhoBERT and lexical baselines
- `scripts/04_train_xlmr.py`: XLM-RoBERTa
- `scripts/05_random_seed_evaluation.py`: repeated-seed assessment

These scripts are exported notebook cells and have **not** been validated as standalone executable programs. Colab-specific setup and Drive paths remain. `requirements.txt` is a provisional dependency list, not a locked execution environment.

## Data and figures

- `data/annotations/ESG_Master_Expert_A_Final_labels_only.csv` and `ESG_Master_Expert_B_Final_labels_only.csv`: 2,000 rows per reconstructed expert master (800 CafeF + 1,200 Vietstock each), with nine non-text columns. These are **not** independently newly annotated 2,000-row sets or the final adjudicated gold labels.
- `figures/figure_3_paired_encoder_differences.svg`: paired encoder performance differences.
- `figures/figure_4_sentiment_class_recall.svg`: sentiment class recall by seed.
- `results/figure_4_class_recall_by_seed.csv`: numerical data supporting Figure 4.

Source figure exports were located in the research Drive `ESG_annotation_sampling/manuscript_figures`. Unreviewed machine-translated supplementary wordclouds are not presented as final publication figures.

The experiments use a larger labeled Vietstock/CafeF corpus of 2,800 articles. The public expert master label-only files are not sufficient for retraining. Original news text, full expert workbooks, quotations, article URLs, and unreleased gold data are intentionally excluded pending permission and release review.

## Reproducibility and release

Read [reproducibility notes](docs/REPRODUCIBILITY.md) and [release checklist](docs/RELEASE_CHECKLIST.md). Frozen split manifests and complete result tables have not yet been published. End-to-end reproduction is **not currently claimed**.

## Citation and license

A definitive manuscript citation, license, and archival DOI will be added after author and publication-rights review.
