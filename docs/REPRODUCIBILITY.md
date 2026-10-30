# Reproducibility and release scope

## Empirical design

- Corpus: 2,800 unique articles; Vietstock (train) 2,000 and CafeF (validation/test) 800
- Six classification outcomes: ESG relevance, all-article sentiment, ESG-conditional sentiment, environmental, social and governance
- Split: Vietstock train 2,000; CafeF validation 400; CafeF held-out test 400. Conditional ESG task samples: train 715, validation 206, test 206
- Models: training-majority reference, TF-IDF logistic regression, TF-IDF linear SVM, PhoBERT-base-v2 (`vinai/phobert-base-v2`) and XLM-RoBERTa-base (`FacebookAI/xlm-roberta-base`)
- Encoder seeds: 13, 42, 77; 6 tasks × 2 encoders × 3 seeds = 36 completed fits
- Encoder text: title+description, opening, continuation and ending views (maximum 256 tokens per view); nonnegative task/model-specific weights selected on CafeF validation and fixed for all three seeds
- Runtime: Python / Google Colab / NVIDIA L4 GPU; bfloat16, learning rate 2 × 10^-5, batch size 16 training and 32 evaluation, accumulation 1
- Primary metric: Macro-F1; 18 encoder task/seed paired bootstrap comparisons on same CafeF test articles

**Interpretation:** A publisher shift and a time-period shift coincide. The experiment uses target-publisher validation and is not zero-shot transfer. The four-view configuration is validation-selected; there is no isolated view ablation and no causal estimate of a view's contribution.

## Independent expert annotation

Expert A and Expert B each independently annotated 2,000 articles, with 1,200 **identical** doubly annotated articles and 800 unique articles per expert. The total unique article count is 2,800. Independent inter-rater reliability uses pre-adjudication A/B labels on the shared 1,200 articles. See [EXPERT_RELIABILITY.md](EXPERT_RELIABILITY.md).

## Public reproducibility assets

- `data/splits/article_split_manifest_labels_only.csv` records the frozen global allocation using article IDs and publisher (no text, title or URL)
- `results/final_6task_model_comparison.csv` records model performance across seeds
- `results/paired_bootstrap_by_task_seed.csv` provides paired deltas and intervals
- `results/baseline_metrics_six_tasks.csv` records results for the six manuscript baselines
- `results/figure_2_class_recall_by_seed.csv` provides class-specific recall underlying Figure 2
- `results/expert_reliability_pairwise_summary.csv` provides direct Expert A/B agreement
- `figures/figure_1_paired_encoder_differences.svg` and `figures/figure_2_sentiment_class_recall.svg` correspond to manuscript Figures 1 and 2

The split manifest has 2,800 distinct article IDs: 2,000 Vietstock train, 400 CafeF validation and 400 CafeF test. This is a **global all-article split**; the ESG-conditional subsets are obtained using withheld adjudicated reference labels. The split manifest is not a release of those labels. Stored train/validation/test fingerprints are recorded in the original experiment logs. Independent end-to-end reproducibility has not been verified.

## Limitations

- The public label-only expert files are not the adjudicated experimental gold-standard dataset; original publisher text/URLs and full expert workbooks are withheld
- Five research scripts were exported from Colab notebooks and have not been verified to run as independent Python CLI programs
- Some scripts still include absolute Google Drive paths; adjust for a supported runtime
- `requirements.txt` is an **unlocked** list of observed dependencies. Versions, checkpoint-selection rules and full clean-environment validation remain pending
- No published event-level near-duplicate holdout or isolated text-view ablation
- The wordcloud panel with machine-generated English phrases requires language review before publication

Public artifacts permit inspection of the methodology and many numerical outputs; they are **not presently sufficient to reproduce model training** without authorized access to underlying gold labels and news content.
