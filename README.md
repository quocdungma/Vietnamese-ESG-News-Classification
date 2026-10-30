# Article-level ESG classification of Vietnamese financial news

Research code, annotation labels, summary metrics and selected figures supporting a cross-publisher comparison of lexical classifiers, PhoBERT and XLM-RoBERTa.

## Study and research questions

The study uses 2,800 unique articles (Vietstock: 2,000; CafeF: 800) and six prediction tasks: (1) ESG relevance, (2) sentiment on all articles, (3) sentiment conditional on reference ESG relevance, and three binary pillar tasks: (4) environmental, (5) social and (6) governance. Sentiment is annotated relative to the article's central target; the models do not automatically extract that target. Experimental `mixed` sentiment also includes originally neutral labels.

Models are trained on Vietstock and configured with a CafeF validation sample; final results are assessed on a separate held-out CafeF test sample. This is **cross-publisher evaluation with target-source validation**, not zero-shot transfer. Publisher and observation period vary together.

## Independent expert annotation design

**Expert A independently annotated 2,000 articles; Expert B independently annotated 2,000 articles.** Their assignments overlap on **1,200 identical articles** (800 CafeF and 400 Vietstock), and each expert separately annotated 800 articles not assigned to the other. The union comprises 2,800 unique articles. The 1,200 shared articles provide direct, **pre-adjudication inter-annotator agreement**, summarized by observed agreement and Cohen's kappa across five recorded labels.

The released files are label-only exports of the complete expert datasets:
- [Expert A, 2,000 labels](data/annotations/ESG_Master_Expert_A_Final_labels_only.csv)
- [Expert B, 2,000 labels](data/annotations/ESG_Master_Expert_B_Final_labels_only.csv)

The adjudicated reference/gold labels used for training and evaluation are distinct from these independent expert labels and are not publicly redistributed. See [expert reliability](docs/EXPERT_RELIABILITY.md). Processing-stage provenance values in an older results file reflect how annotation records were assembled for export; they do **not** describe whether experts independently made their judgments.

## Models and evaluation

- Majority-class baseline; TF-IDF logistic regression; TF-IDF linear SVM; PhoBERT-base-v2; XLM-RoBERTa-base
- Shared split: Vietstock train 2,000; CafeF validation 400; CafeF test 400. Conditional ESG task sizes: 715 / 206 / 206
- Three encoder seeds: 13, 42, 77; task/encoder-specific four-view weights selected on CafeF validation and then locked across seeds
- Each view uses up to 256 tokens; outputs are combined using locked nonnegative weights
- Training environment: Python on Google Colab, NVIDIA L4 GPU; bfloat16, learning rate 2e-5, training batch 16, evaluation batch 32, accumulation 1

## Research code

The five files in [scripts](scripts) are exported notebook code:
1. `01_expert_agreement.py`
2. `02_expert_adjudication.py`
3. `03_train_phobert.py`
4. `04_train_xlmr.py`
5. `05_random_seed_evaluation.py`

**The exported scripts have not been validated as standalone, end-to-end runnable programs**. Google Colab paths and private input data are still required. Dependencies in `requirements.txt` are provisional, not a locked environment.

## Released data, results and figures

- `data/splits/article_split_manifest_labels_only.csv`: article identifiers, source and frozen train/validation/test assignments; no original text, titles or URLs
- `results/final_6task_model_comparison.csv`: six-task encoder mean and sample SD across seeds
- `results/paired_bootstrap_by_task_seed.csv`: 18 paired seed-task Macro-F1 differences and intervals
- `results/baseline_metrics_six_tasks.csv`: validation-selected lexical and majority-class reference scores
- `results/expert_reliability_pairwise_summary.csv`: direct Expert A/B comparisons across 1,200 shared articles, pooled and by publisher
- `results/expert_vs_final_2000_summary.csv`: **secondary** expert-to-final comparisons; not inter-annotator reliability
- `figures/figure_1_paired_encoder_differences.svg`: Figure 1 in the manuscript
- `figures/figure_2_sentiment_class_recall.svg`: Figure 2 in the manuscript
- `results/figure_2_class_recall_by_seed.csv`: numerical source for Figure 2

Figure 3 in the manuscript is a descriptive ESG pillar wordcloud. Machine-translated phrases need manual review before a publication-ready figure is added.

## Limitations, availability and anonymization

Article full text, titles, URLs, detailed expert workbooks and the adjudicated final reference labels are not publicly released pending redistribution and data-rights review. Public artifacts do not presently enable full retraining. See [reproducibility and scope](docs/REPRODUCIBILITY.md) and [release checklist](docs/RELEASE_CHECKLIST.md).

This repository identifies its account owner. **Do not put its direct URL in a double-anonymized review manuscript**; provide an appropriately reviewed anonymized repository snapshot if required. A license, archival DOI and definitive citation remain pending publication-rights review.
