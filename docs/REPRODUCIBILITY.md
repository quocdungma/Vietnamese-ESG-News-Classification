# Code and data audit

## Source folders

- Google Drive `Crawling_news/Analyze/Update_Crawling`: research Colab notebooks used for collection, annotation, comparison and evaluation.
- Google Drive `Crawling_news/ESG_annotation_sampling/annotation_reliability_expert_assignment/returned`: 600 expert A and 600 expert B returned records, with 400 CafeF + 200 Vietstock assignments per expert.

## Included core code

1. `01_crawl_news.py`: news collection/update.
2. `02_prepare_vietstock_annotation.py` and `03_prepare_cafef_annotation.py`: sampling/annotation preparation.
3. `04_expert_agreement.py` and `05_expert_adjudication.py`: expert-label agreement and adjudication.
4. `06_train_phobert.py` and `07_train_xlmr.py`: cross-source task definitions, preprocessing, baseline and model-training code.
5. `08_random_seed_evaluation.py`: repeated-seed assessment.

The code was extracted in cell order from original Jupyter notebooks, with notebook outputs excluded. Notebook shell/magic commands were commented with `# COLAB COMMAND:`. These are **archival scripts**, not independently tested command-line programs. Manual environment setup, path adaptation, execution ordering and fixes to Colab-specific commands may be required. Source notebooks remain in private Drive.

## Label-only release

The two files in `data/annotations` contain assignment IDs, source, article ID, expert identifier, five main reference fields and annotation status. They exclude titles, descriptions, content, article URLs, supporting quotations and comments. The files reflect **separate expert-return assignments**, not finalized adjudicated gold labels. Do not use them as a substitute for the frozen experiment split or finalized gold standard.

The two experts did not each independently annotate all the same 1,200 articles. Report agreement design accurately; the publication documents expert-to-existing-reference comparisons on disjoint subsets. No personal expert workbooks or full text should be published without appropriate permissions.

## Remaining gaps before archival release

- Validate extracted Python syntax and remove or convert all Colab-only commands.
- Align Drive folder paths and package versions; supply configuration templates.
- Add frozen split manifest fingerprints and final label-only gold data once publication rights and release protocol are cleared.
- Compare exported scripts against final manuscript and final experiment logs.
- Check near-duplicate/event-level leakage and date/source confounding.
- Check whether raw publisher content can be redistributed under license; current repository does not include it.
