# Reproducibility and release scope

## Scope
This branch archives five exported Colab research scripts: expert agreement, expert adjudication assessment, PhoBERT and lexical-baseline training, XLM-RoBERTa training, and repeated-seed evaluation. Scripts are not yet validated as stand-alone CLI programs.

The public expert CSVs are **2,000-row label-only exports from the user-supplied Expert A/B Final master workbooks**; they are not the adjudicated final gold set and do not establish 2,000 independently new expert annotations per rater. Original publisher article texts, URLs, quotations, expert notes and full workbooks are intentionally excluded. Neither the released labels nor this repository alone suffice to retrain the classifiers.

## Evaluation protocol
Six reported tasks: ESG relevance; all-article sentiment; sentiment restricted to ESG-relevant articles; and E, S, G presence. Vietstock is used for training and CafeF for target-source validation and held-out testing. Publisher and period may be confounded, so this is not proof of isolated publisher-domain transfer or zero-shot transfer.

## Reproduction status
- Source notebooks were exported in cell order without execution outputs.
- Google Colab mounts and absolute Drive paths must be adapted.
- Dependency versions are not frozen; requirements.txt is provisional.
- Frozen train/validation/test manifests, signatures, and adjudicated gold labels are not public.
- Exported scripts have not passed end-to-end execution checks.
- Before claiming reproducibility, run syntax compilation, import checks, deterministic split verification, data leakage tests, and alignment against final manuscript/experiment logs.

## Figures and result files
Figure 3: paired encoder Macro-F1 differences. Figure 4: sentiment class recall by seed. Keep numerical supporting data alongside figures. Supplementary word clouds that use machine-generated English translations are **draft/unreviewed**, and should not be identified as final without language review.

## Privacy and licensing
Do not upload article full text or unredacted expert workbooks without explicit redistribution permission. Review any article identifiers for privacy/linkability and verify release rights before merge.

## Source
Figures were identified in the private Drive folder `Crawling_news/ESG_annotation_sampling/manuscript_figures`, Sept 2026. The public repository should contain only intentionally released aggregate visualizations and non-sensitive summaries.
