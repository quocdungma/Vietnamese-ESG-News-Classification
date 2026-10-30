# Expert reliability: aggregate results and interpretation

Source: `ESG_Independent_Experts_Evaluation.xlsx` (private research Drive, modified 2026-09-27). These files reproduce aggregate values from the workbook; no article titles, texts or individual-level disagreement records are included.

## Design and provenance (important)

Each reconstructed expert master contains 2,000 articles. The union contains 2,800 articles and the reconstructed A–B overlap contains 1,200 (800 CafeF, 400 Vietstock). However, only **600 entries per expert** are flagged `new_returned_annotation`, with 800 `pre_existing_current_final_label` and 600 `current_done_final_label` each. Therefore **the full reconstructed 1,200-overlap comparison must not be described as 1,200 independently newly annotated overlapping articles**.

The two public `returned_expert_*.csv` files represent returned labels only. The reconstructed master comparisons include pre-existing and reference-derived labels. Some very high Expert-vs-Final values are not independent evidence of reliability.

## Reconstructed Expert A–B agreement (n=1,200)

| Task | Observed agreement | Cohen's kappa | Macro-F1 |
| --- | ---: | ---: | ---: |
| ESG relevance | 0.8242 | 0.6460 | 0.8230 |
| Sentiment | 0.8017 | 0.6750 | 0.7786 |
| Environmental | 0.9417 | 0.7080 | 0.8539 |
| Social | 0.8458 | 0.6485 | 0.8242 |
| Governance | 0.8517 | 0.5308 | 0.7653 |

These figures describe agreement on the reconstructed overlap. Interpret kappa alongside class prevalence and observed agreement; do not conflate agreement with accuracy or with independent expert validation.

## Files

- `results/expert_reliability_pairwise_summary.csv`: source-specific and pooled Expert A–B aggregated agreement, confidence intervals, macro-F1, balanced accuracy and MCC.
- `results/expert_reliability_provenance.csv`: assignment provenance and number of genuinely returned annotations.
- `results/expert_reliability_coverage_checks.csv`: structural overlap and union checks.

## Limitations and required follow-up

- Report the comparison as *reconstructed expert-label agreement* unless independent overlap can be verified from raw assignment provenance.
- Do not equate all reconstructed A–B matches to true independently observed agreement.
- Separate reference-based Expert-vs-Final agreement from independent inter-rater reliability.
- Validate the 600 newly returned labels per expert and source-specific independent overlap against the assignment design before using these numbers as a manuscript inter-annotator reliability claim.
- Publish only aggregated non-sensitive outputs. No article text, URLs, titles, reviewer notes or original expert workbooks are included.
