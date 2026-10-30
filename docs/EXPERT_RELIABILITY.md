# Independent expert agreement and annotation design

## Annotation assignments

The study's annotation protocol involved **two experts working independently** on article-level labels before discussion and adjudication:

| Design component | Expert A | Expert B |
| --- | ---: | ---: |
| Independently annotated articles | 2,000 | 2,000 |
| CafeF articles in assignment | 800 | 800 |
| Vietstock articles in assignment | 1,200 | 1,200 |
| Articles shared by both experts | 1,200 | 1,200 |
| Articles unique to each assignment | 800 | 800 |

There are **4,000 independent article-annotation assignments across 2,800 distinct articles**. The intersection includes 800 CafeF and 400 Vietstock articles. The **same 1,200 articles** were independently labeled by both experts, providing the dataset for direct A-versus-B inter-annotator reliability.

After the independent annotations, disagreements were discussed. The reference label set is finalized by adjudication; it is **not** used to calculate the direct A/B agreement shown below. The second author's deciding vote is an adjudication mechanism, not a third independent rater.

## Direct A/B agreement on the 1,200 common articles

| Task | n | Observed agreement | Cohen's kappa | Symmetric macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ESG relevance | 1,200 | 82.42% | 0.6460 | 0.8230 |
| Sentiment | 1,200 | 80.17% | 0.6750 | 0.7786 |
| Environmental | 1,200 | 94.17% | 0.7080 | 0.8539 |
| Social | 1,200 | 84.58% | 0.6485 | 0.8242 |
| Governance | 1,200 | 85.17% | 0.5308 | 0.7653 |

The lowest observed agreement is for sentiment (80.17%) and the highest for environmental (94.17%). The lowest chance-corrected kappa is governance (0.5308), so observed agreement must be interpreted alongside class prevalence. Kappa is unweighted for nominal labels. Macro-F1 in pairwise comparisons is reported as an additional descriptive diagnostic, not as annotation accuracy.

The full summary [results/expert_reliability_pairwise_summary.csv](../results/expert_reliability_pairwise_summary.csv) includes pooled and publisher-specific results, intervals for observed agreement, balanced accuracy and MCC. No per-article original text is published.

## Expert vs adjudicated final labels: a different comparison

[results/expert_vs_final_2000_summary.csv](../results/expert_vs_final_2000_summary.csv) describes each expert's labels against the final adjudicated reference across each expert's 2,000 articles. Because the final reference can incorporate the experts' judgments, **agreement with Final is not an independent measure of inter-rater reliability** and should not replace the direct A/B agreement above.

## Data provenance

The two public label-only CSVs contain **all 2,000 independent expert judgments each**. Some archived processing metadata classify records by import or assembly stage, such as `new_returned_annotation`, `pre_existing_current_final_label` and `current_done_final_label`. These terms record **data-processing provenance**, not whether the experts made their initial annotation decisions independently. They must not be used to infer that only 600 annotations per expert were independent.

The released data omit original news text, article titles, article URLs and expert reasoning notes. Files retain article keys, publisher, five labels and annotation status. The final adjudicated gold-standard dataset is not publicly released.

## Manuscript alignment

- Section 3.3: describe direct A/B pre-adjudication agreement on 1,200 jointly assigned articles
- Supplementary Table S1: report observed agreement and Cohen's kappa, pooled and optionally by publisher
- Sections 5.1 and Conclusion: separate annotation quality from predictive performance and from Expert-vs-Final concordance
