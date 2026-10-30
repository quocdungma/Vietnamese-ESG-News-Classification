# Release checklist (complete before merging PR #2)

- [ ] Verify five scripts compile with Python and run in a clean environment.
- [ ] Freeze package versions from the execution logs, then update requirements.txt.
- [ ] Replace absolute Google Drive paths with documented configuration.
- [ ] Verify six task definitions, splits, label encodings, and seed values against the final manuscript.
- [ ] Verify no target-source/test-set leakage from model selection.
- [ ] Run near-duplicate and event-group leakage checks.
- [ ] Add test results, Macro-F1 tables, confidence intervals and class-level metrics from final experiment exports.
- [ ] Add and inspect Figure 3 and Figure 4 (SVG/PNG) with corresponding data tables.
- [ ] Confirm translated supplementary wordcloud phrases have been manually reviewed.
- [ ] Check release rights for expert labels, figures, identifiers and original news.
- [ ] Verify README, license, authorship and citation text.
- [ ] Merge only after all publication-sensitive checks.
