# Manuscript / repository consistency checklist (October 2026)

## Verified and published
- [x] Document two independent expert assignments of 2,000 articles each, 1,200 shared, 2,800 unique
- [x] Publish label-only Expert A and Expert B 2,000-record CSVs
- [x] Publish direct A/B agreement summary separately from expert-to-final comparisons
- [x] Align figure numbers with manuscript Figure 1 (paired Macro-F1) and Figure 2 (sentiment recall)
- [x] Publish non-text global split manifest: 2,000 / 400 / 400 and 2,800 unique IDs
- [x] Publish six-task baseline and paired three-seed performance results
- [x] Document Google Colab and NVIDIA L4 GPU configuration; seeds 13, 42, 77
- [x] Distinguish target-source validation from zero-shot transfer
- [x] State four-view ablation has not been conducted

## Still required before making a full reproducibility claim
- [ ] Validate five scripts end-to-end in a clean Colab session with approved private inputs
- [ ] Lock Python/package versions from actual environment and update requirements
- [ ] Document early stopping/epoch count and checkpoint selection from original run logs
- [ ] Check public split manifest against frozen experiment fingerprints with the original hashing method
- [ ] Audit near-duplicate/event leakage across publishers, beyond the recorded article-level split checks
- [ ] Review rights to share article IDs and expert labels (no news text or URLs in public files)
- [ ] Prepare/anonymize repository snapshot for double-anonymized peer review
- [ ] Inspect English translations for manuscript Figure 3 wordcloud and publish only reviewed version
- [ ] Confirm DOI/citation/license and journal-specific data availability policy
