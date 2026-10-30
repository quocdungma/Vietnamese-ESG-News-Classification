# Extracted from original Colab notebook. Review paths and execute cells in order.


# === Notebook cell 2 ===
from google.colab import drive
drive.mount("/content/gdrive")

# === Notebook cell 3 ===
# ============================================================
# CELL 1 — CONFIGURATION
# ============================================================

from pathlib import Path

# Update these paths before running.
EXPERT_A_PATH = Path(
    "/content/gdrive/MyDrive/Project_Fin_Data_Crawling/Crawling_news/ESG_annotation_sampling/annotation_reliability_expert_assignment/reconstructed_expert_datasets/ESG_Master_Expert_A.xlsx"
)

EXPERT_B_PATH = Path(
    "/content/gdrive/MyDrive/Project_Fin_Data_Crawling/Crawling_news/ESG_annotation_sampling/annotation_reliability_expert_assignment/reconstructed_expert_datasets/ESG_Master_Expert_B.xlsx"
)

FINAL_PATH = Path(
    "/content/gdrive/MyDrive/Project_Fin_Data_Crawling/Crawling_news/ESG_annotation_sampling/annotation_reliability_expert_assignment/reconstructed_expert_datasets/ESG_Final_Full_Labels.xlsx"
)

OUTPUT_DIR = Path(
    "/content/gdrive/MyDrive/Project_Fin_Data_Crawling/Crawling_news/ESG_annotation_sampling/annotation_reliability_expert_assignment/reconstructed_expert_datasets/independent_expert_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


KEY = "article_key"

TASKS = [
    "esg_relevance",
    "sentiment",
    "environmental",
    "social",
    "governance",
]

EXPECTED_A_ROWS = 2000
EXPECTED_B_ROWS = 2000
EXPECTED_OVERLAP = 1200
EXPECTED_A_EXCLUSIVE = 800
EXPECTED_B_EXCLUSIVE = 800
EXPECTED_UNION = 2800
EXPECTED_FINAL_ROWS = 2800

RESULTS_PATH = (
    OUTPUT_DIR
    / "ESG_Independent_Experts_Evaluation.xlsx"
)


# === Notebook cell 4 ===
# ============================================================
# CELL 2 — IMPORTS AND NORMALIZATION HELPERS
# ============================================================

import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def normalize_text(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def missing_mask(series):
    normalized = normalize_text(series)

    return (
        series.isna()
        | normalized.isin(
            [
                "",
                "nan",
                "none",
                "null",
                "<na>",
            ]
        )
    )


def normalize_boolean(series):
    normalized = normalize_text(series)

    return (
        series.eq(True)
        | normalized.isin(
            [
                "true",
                "1",
                "yes",
                "y",
            ]
        )
    )


def normalize_task(task, series):
    normalized = normalize_text(series)

    if task == "esg_relevance":
        normalized = normalized.replace(
            {
                "esg": "relevant",
                "non_esg": "not_relevant",
                "non-esg": "not_relevant",
                "not relevant": "not_relevant",
            }
        )

    elif task in {
        "environmental",
        "social",
        "governance",
    }:
        normalized = normalized.replace(
            {
                "0.0": "0",
                "1.0": "1",
                "false": "0",
                "true": "1",
            }
        )

    return normalized


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (
        proportion
        + z**2 / (2 * total)
    ) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return (
        max(0, centre - half_width),
        min(1, centre + half_width),
    )


def interpret_kappa(value):
    if pd.isna(value):
        return "not_estimable"
    if value < 0:
        return "below_chance"
    if value < 0.21:
        return "slight"
    if value < 0.41:
        return "fair"
    if value < 0.61:
        return "moderate"
    if value < 0.81:
        return "substantial"
    return "almost_perfect"


def read_first_or_named_sheet(path, preferred_sheet=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    workbook = pd.ExcelFile(path)

    if preferred_sheet in workbook.sheet_names:
        sheet = preferred_sheet
    else:
        sheet = workbook.sheet_names[0]

    frame = pd.read_excel(path, sheet_name=sheet)
    print(f"Loaded {path.name} | sheet={sheet} | shape={frame.shape}")
    return frame


# === Notebook cell 5 ===
# ============================================================
# CELL 3 — LOAD THE THREE AUTHORITATIVE FILES
# ============================================================

A = read_first_or_named_sheet(
    EXPERT_A_PATH,
    preferred_sheet="Expert_A_Master",
)

B = read_first_or_named_sheet(
    EXPERT_B_PATH,
    preferred_sheet="Expert_B_Master",
)

F = read_first_or_named_sheet(
    FINAL_PATH,
    preferred_sheet="Final_Full_Labels",
)


# === Notebook cell 6 ===
# ============================================================
# CELL 4 — STANDARDIZE AND VALIDATE THE THREE FILES
# ============================================================

datasets = {
    "Expert A": A,
    "Expert B": B,
    "Final": F,
}

required_columns = [
    KEY,
    "source",
    *TASKS,
]

qa_records = []


for name, frame in datasets.items():
    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {missing_columns}"
        )

    frame[KEY] = (
        frame[KEY]
        .astype("string")
        .str.strip()
    )

    if missing_mask(frame[KEY]).any():
        raise ValueError(
            f"{name} contains missing article_key values."
        )

    if frame[KEY].duplicated().any():
        duplicate_rows = frame.loc[
            frame[KEY].duplicated(keep=False)
        ].sort_values(KEY)
        display(duplicate_rows.head(100))
        raise ValueError(
            f"{name} contains duplicate article_key values."
        )

    frame["source"] = normalize_text(frame["source"])

    for task in TASKS:
        frame[task] = normalize_task(task, frame[task])

        if missing_mask(frame[task]).any():
            raise ValueError(
                f"{name} contains missing labels for {task}."
            )

    qa_records.append(
        {
            "file": name,
            "rows": len(frame),
            "unique_articles": frame[KEY].nunique(),
            "cafef": int(frame["source"].eq("cafef").sum()),
            "vietstock": int(frame["source"].eq("vietstock").sum()),
        }
    )


# Provenance is reported for transparency but does not block
# the requested all-2,000-row evaluation.
provenance_diagnostics = []

for name, frame in {
    "Expert A": A,
    "Expert B": B,
}.items():
    if "label_provenance" in frame.columns:
        provenance = set(
            normalize_text(frame["label_provenance"])
            .dropna()
        )

        provenance_counts = (
            normalize_text(frame["label_provenance"])
            .value_counts(dropna=False)
            .to_dict()
        )
    else:
        provenance_counts = {}

    if "is_direct_new_annotation" in frame.columns:
        direct_flag = normalize_boolean(
            frame["is_direct_new_annotation"]
        )

        direct_count = int(direct_flag.sum())
    else:
        direct_count = np.nan

    provenance_diagnostics.append(
        {
            "file": name,
            "rows_used_for_evaluation": len(frame),
            "direct_annotation_flag_count": direct_count,
            "provenance_counts": str(provenance_counts),
            "blocks_evaluation": False,
        }
    )


qa_summary = pd.DataFrame(qa_records)
display(qa_summary)

provenance_summary = pd.DataFrame(provenance_diagnostics)
display(provenance_summary)


if len(A) != EXPECTED_A_ROWS:
    raise ValueError(f"Expert A has {len(A)} rows; expected 2,000.")

if len(B) != EXPECTED_B_ROWS:
    raise ValueError(f"Expert B has {len(B)} rows; expected 2,000.")

if len(F) != EXPECTED_FINAL_ROWS:
    raise ValueError(f"Final has {len(F)} rows; expected 2,800.")

print("Three-file QA: PASS")


# === Notebook cell 7 ===
# ============================================================
# CELL 5 — VERIFY 1,200 OVERLAP AND 2,800 UNION
# ============================================================

keys_a = set(A[KEY])
keys_b = set(B[KEY])
keys_f = set(F[KEY])

overlap_keys = keys_a & keys_b
a_exclusive_keys = keys_a - keys_b
b_exclusive_keys = keys_b - keys_a
union_keys = keys_a | keys_b


set_qa = pd.DataFrame(
    [
        {
            "measure": "Expert A articles",
            "observed": len(keys_a),
            "expected": EXPECTED_A_ROWS,
        },
        {
            "measure": "Expert B articles",
            "observed": len(keys_b),
            "expected": EXPECTED_B_ROWS,
        },
        {
            "measure": "A-B overlap",
            "observed": len(overlap_keys),
            "expected": EXPECTED_OVERLAP,
        },
        {
            "measure": "A-exclusive",
            "observed": len(a_exclusive_keys),
            "expected": EXPECTED_A_EXCLUSIVE,
        },
        {
            "measure": "B-exclusive",
            "observed": len(b_exclusive_keys),
            "expected": EXPECTED_B_EXCLUSIVE,
        },
        {
            "measure": "A-B union",
            "observed": len(union_keys),
            "expected": EXPECTED_UNION,
        },
        {
            "measure": "Final articles",
            "observed": len(keys_f),
            "expected": EXPECTED_FINAL_ROWS,
        },
        {
            "measure": "Union symmetric difference with Final",
            "observed": len(union_keys ^ keys_f),
            "expected": 0,
        },
    ]
)

set_qa["status"] = np.where(
    set_qa["observed"].eq(set_qa["expected"]),
    "PASS",
    "FAIL",
)

display(set_qa)


if set_qa["status"].eq("FAIL").any():
    raise ValueError(
        "The expert overlap/union design does not match 2,000/2,000/1,200/2,800."
    )

print("Overlap and union QA: PASS")


# === Notebook cell 8 ===
# ============================================================
# CELL 6 — BUILD ALIGNED COMPARISON TABLES
# ============================================================

metadata_columns = [
    column
    for column in [
        KEY,
        "source",
        "source_batch",
        "title",
    ]
    if column in F.columns
]


def prepare_labels(frame, prefix):
    selected = frame[
        [
            column
            for column in [
                KEY,
                "source",
                "source_batch",
                "title",
                *TASKS,
            ]
            if column in frame.columns
        ]
    ].copy()

    return selected.rename(
        columns={
            task: f"{prefix}_{task}"
            for task in TASKS
        }
    )


A_prepared = prepare_labels(A, "A")
B_prepared = prepare_labels(B, "B")
F_prepared = prepare_labels(F, "Final")


A_vs_final = A_prepared.merge(
    F_prepared[[KEY, *[f"Final_{task}" for task in TASKS]]],
    on=KEY,
    how="left",
    validate="one_to_one",
)

B_vs_final = B_prepared.merge(
    F_prepared[[KEY, *[f"Final_{task}" for task in TASKS]]],
    on=KEY,
    how="left",
    validate="one_to_one",
)

ABF_overlap = (
    A_prepared.loc[A_prepared[KEY].isin(overlap_keys)]
    .merge(
        B_prepared.loc[B_prepared[KEY].isin(overlap_keys)][
            [KEY, *[f"B_{task}" for task in TASKS]]
        ],
        on=KEY,
        how="inner",
        validate="one_to_one",
    )
    .merge(
        F_prepared[[KEY, *[f"Final_{task}" for task in TASKS]]],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
)


if len(A_vs_final) != 2000:
    raise AssertionError("A-vs-Final must contain 2,000 rows.")

if len(B_vs_final) != 2000:
    raise AssertionError("B-vs-Final must contain 2,000 rows.")

if len(ABF_overlap) != 1200:
    raise AssertionError("A-B-Final overlap must contain 1,200 rows.")

print("A vs Final:", A_vs_final.shape)
print("B vs Final:", B_vs_final.shape)
print("A vs B vs Final overlap:", ABF_overlap.shape)


# === Notebook cell 9 ===
# ============================================================
# CELL 7 — METRIC FUNCTIONS
# ============================================================

def calculate_pair_metrics(
    frame,
    left_prefix,
    right_prefix,
    comparison_name,
    source_group="all",
):
    if source_group == "all":
        subset = frame.copy()
    else:
        subset = frame.loc[
            frame["source"].eq(source_group)
        ].copy()

    records = []

    for task in TASKS:
        left_column = f"{left_prefix}_{task}"
        right_column = f"{right_prefix}_{task}"

        valid = subset.loc[
            ~missing_mask(subset[left_column])
            & ~missing_mask(subset[right_column])
        ].copy()

        left = valid[left_column].astype(str).to_numpy()
        right = valid[right_column].astype(str).to_numpy()

        n = len(valid)

        if n == 0:
            continue

        matches = left == right
        n_agree = int(matches.sum())
        n_disagree = int((~matches).sum())
        agreement = n_agree / n
        ci_low, ci_high = wilson_interval(n_agree, n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                kappa = cohen_kappa_score(right, left)
            except Exception:
                kappa = np.nan

            try:
                macro_f1 = f1_score(
                    right,
                    left,
                    average="macro",
                    zero_division=0,
                )
            except Exception:
                macro_f1 = np.nan

            try:
                balanced_accuracy = balanced_accuracy_score(right, left)
            except Exception:
                balanced_accuracy = np.nan

            try:
                mcc = matthews_corrcoef(right, left)
            except Exception:
                mcc = np.nan

        records.append(
            {
                "comparison": comparison_name,
                "source": source_group,
                "task": task,
                "n_compared": n,
                "n_agree": n_agree,
                "n_disagree": n_disagree,
                "agreement": agreement,
                "agreement_ci_low": ci_low,
                "agreement_ci_high": ci_high,
                "cohen_kappa": kappa,
                "kappa_interpretation": interpret_kappa(kappa),
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_accuracy,
                "matthews_correlation": mcc,
            }
        )

    return pd.DataFrame(records)


# === Notebook cell 10 ===
# ============================================================
# CELL 8 — EXPERT A/EXPERT B VERSUS FINAL ON 2,000 ARTICLES
# ============================================================

expert_final_metric_frames = []

for source_group in ["all", "cafef", "vietstock"]:
    expert_final_metric_frames.append(
        calculate_pair_metrics(
            A_vs_final,
            left_prefix="A",
            right_prefix="Final",
            comparison_name="Expert A vs Final",
            source_group=source_group,
        )
    )

    expert_final_metric_frames.append(
        calculate_pair_metrics(
            B_vs_final,
            left_prefix="B",
            right_prefix="Final",
            comparison_name="Expert B vs Final",
            source_group=source_group,
        )
    )


expert_vs_final_metrics = (
    pd.concat(
        expert_final_metric_frames,
        ignore_index=True,
    )
    .sort_values(
        ["task", "source", "comparison"]
    )
    .reset_index(drop=True)
)


print("===== EXPERT VS FINAL METRICS =====")
display(expert_vs_final_metrics)


print("===== OVERALL 2,000-ARTICLE RESULTS =====")
display(
    expert_vs_final_metrics.loc[
        expert_vs_final_metrics["source"].eq("all")
    ].reset_index(drop=True)
)


# === Notebook cell 11 ===
# ============================================================
# CELL 9 — A-B INTER-ANNOTATOR AGREEMENT ON 1,200 ARTICLES
# ============================================================

ab_metric_frames = []

for source_group in ["all", "cafef", "vietstock"]:
    ab_metric_frames.append(
        calculate_pair_metrics(
            ABF_overlap,
            left_prefix="A",
            right_prefix="B",
            comparison_name="Expert A vs Expert B",
            source_group=source_group,
        )
    )


inter_annotator_metrics = (
    pd.concat(
        ab_metric_frames,
        ignore_index=True,
    )
    .sort_values(["task", "source"])
    .reset_index(drop=True)
)


print("===== INTER-ANNOTATOR AGREEMENT =====")
display(inter_annotator_metrics)


print("===== OVERALL 1,200-ARTICLE A-B RESULTS =====")
display(
    inter_annotator_metrics.loc[
        inter_annotator_metrics["source"].eq("all")
    ].reset_index(drop=True)
)


# === Notebook cell 12 ===
# ============================================================
# CELL 10 — FINAL ADJUDICATION ALIGNMENT ON 1,200 OVERLAP
# ============================================================

alignment_records = []
alignment_detail_frames = []


for task in TASKS:
    a_column = f"A_{task}"
    b_column = f"B_{task}"
    final_column = f"Final_{task}"

    detail = ABF_overlap[
        [
            column
            for column in [
                KEY,
                "source",
                "source_batch",
                "title",
                a_column,
                b_column,
                final_column,
            ]
            if column in ABF_overlap.columns
        ]
    ].copy()

    a_equals_b = detail[a_column].eq(detail[b_column])
    final_equals_a = detail[final_column].eq(detail[a_column])
    final_equals_b = detail[final_column].eq(detail[b_column])

    detail["adjudication_alignment"] = np.select(
        [
            a_equals_b & final_equals_a,
            (~a_equals_b) & final_equals_a,
            (~a_equals_b) & final_equals_b,
            a_equals_b & (~final_equals_a),
        ],
        [
            "both",
            "A_only",
            "B_only",
            "neither_despite_A_B_agreement",
        ],
        default="neither",
    )

    detail["task"] = task
    detail = detail.rename(
        columns={
            a_column: "A_label",
            b_column: "B_label",
            final_column: "Final_label",
        }
    )

    alignment_detail_frames.append(detail)

    counts = (
        detail["adjudication_alignment"]
        .value_counts()
    )

    for category in [
        "both",
        "A_only",
        "B_only",
        "neither_despite_A_B_agreement",
        "neither",
    ]:
        count = int(counts.get(category, 0))

        alignment_records.append(
            {
                "task": task,
                "alignment": category,
                "n_article_task_decisions": count,
                "share": count / len(detail),
            }
        )


adjudication_alignment = pd.DataFrame(alignment_records)
adjudication_details = pd.concat(
    alignment_detail_frames,
    ignore_index=True,
)


print("===== FINAL ALIGNMENT BY TASK =====")
display(adjudication_alignment)


alignment_pivot = (
    adjudication_alignment.pivot(
        index="task",
        columns="alignment",
        values="n_article_task_decisions",
    )
    .fillna(0)
    .astype(int)
    .reset_index()
)

print("===== FINAL ALIGNMENT COUNT MATRIX =====")
display(alignment_pivot)


# === Notebook cell 13 ===
# ============================================================
# CELL 11 — ARTICLE-LEVEL FINAL ALIGNMENT
# ============================================================

article_alignment = ABF_overlap[
    [
        column
        for column in [
            KEY,
            "source",
            "source_batch",
            "title",
        ]
        if column in ABF_overlap.columns
    ]
].copy()


for task in TASKS:
    article_alignment[f"Final_matches_A_{task}"] = (
        ABF_overlap[f"Final_{task}"]
        .eq(ABF_overlap[f"A_{task}"])
    )

    article_alignment[f"Final_matches_B_{task}"] = (
        ABF_overlap[f"Final_{task}"]
        .eq(ABF_overlap[f"B_{task}"])
    )


a_match_columns = [
    f"Final_matches_A_{task}"
    for task in TASKS
]

b_match_columns = [
    f"Final_matches_B_{task}"
    for task in TASKS
]


article_alignment["n_final_labels_matching_A"] = (
    article_alignment[a_match_columns]
    .sum(axis=1)
)

article_alignment["n_final_labels_matching_B"] = (
    article_alignment[b_match_columns]
    .sum(axis=1)
)

article_alignment["all_five_match_A"] = (
    article_alignment[a_match_columns]
    .all(axis=1)
)

article_alignment["all_five_match_B"] = (
    article_alignment[b_match_columns]
    .all(axis=1)
)


article_level_summary = pd.DataFrame(
    [
        {
            "measure": "Overlap articles",
            "count": len(article_alignment),
        },
        {
            "measure": "Final matches A on all five labels",
            "count": int(article_alignment["all_five_match_A"].sum()),
        },
        {
            "measure": "Final matches B on all five labels",
            "count": int(article_alignment["all_five_match_B"].sum()),
        },
        {
            "measure": "Final matches both A and B on all five labels",
            "count": int(
                (
                    article_alignment["all_five_match_A"]
                    & article_alignment["all_five_match_B"]
                ).sum()
            ),
        },
    ]
)


print("===== ARTICLE-LEVEL ALIGNMENT =====")
display(article_level_summary)


# === Notebook cell 14 ===
# ============================================================
# CELL 12 — CONFUSION MATRICES AND DISAGREEMENTS
# ============================================================

confusion_records = []
disagreement_records = []


comparison_specs = [
    (
        "Expert A vs Final",
        A_vs_final,
        "A",
        "Final",
    ),
    (
        "Expert B vs Final",
        B_vs_final,
        "B",
        "Final",
    ),
    (
        "Expert A vs Expert B",
        ABF_overlap,
        "A",
        "B",
    ),
]


for comparison_name, frame, left_prefix, right_prefix in comparison_specs:
    for task in TASKS:
        left_column = f"{left_prefix}_{task}"
        right_column = f"{right_prefix}_{task}"

        labels = sorted(
            set(frame[left_column].dropna().astype(str))
            | set(frame[right_column].dropna().astype(str))
        )

        matrix = confusion_matrix(
            frame[right_column].astype(str),
            frame[left_column].astype(str),
            labels=labels,
        )

        for row_index, right_label in enumerate(labels):
            for column_index, left_label in enumerate(labels):
                confusion_records.append(
                    {
                        "comparison": comparison_name,
                        "task": task,
                        "reference_label": right_label,
                        "comparison_label": left_label,
                        "count": int(matrix[row_index, column_index]),
                    }
                )

        disagreement_mask = frame[left_column].ne(frame[right_column])

        disagreement_columns = [
            column
            for column in [
                KEY,
                "source",
                "source_batch",
                "title",
                left_column,
                right_column,
            ]
            if column in frame.columns
        ]

        disagreement_subset = frame.loc[
            disagreement_mask,
            disagreement_columns,
        ].copy()

        disagreement_subset["comparison"] = comparison_name
        disagreement_subset["task"] = task
        disagreement_subset = disagreement_subset.rename(
            columns={
                left_column: "comparison_label",
                right_column: "reference_label",
            }
        )

        disagreement_records.append(disagreement_subset)


confusion_long = pd.DataFrame(confusion_records)
disagreements = pd.concat(
    disagreement_records,
    ignore_index=True,
    sort=False,
)


disagreement_summary = (
    disagreements
    .groupby(
        ["comparison", "task"],
        dropna=False,
    )
    .size()
    .rename("n_disagreements")
    .reset_index()
)


print("===== DISAGREEMENT SUMMARY =====")
display(disagreement_summary)

print("===== FIRST 200 DISAGREEMENTS =====")
display(disagreements.head(200))


# === Notebook cell 15 ===
# ============================================================
# CELL 13 — EXPORT THE EVALUATION WORKBOOK
# ============================================================

with pd.ExcelWriter(
    RESULTS_PATH,
    engine="openpyxl",
) as writer:
    qa_summary.to_excel(
        writer,
        sheet_name="File_QA",
        index=False,
    )

    provenance_summary.to_excel(
        writer,
        sheet_name="Provenance_Info",
        index=False,
    )

    set_qa.to_excel(
        writer,
        sheet_name="Set_QA",
        index=False,
    )

    expert_vs_final_metrics.to_excel(
        writer,
        sheet_name="Expert_vs_Final",
        index=False,
    )

    inter_annotator_metrics.to_excel(
        writer,
        sheet_name="A_vs_B_Agreement",
        index=False,
    )

    adjudication_alignment.to_excel(
        writer,
        sheet_name="Final_Alignment",
        index=False,
    )

    alignment_pivot.to_excel(
        writer,
        sheet_name="Alignment_Matrix",
        index=False,
    )

    article_level_summary.to_excel(
        writer,
        sheet_name="Article_Summary",
        index=False,
    )

    article_alignment.to_excel(
        writer,
        sheet_name="Article_Alignment",
        index=False,
    )

    confusion_long.to_excel(
        writer,
        sheet_name="Confusion_Long",
        index=False,
    )

    disagreement_summary.to_excel(
        writer,
        sheet_name="Disagreement_Summary",
        index=False,
    )

    disagreements.to_excel(
        writer,
        sheet_name="Disagreements",
        index=False,
    )

    adjudication_details.to_excel(
        writer,
        sheet_name="Adjudication_Detail",
        index=False,
    )

    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions


if not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0:
    raise IOError("Evaluation workbook export failed.")

print("Saved:", RESULTS_PATH)
print("All-2,000-row evaluation: COMPLETE")


# === Notebook cell 17 ===
