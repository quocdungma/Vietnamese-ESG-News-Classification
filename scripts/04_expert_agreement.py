# Extracted from original Colab notebook. Review paths and execute cells in order.


# === Notebook cell 2 ===

from google.colab import drive
drive.mount("/content/gdrive")

# === Notebook cell 3 ===
# CELL 1 — IMPORTS AND CONFIGURATION
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import cohen_kappa_score

EXPORT_DIR = Path(
    '/content/gdrive/MyDrive/Project_Fin_Data_Crawling/'
    'Crawling_news/ESG_annotation_sampling/'
    'annotation_reliability_expert_assignment/'
    'reconstructed_expert_datasets'
)

FILE_A = EXPORT_DIR / 'ESG_Master_Expert_A.xlsx'
FILE_B = EXPORT_DIR / 'ESG_Master_Expert_B.xlsx'
FILE_FINAL = EXPORT_DIR / 'ESG_Final_Full_Labels.xlsx'

KEY = 'assignment_id'
TASKS = ['esg_relevance', 'sentiment', 'environmental', 'social', 'governance']
EXPECTED_A = 2000
EXPECTED_B = 2000
EXPECTED_FINAL = 2800
EXPECTED_SHARED = 1200

print('Configuration loaded.')

# === Notebook cell 4 ===
# ============================================================
# VALIDATE AND EXTRACT THE 1,200 SHARED ARTICLES
# ============================================================

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# article_key is the canonical cross-file identifier.
# assignment_id is not required because exclusive articles
# were not part of the reliability assignment.

KEY = "article_key"

EXPECTED_EXPERT_ROWS = 2000
EXPECTED_SHARED_ROWS = 1200

RELIABILITY_TASKS = [
    "esg_relevance",
    "sentiment",
    "environmental",
    "social",
    "governance",
]


# Change only these two names if the loading cell used
# different DataFrame variable names.
expert_files = {
    "Expert A": expert_a,
    "Expert B": expert_b,
}


# ------------------------------------------------------------
# Normalize shared/reliability indicator
# ------------------------------------------------------------
def normalize_boolean(
    series,
):
    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

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


# ------------------------------------------------------------
# Validate each complete expert master
# ------------------------------------------------------------
qa_rows = []
shared_frames = {}


for name, original_df in expert_files.items():

    df = original_df.copy()

    required_columns = [
        KEY,
        "is_shared_for_reliability",
        *RELIABILITY_TASKS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} thiếu cột: "
            f"{missing_columns}"
        )

    # Normalize article key.
    df[
        KEY
    ] = (
        df[
            KEY
        ]
        .astype("string")
        .str.strip()
    )

    missing_key_mask = (
        df[
            KEY
        ].isna()
        | df[
            KEY
        ].eq("")
    )

    if missing_key_mask.any():
        raise ValueError(
            f"{name} có "
            f"{int(missing_key_mask.sum())} "
            "article_key bị thiếu."
        )

    if df[
        KEY
    ].duplicated().any():

        duplicated_rows = df.loc[
            df[
                KEY
            ].duplicated(
                keep=False
            )
        ].sort_values(
            KEY
        )

        display(
            duplicated_rows.head(100)
        )

        raise ValueError(
            f"{name} có article_key trùng lặp."
        )

    if len(df) != EXPECTED_EXPERT_ROWS:
        raise ValueError(
            f"{name} có {len(df)} bài; "
            f"cần đúng "
            f"{EXPECTED_EXPERT_ROWS}."
        )

    # Normalize reliability indicator.
    df[
        "is_shared_for_reliability"
    ] = normalize_boolean(
        df[
            "is_shared_for_reliability"
        ]
    )

    # Retain only the shared reliability sample.
    shared_df = (
        df.loc[
            df[
                "is_shared_for_reliability"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        len(shared_df)
# COLAB COMMAND:         != EXPECTED_SHARED_ROWS
    ):
        raise ValueError(
            f"{name} có "
            f"{len(shared_df)} bài chung; "
            f"cần đúng "
            f"{EXPECTED_SHARED_ROWS}."
        )

    if shared_df[
        KEY
    ].duplicated().any():
        raise ValueError(
            f"{name} có article_key trùng "
            "trong mẫu reliability."
        )

    shared_frames[
        name
    ] = shared_df

    qa_rows.append(
        {
            "file": name,
            "total_rows": len(df),
            "shared_rows": len(
                shared_df
            ),
            "exclusive_rows": (
                len(df)
                - len(shared_df)
            ),
            "unique_article_keys": (
                df[
                    KEY
                ].nunique()
            ),
        }
    )


expert_master_qa = pd.DataFrame(
    qa_rows
)

display(
    expert_master_qa
)


# ------------------------------------------------------------
# Assign standardized shared frames
# ------------------------------------------------------------
shared_a = shared_frames[
    "Expert A"
].copy()

shared_b = shared_frames[
    "Expert B"
].copy()


# ------------------------------------------------------------
# Confirm both experts contain the same 1,200 articles
# ------------------------------------------------------------
shared_keys_a = set(
    shared_a[
        KEY
    ]
)

shared_keys_b = set(
    shared_b[
        KEY
    ]
)

only_in_a = (
    shared_keys_a
    - shared_keys_b
)

only_in_b = (
    shared_keys_b
    - shared_keys_a
)

shared_key_intersection = (
    shared_keys_a
    & shared_keys_b
)


print()
print(
    "Expert A shared articles:",
    len(shared_keys_a),
)

print(
    "Expert B shared articles:",
    len(shared_keys_b),
)

print(
    "Shared-key intersection:",
    len(shared_key_intersection),
)

print(
    "Only in Expert A:",
    len(only_in_a),
)

print(
    "Only in Expert B:",
    len(only_in_b),
)


if only_in_a or only_in_b:

    mismatch_records = []

    for article_key in sorted(
        only_in_a
    ):
        mismatch_records.append(
            {
                "article_key": article_key,
                "location": "only_in_expert_A",
            }
        )

    for article_key in sorted(
        only_in_b
    ):
        mismatch_records.append(
            {
                "article_key": article_key,
                "location": "only_in_expert_B",
            }
        )

    shared_key_mismatches = (
        pd.DataFrame(
            mismatch_records
        )
    )

    display(
        shared_key_mismatches
    )

    raise ValueError(
        "Hai expert không có cùng một tập "
        "1.200 bài reliability."
    )


if (
    len(shared_key_intersection)
# COLAB COMMAND:     != EXPECTED_SHARED_ROWS
):
    raise ValueError(
        "Giao của hai master không bằng "
        f"{EXPECTED_SHARED_ROWS} bài."
    )


# ------------------------------------------------------------
# Sort and align rows before later evaluation
# ------------------------------------------------------------
shared_a = (
    shared_a
    .sort_values(
        KEY
    )
    .reset_index(
        drop=True
    )
)

shared_b = (
    shared_b
    .sort_values(
        KEY
    )
    .reset_index(
        drop=True
    )
)


if not shared_a[
    KEY
].equals(
    shared_b[
        KEY
    ]
):
    raise AssertionError(
        "article_key của A và B chưa được "
        "căn chỉnh cùng thứ tự."
    )


print()
print(
    "Shared reliability sample QA: PASS"
)

print(
    "Join key:",
    KEY,
)

print(
    "Articles ready for later evaluation:",
    len(shared_a),
)

# === Notebook cell 5 ===
# CELL 3 — NORMALIZE LABELS AND BUILD THE SHARED A–B SAMPLE
def normalize_string(series):
    return (series.astype('string').str.strip().str.lower()
            .str.replace(r'\s+', '_', regex=True)
            .str.replace('-', '_', regex=False))

def normalize_relevance(series):
    values = normalize_string(series)
    mapping = {
        'esg': 'relevant', 'relevant': 'relevant',
        'non_esg': 'not_relevant', 'nonesg': 'not_relevant',
        'not_relevant': 'not_relevant', 'irrelevant': 'not_relevant',
        'uncertain': 'uncertain', 'needs_review': 'uncertain',
    }
    return values.map(mapping).fillna(values)

def normalize_sentiment(series):
    values = normalize_string(series)
    mapping = {
        'positive': 'positive', 'pos': 'positive',
        'negative': 'negative', 'neg': 'negative',
        'mixed': 'mixed', 'neutral': 'mixed', 'mix': 'mixed',
    }
    return values.map(mapping).fillna(values)

def normalize_binary(series):
    values = normalize_string(series)
    mapping = {
        '1': 1, '1.0': 1, 'true': 1, 'yes': 1, 'y': 1,
        '0': 0, '0.0': 0, 'false': 0, 'no': 0, 'n': 0,
    }
    return values.map(mapping).astype('Int64')

def normalize_task(series, task):
    if task == 'esg_relevance':
        return normalize_relevance(series)
    if task == 'sentiment':
        return normalize_sentiment(series)
    return normalize_binary(series)

shared_ab = expert_a[[KEY] + TASKS].merge(
    expert_b[[KEY] + TASKS],
    on=KEY, how='inner', suffixes=('_A', '_B'), validate='one_to_one'
)

if len(shared_ab) != EXPECTED_SHARED:
    raise ValueError(f'A–B shared = {len(shared_ab):,}; expected = {EXPECTED_SHARED:,}.')

for task in TASKS:
    shared_ab[f'{task}_A'] = normalize_task(shared_ab[f'{task}_A'], task)
    shared_ab[f'{task}_B'] = normalize_task(shared_ab[f'{task}_B'], task)

VALID_LABELS = {
    'esg_relevance': {'relevant', 'not_relevant', 'uncertain'},
    'sentiment': {'positive', 'negative', 'mixed'},
    'environmental': {0, 1}, 'social': {0, 1}, 'governance': {0, 1},
}

invalid = []
for task in TASKS:
    for expert in ['A', 'B']:
        col = f'{task}_{expert}'
        mask = shared_ab[col].isna() | ~shared_ab[col].isin(VALID_LABELS[task])
        if mask.any():
            invalid.append({'task': task, 'expert': expert, 'n_invalid': int(mask.sum()),
                            'values': shared_ab.loc[mask, col].astype('string').value_counts(dropna=False).to_dict()})

if invalid:
    display(pd.DataFrame(invalid))
    raise ValueError('Phát hiện nhãn không hợp lệ sau chuẩn hóa.')

print(f'Shared A–B sample: {len(shared_ab):,}')
print('CELL 3 SHARED-SAMPLE QA: PASS')

# === Notebook cell 6 ===
# CELL 4 — AGREEMENT RATE, COHEN'S KAPPA AND CONFUSION MATRICES
def interpret_kappa(kappa):
    if pd.isna(kappa): return 'not_defined'
    if kappa < 0: return 'poor'
    if kappa < 0.21: return 'slight'
    if kappa < 0.41: return 'fair'
    if kappa < 0.61: return 'moderate'
    if kappa < 0.81: return 'substantial'
    return 'almost_perfect'

agreement_rows = []
confusion_tables = {}

for task in TASKS:
    a = shared_ab[f'{task}_A']
    b = shared_ab[f'{task}_B']
    valid = a.notna() & b.notna()
    a, b = a[valid], b[valid]
    n = len(a)
    n_agree = int((a == b).sum())
    kappa = cohen_kappa_score(a.astype(str), b.astype(str)) if len(set(a) | set(b)) > 1 else np.nan
    agreement_rows.append({
        'task': task, 'n_compared': n, 'n_agree': n_agree,
        'n_disagree': n - n_agree, 'agreement_rate': n_agree / n,
        'cohen_kappa': kappa, 'kappa_interpretation': interpret_kappa(kappa),
    })
    confusion_tables[task] = pd.crosstab(
        a, b, rownames=['Expert A'], colnames=['Expert B'], dropna=False
    )

agreement_summary = pd.DataFrame(agreement_rows)
display(agreement_summary.style.format({'agreement_rate': '{:.4f}', 'cohen_kappa': '{:.4f}'}))

for task in TASKS:
    print(f'\n--- {task} ---')
    display(confusion_tables[task])

if not (agreement_summary['n_compared'] == EXPECTED_SHARED).all():
    raise ValueError('Không phải mọi task đều dùng đủ 1.200 bài chung.')

print('\nCELL 4 INTER-ANNOTATOR AGREEMENT: PASS')
print('Không có file mới được xuất; các kết quả nằm trong agreement_summary và confusion_tables.')

# === Notebook cell 7 ===
# ============================================================
# FULL DATA QA
# EXPERT A, EXPERT B, AND FINAL CORPUS
#
# Scope:
# - Structural QA
# - Allocation QA
# - Key-set QA
# - Provenance QA
# - Label completeness
# - Valid label domains
# - Cross-label ESG logic
#
# Excluded from PASS/FAIL:
# - evidence_text
# - annotator_note
# - agreement statistics
# - Cohen's kappa
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display


# ------------------------------------------------------------
# 1. Locate the three DataFrames
# ------------------------------------------------------------
def find_dataframe(
    candidate_names,
):
    for variable_name in candidate_names:

        if (
            variable_name in globals()
            and isinstance(
                globals()[
                    variable_name
                ],
                pd.DataFrame,
            )
        ):
            return (
                globals()[
                    variable_name
                ].copy()
            )

    return None


A = find_dataframe(
    [
        "expert_a",
        "master_expert_a",
        "df_a",
        "A",
    ]
)

B = find_dataframe(
    [
        "expert_b",
        "master_expert_b",
        "df_b",
        "B",
    ]
)

F = find_dataframe(
    [
        "final_full_labels",
        "final_labels",
        "final_corpus",
        "final_df",
        "F",
    ]
)


if A is None:
    raise NameError(
        "Không tìm thấy DataFrame Expert A."
    )

if B is None:
    raise NameError(
        "Không tìm thấy DataFrame Expert B."
    )


# If the final DataFrame is not yet in memory,
# load it from FINAL_PATH when available.
if F is None:

    if (
        "FINAL_PATH"
        not in globals()
    ):
        raise NameError(
            "Không tìm thấy DataFrame final và "
            "FINAL_PATH chưa được khai báo."
        )

    final_path = Path(
        FINAL_PATH
    )

    if not final_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy file final tại: "
            f"{final_path}"
        )

    F = pd.read_excel(
        final_path,
        sheet_name=(
            "Final_Full_Labels"
        ),
    )


print(
    "Loaded Expert A:",
    A.shape,
)

print(
    "Loaded Expert B:",
    B.shape,
)

print(
    "Loaded Final:",
    F.shape,
)


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------
KEY = "article_key"

TASKS = [
    "esg_relevance",
    "sentiment",
    "environmental",
    "social",
    "governance",
]


EXPECTED = {
    "expert_rows": 2000,
    "final_rows": 2800,

    "shared_rows_per_expert": 1200,
    "exclusive_rows_per_expert": 800,

    "direct_new_per_expert": 600,

    "expert_cafef": 800,
    "expert_vietstock": 1200,

    "final_cafef": 800,
    "final_vietstock": 2000,

    "shared_cafef": 800,
    "shared_vietstock": 400,
    "exclusive_vietstock": 800,

    "direct_cafef_per_expert": 400,
    "direct_vietstock_per_expert": 200,

    "ab_intersection": 1200,
    "ab_union": 2800,
}


VALID_CATEGORICAL_LABELS = {
    "esg_relevance": {
        "relevant",
        "not_relevant",
        "uncertain",
    },
    "sentiment": {
        "positive",
        "negative",
        "mixed",
        "neutral",
    },
}


VALID_BINARY_LABELS = {
    "environmental": {
        0,
        1,
    },
    "social": {
        0,
        1,
    },
    "governance": {
        0,
        1,
    },
}


EXPECTED_EXPERT_PROVENANCE = {
    "new_returned_annotation": 600,
    "current_done_final_label": 600,
    "pre_existing_current_final_label": 800,
}


EXPECTED_FINAL_PROVENANCE = {
    "authoritative_current_done_label": 2800,
}


# ------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------
def normalize_text(
    series,
):
    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def missing_mask(
    series,
):
    normalized = normalize_text(
        series
    )

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


def normalize_boolean(
    series,
):
    normalized = normalize_text(
        series
    )

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


def normalize_binary(
    series,
):
    normalized = (
        normalize_text(
            series
        )
        .replace(
            {
                "0.0": "0",
                "1.0": "1",
                "false": "0",
                "true": "1",
            }
        )
    )

    return pd.to_numeric(
        normalized,
        errors="coerce",
    )


def canonicalize_esg_relevance(
    series,
):
    """
    Treat:
        esg         == relevant
        non_esg     == not_relevant

    Unknown values are retained so the valid-domain
    check can still detect them.
    """

    return (
        normalize_text(
            series
        )
        .replace(
            {
                "esg": "relevant",
                "non_esg": "not_relevant",
                "non-esg": "not_relevant",
                "relevant": "relevant",
                "not_relevant": "not_relevant",
                "not relevant": "not_relevant",
                "uncertain": "uncertain",
            }
        )
    )


def value_count_dict(
    series,
):
    return (
        normalize_text(
            series
        )
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_dict()
    )


def source_count_dict(
    frame,
):
    return (
        normalize_text(
            frame[
                "source"
            ]
        )
        .value_counts()
        .to_dict()
    )


def add_check(
    records,
    check_group,
    check_name,
    observed,
    expected,
    passed,
    detail="",
):
    records.append(
        {
            "check_group": (
                check_group
            ),
            "check_name": (
                check_name
            ),
            "observed": observed,
            "expected": expected,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "detail": detail,
        }
    )


# ------------------------------------------------------------
# 4. Work only on QA copies
# ------------------------------------------------------------
datasets = {
    "Expert A": A,
    "Expert B": B,
    "Final": F,
}


for name, frame in (
    datasets.items()
):

    # Normalize key.
    if KEY in frame.columns:

        frame[
            KEY
        ] = (
            frame[
                KEY
            ]
            .astype("string")
            .str.strip()
        )

    # Preserve original ESG-relevance value.
    if (
        "esg_relevance_raw"
        not in frame.columns
        and "esg_relevance"
        in frame.columns
    ):
        frame[
            "esg_relevance_raw"
        ] = frame[
            "esg_relevance"
        ].copy()

    # If rerunning the cell, use raw values as source.
    if (
        "esg_relevance_raw"
        in frame.columns
    ):
        frame[
            "esg_relevance"
        ] = (
            canonicalize_esg_relevance(
                frame[
                    "esg_relevance_raw"
                ]
            )
        )

    # Normalize Boolean metadata.
    if (
        "is_shared_for_reliability"
        in frame.columns
    ):
        frame[
            "is_shared_for_reliability"
        ] = normalize_boolean(
            frame[
                "is_shared_for_reliability"
            ]
        )

    if (
        "is_direct_new_annotation"
        in frame.columns
    ):
        frame[
            "is_direct_new_annotation"
        ] = normalize_boolean(
            frame[
                "is_direct_new_annotation"
            ]
        )


# ------------------------------------------------------------
# 5. QA containers
# ------------------------------------------------------------
qa_records = []
issue_tables = {}


# ------------------------------------------------------------
# 6. Required-column QA
# ------------------------------------------------------------
# evidence_text and annotator_note are intentionally excluded.

EXPERT_REQUIRED_COLUMNS = [
    KEY,
    "source",
    *TASKS,
    "dataset_expert_id",
    "is_direct_new_annotation",
    "label_provenance",
    "coverage_type",
    "is_shared_for_reliability",
]


FINAL_REQUIRED_COLUMNS = [
    KEY,
    "source",
    *TASKS,
    "label_provenance",
    "authoritative_final_source",
]


for name, frame in (
    datasets.items()
):

    if name == "Final":
        required_columns = (
            FINAL_REQUIRED_COLUMNS
        )
    else:
        required_columns = (
            EXPERT_REQUIRED_COLUMNS
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    add_check(
        qa_records,
        "Required columns",
        name,
        observed=(
            "All present"
            if not missing_columns
            else missing_columns
        ),
        expected=(
            "All required columns"
        ),
        passed=(
            len(
                missing_columns
            )
            == 0
        ),
    )


required_failures = [
    record
    for record in qa_records
    if (
        record[
            "check_group"
        ]
        == "Required columns"
        and record[
            "status"
        ]
        == "FAIL"
    )
]


if required_failures:

    display(
        pd.DataFrame(
            required_failures
        )
    )

    raise ValueError(
        "Thiếu cột bắt buộc. "
        "Không thể tiếp tục QA."
    )


# ------------------------------------------------------------
# 7. File-size and key QA
# ------------------------------------------------------------
expected_rows = {
    "Expert A": EXPECTED[
        "expert_rows"
    ],
    "Expert B": EXPECTED[
        "expert_rows"
    ],
    "Final": EXPECTED[
        "final_rows"
    ],
}


for name, frame in (
    datasets.items()
):

    add_check(
        qa_records,
        "File structure",
        f"{name}: row count",
        observed=len(frame),
        expected=expected_rows[
            name
        ],
        passed=(
            len(frame)
            == expected_rows[
                name
            ]
        ),
    )

    key_missing = missing_mask(
        frame[
            KEY
        ]
    )

    add_check(
        qa_records,
        "File structure",
        f"{name}: missing article_key",
        observed=int(
            key_missing.sum()
        ),
        expected=0,
        passed=(
            not key_missing.any()
        ),
    )

    duplicate_key = (
        frame[
            KEY
        ].duplicated(
            keep=False
        )
    )

    add_check(
        qa_records,
        "File structure",
        f"{name}: duplicate article_key",
        observed=int(
            duplicate_key.sum()
        ),
        expected=0,
        passed=(
            not duplicate_key.any()
        ),
    )

    if duplicate_key.any():

        issue_tables[
            f"{name}: duplicated keys"
        ] = (
            frame.loc[
                duplicate_key
            ]
            .sort_values(
                KEY
            )
        )


# ------------------------------------------------------------
# 8. Source-distribution QA
# ------------------------------------------------------------
expected_expert_sources = {
    "cafef": EXPECTED[
        "expert_cafef"
    ],
    "vietstock": EXPECTED[
        "expert_vietstock"
    ],
}

expected_final_sources = {
    "cafef": EXPECTED[
        "final_cafef"
    ],
    "vietstock": EXPECTED[
        "final_vietstock"
    ],
}


for name, frame in [
    (
        "Expert A",
        A,
    ),
    (
        "Expert B",
        B,
    ),
]:

    observed_sources = (
        source_count_dict(
            frame
        )
    )

    add_check(
        qa_records,
        "Source distribution",
        f"{name}: source counts",
        observed_sources,
        expected_expert_sources,
        observed_sources
        == expected_expert_sources,
    )


final_sources = source_count_dict(
    F
)

add_check(
    qa_records,
    "Source distribution",
    "Final: source counts",
    final_sources,
    expected_final_sources,
    final_sources
    == expected_final_sources,
)


# ------------------------------------------------------------
# 9. Create shared and exclusive subsets
# ------------------------------------------------------------
shared_a = A.loc[
    A[
        "is_shared_for_reliability"
    ]
].copy()

shared_b = B.loc[
    B[
        "is_shared_for_reliability"
    ]
].copy()

exclusive_a = A.loc[
    ~A[
        "is_shared_for_reliability"
    ]
].copy()

exclusive_b = B.loc[
    ~B[
        "is_shared_for_reliability"
    ]
].copy()


allocation_counts = {
    "Expert A shared": len(
        shared_a
    ),
    "Expert A exclusive": len(
        exclusive_a
    ),
    "Expert B shared": len(
        shared_b
    ),
    "Expert B exclusive": len(
        exclusive_b
    ),
}

expected_allocation_counts = {
    "Expert A shared": EXPECTED[
        "shared_rows_per_expert"
    ],
    "Expert A exclusive": EXPECTED[
        "exclusive_rows_per_expert"
    ],
    "Expert B shared": EXPECTED[
        "shared_rows_per_expert"
    ],
    "Expert B exclusive": EXPECTED[
        "exclusive_rows_per_expert"
    ],
}


add_check(
    qa_records,
    "Allocation",
    "Shared and exclusive counts",
    allocation_counts,
    expected_allocation_counts,
    allocation_counts
    == expected_allocation_counts,
)


# ------------------------------------------------------------
# 10. Coverage-type QA
# ------------------------------------------------------------
expected_coverage_counts = {
    "shared_cafef": EXPECTED[
        "shared_cafef"
    ],
    "shared_vietstock": EXPECTED[
        "shared_vietstock"
    ],
    "exclusive_vietstock": EXPECTED[
        "exclusive_vietstock"
    ],
}


for name, frame in [
    (
        "Expert A",
        A,
    ),
    (
        "Expert B",
        B,
    ),
]:

    observed_coverage = (
        value_count_dict(
            frame[
                "coverage_type"
            ]
        )
    )

    add_check(
        qa_records,
        "Allocation",
        f"{name}: coverage types",
        observed_coverage,
        expected_coverage_counts,
        observed_coverage
        == expected_coverage_counts,
    )


# ------------------------------------------------------------
# 11. Shared-source and exclusive-source QA
# ------------------------------------------------------------
expected_shared_sources = {
    "cafef": EXPECTED[
        "shared_cafef"
    ],
    "vietstock": EXPECTED[
        "shared_vietstock"
    ],
}

expected_exclusive_sources = {
    "vietstock": EXPECTED[
        "exclusive_vietstock"
    ],
}


for name, shared_frame, exclusive_frame in [
    (
        "Expert A",
        shared_a,
        exclusive_a,
    ),
    (
        "Expert B",
        shared_b,
        exclusive_b,
    ),
]:

    observed_shared_sources = (
        source_count_dict(
            shared_frame
        )
    )

    observed_exclusive_sources = (
        source_count_dict(
            exclusive_frame
        )
    )

    add_check(
        qa_records,
        "Allocation",
        f"{name}: shared source counts",
        observed_shared_sources,
        expected_shared_sources,
        observed_shared_sources
        == expected_shared_sources,
    )

    add_check(
        qa_records,
        "Allocation",
        f"{name}: exclusive source counts",
        observed_exclusive_sources,
        expected_exclusive_sources,
        observed_exclusive_sources
        == expected_exclusive_sources,
    )


# ------------------------------------------------------------
# 12. Key-set relationship QA
# ------------------------------------------------------------
keys_a = set(
    A[
        KEY
    ]
)

keys_b = set(
    B[
        KEY
    ]
)

keys_f = set(
    F[
        KEY
    ]
)

shared_keys_a = set(
    shared_a[
        KEY
    ]
)

shared_keys_b = set(
    shared_b[
        KEY
    ]
)

exclusive_keys_a = set(
    exclusive_a[
        KEY
    ]
)

exclusive_keys_b = set(
    exclusive_b[
        KEY
    ]
)

ab_intersection = (
    keys_a
    & keys_b
)

ab_union = (
    keys_a
    | keys_b
)

exclusive_overlap = (
    exclusive_keys_a
    & exclusive_keys_b
)


add_check(
    qa_records,
    "Key relationships",
    "A–B intersection",
    len(
        ab_intersection
    ),
    EXPECTED[
        "ab_intersection"
    ],
    len(
        ab_intersection
    )
    == EXPECTED[
        "ab_intersection"
    ],
)

add_check(
    qa_records,
    "Key relationships",
    "A–B union",
    len(
        ab_union
    ),
    EXPECTED[
        "ab_union"
    ],
    len(
        ab_union
    )
    == EXPECTED[
        "ab_union"
    ],
)

add_check(
    qa_records,
    "Key relationships",
    "Shared keys A equal shared keys B",
    len(
        shared_keys_a
        ^ shared_keys_b
    ),
    0,
    shared_keys_a
    == shared_keys_b,
)

add_check(
    qa_records,
    "Key relationships",
    "Shared keys equal A–B intersection",
    len(
        shared_keys_a
        ^ ab_intersection
    ),
    0,
    shared_keys_a
    == ab_intersection,
)

add_check(
    qa_records,
    "Key relationships",
    "Exclusive A–B overlap",
    len(
        exclusive_overlap
    ),
    0,
    len(
        exclusive_overlap
    )
    == 0,
)

add_check(
    qa_records,
    "Key relationships",
    "A–B union equals Final",
    len(
        ab_union
        ^ keys_f
    ),
    0,
    ab_union
    == keys_f,
)

add_check(
    qa_records,
    "Key relationships",
    "Expert A keys contained in Final",
    len(
        keys_a
        - keys_f
    ),
    0,
    keys_a.issubset(
        keys_f
    ),
)

add_check(
    qa_records,
    "Key relationships",
    "Expert B keys contained in Final",
    len(
        keys_b
        - keys_f
    ),
    0,
    keys_b.issubset(
        keys_f
    ),
)


# ------------------------------------------------------------
# 13. Expert identity QA
# ------------------------------------------------------------
for expert_id, name, frame in [
    (
        "A",
        "Expert A",
        A,
    ),
    (
        "B",
        "Expert B",
        B,
    ),
]:

    observed_ids = set(
        frame[
            "dataset_expert_id"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    add_check(
        qa_records,
        "Expert identity",
        f"{name}: dataset_expert_id",
        observed_ids,
        {
            expert_id,
        },
        observed_ids
        == {
            expert_id,
        },
    )

    direct_count = int(
        frame[
            "is_direct_new_annotation"
        ].sum()
    )

    add_check(
        qa_records,
        "Expert identity",
        f"{name}: direct new annotations",
        direct_count,
        EXPECTED[
            "direct_new_per_expert"
        ],
        direct_count
        == EXPECTED[
            "direct_new_per_expert"
        ],
    )

    direct_exclusive = int(
        frame.loc[
            ~frame[
                "is_shared_for_reliability"
            ],
            "is_direct_new_annotation",
        ].sum()
    )

    add_check(
        qa_records,
        "Expert identity",
        f"{name}: direct labels in exclusive subset",
        direct_exclusive,
        0,
        direct_exclusive == 0,
    )

    direct_subset = frame.loc[
        frame[
            "is_direct_new_annotation"
        ]
    ]

    direct_source_counts = (
        source_count_dict(
            direct_subset
        )
    )

    expected_direct_sources = {
        "cafef": EXPECTED[
            "direct_cafef_per_expert"
        ],
        "vietstock": EXPECTED[
            "direct_vietstock_per_expert"
        ],
    }

    add_check(
        qa_records,
        "Expert identity",
        f"{name}: direct annotation sources",
        direct_source_counts,
        expected_direct_sources,
        direct_source_counts
        == expected_direct_sources,
    )


# ------------------------------------------------------------
# 14. Provenance QA
# ------------------------------------------------------------
provenance_records = []


for name, frame in (
    datasets.items()
):

    provenance_counts = (
        value_count_dict(
            frame[
                "label_provenance"
            ]
        )
    )

    for provenance, count in (
        provenance_counts.items()
    ):
        provenance_records.append(
            {
                "file": name,
                "label_provenance": (
                    provenance
                ),
                "n_rows": int(
                    count
                ),
            }
        )

    if name == "Final":
        expected_provenance = (
            EXPECTED_FINAL_PROVENANCE
        )
    else:
        expected_provenance = (
            EXPECTED_EXPERT_PROVENANCE
        )

    add_check(
        qa_records,
        "Provenance",
        f"{name}: provenance counts",
        provenance_counts,
        expected_provenance,
        provenance_counts
        == expected_provenance,
    )


provenance_summary = pd.DataFrame(
    provenance_records
)


# Final authoritative source
final_source_marker = (
    value_count_dict(
        F[
            "authoritative_final_source"
        ]
    )
)

expected_final_source_marker = {
    "current_done": EXPECTED[
        "final_rows"
    ]
}


add_check(
    qa_records,
    "Provenance",
    "Final authoritative source",
    final_source_marker,
    expected_final_source_marker,
    final_source_marker
    == expected_final_source_marker,
)


# ------------------------------------------------------------
# 15. Label completeness QA
# ------------------------------------------------------------
label_missing_records = []


for name, frame in (
    datasets.items()
):

    for task in TASKS:

        task_missing = missing_mask(
            frame[
                task
            ]
        )

        n_missing = int(
            task_missing.sum()
        )

        label_missing_records.append(
            {
                "file": name,
                "task": task,
                "n_missing": n_missing,
            }
        )

        add_check(
            qa_records,
            "Label completeness",
            f"{name}: {task}",
            n_missing,
            0,
            n_missing == 0,
        )

        if task_missing.any():

            issue_columns = [
                column
                for column in [
                    KEY,
                    "source",
                    "source_batch",
                    "title",
                    task,
                ]
                if column
                in frame.columns
            ]

            issue_tables[
                f"{name}: missing {task}"
            ] = frame.loc[
                task_missing,
                issue_columns,
            ]


label_missing_summary = pd.DataFrame(
    label_missing_records
)


# ------------------------------------------------------------
# 16. Valid label-domain QA
# ------------------------------------------------------------
invalid_label_frames = []


for name, frame in (
    datasets.items()
):

    # Categorical tasks
    for task, allowed_values in (
        VALID_CATEGORICAL_LABELS.items()
    ):

        normalized_values = (
            normalize_text(
                frame[
                    task
                ]
            )
        )

        invalid_mask = (
            ~missing_mask(
                frame[
                    task
                ]
            )
            & ~normalized_values.isin(
                allowed_values
            )
        )

        invalid_values = (
            normalized_values.loc[
                invalid_mask
            ]
            .value_counts()
            .to_dict()
        )

        add_check(
            qa_records,
            "Valid label domains",
            f"{name}: {task}",
            invalid_values,
            "No invalid values",
            not invalid_mask.any(),
        )

        if invalid_mask.any():

            invalid_rows = frame.loc[
                invalid_mask,
                [
                    column
                    for column in [
                        KEY,
                        "source",
                        "source_batch",
                        "title",
                        task,
                    ]
                    if column
                    in frame.columns
                ],
            ].copy()

            invalid_rows[
                "file"
            ] = name

            invalid_rows[
                "invalid_task"
            ] = task

            invalid_label_frames.append(
                invalid_rows
            )

    # Binary E/S/G tasks
    for task, allowed_values in (
        VALID_BINARY_LABELS.items()
    ):

        normalized_values = (
            normalize_binary(
                frame[
                    task
                ]
            )
        )

        invalid_mask = (
            ~missing_mask(
                frame[
                    task
                ]
            )
            & ~normalized_values.isin(
                allowed_values
            )
        )

        invalid_values = (
            frame.loc[
                invalid_mask,
                task,
            ]
            .astype("string")
            .value_counts()
            .to_dict()
        )

        add_check(
            qa_records,
            "Valid label domains",
            f"{name}: {task}",
            invalid_values,
            "Only 0 or 1",
            not invalid_mask.any(),
        )

        if invalid_mask.any():

            invalid_rows = frame.loc[
                invalid_mask,
                [
                    column
                    for column in [
                        KEY,
                        "source",
                        "source_batch",
                        "title",
                        task,
                    ]
                    if column
                    in frame.columns
                ],
            ].copy()

            invalid_rows[
                "file"
            ] = name

            invalid_rows[
                "invalid_task"
            ] = task

            invalid_label_frames.append(
                invalid_rows
            )


if invalid_label_frames:

    invalid_labels = pd.concat(
        invalid_label_frames,
        ignore_index=True,
        sort=False,
    )

    issue_tables[
        "Invalid label values"
    ] = invalid_labels

else:

    invalid_labels = pd.DataFrame()


# ------------------------------------------------------------
# 17. Cross-label ESG logic QA
# ------------------------------------------------------------
# Evidence and annotator_note are intentionally not checked.

logic_issue_frames = []


for name, frame in (
    datasets.items()
):

    relevance = normalize_text(
        frame[
            "esg_relevance"
        ]
    )

    environmental = (
        normalize_binary(
            frame[
                "environmental"
            ]
        )
    )

    social = (
        normalize_binary(
            frame[
                "social"
            ]
        )
    )

    governance = (
        normalize_binary(
            frame[
                "governance"
            ]
        )
    )

    pillar_sum = (
        environmental.fillna(0)
        + social.fillna(0)
        + governance.fillna(0)
    )

    # Rule 1:
    # not_relevant -> E=S=G=0
    not_relevant_with_pillar = (
        relevance.eq(
            "not_relevant"
        )
        & pillar_sum.ne(0)
    )

    # Rule 2:
    # relevant -> at least one E/S/G=1
    relevant_without_pillar = (
        relevance.eq(
            "relevant"
        )
        & pillar_sum.lt(1)
    )

    logic_checks = [
        (
            "not_relevant but E/S/G not all zero",
            not_relevant_with_pillar,
        ),
        (
            "relevant but no E/S/G pillar selected",
            relevant_without_pillar,
        ),
    ]

    # Optional impact check
    if (
        "esg_impact"
        in frame.columns
    ):

        impact = normalize_text(
            frame[
                "esg_impact"
            ]
        )

        invalid_not_relevant_impact = (
            relevance.eq(
                "not_relevant"
            )
            & ~impact.eq(
                "not_applicable"
            )
            & ~missing_mask(
                frame[
                    "esg_impact"
                ]
            )
        )

        logic_checks.append(
            (
                "not_relevant but impact is not not_applicable",
                invalid_not_relevant_impact,
            )
        )

    # Optional information-type check
    if (
        "esg_information_type"
        in frame.columns
    ):

        information_type = (
            normalize_text(
                frame[
                    "esg_information_type"
                ]
            )
        )

        invalid_not_relevant_info = (
            relevance.eq(
                "not_relevant"
            )
            & ~information_type.eq(
                "not_applicable"
            )
            & ~missing_mask(
                frame[
                    "esg_information_type"
                ]
            )
        )

        logic_checks.append(
            (
                "not_relevant but information type is not not_applicable",
                invalid_not_relevant_info,
            )
        )

    for issue_name, issue_mask in (
        logic_checks
    ):

        n_issues = int(
            issue_mask.sum()
        )

        add_check(
            qa_records,
            "Cross-label logic",
            f"{name}: {issue_name}",
            n_issues,
            0,
            n_issues == 0,
        )

        if n_issues > 0:

            issue_columns = [
                column
                for column in [
                    KEY,
                    "source",
                    "source_batch",
                    "title",
                    "esg_relevance_raw",
                    "esg_relevance",
                    "sentiment",
                    "environmental",
                    "social",
                    "governance",
                    "esg_impact",
                    "esg_information_type",
                ]
                if column
                in frame.columns
            ]

            issue_rows = frame.loc[
                issue_mask,
                issue_columns,
            ].copy()

            issue_rows[
                "file"
            ] = name

            issue_rows[
                "logic_issue"
            ] = issue_name

            logic_issue_frames.append(
                issue_rows
            )


if logic_issue_frames:

    logic_issues = pd.concat(
        logic_issue_frames,
        ignore_index=True,
        sort=False,
    )

    issue_tables[
        "Cross-label logic issues"
    ] = logic_issues

else:

    logic_issues = pd.DataFrame()


# ------------------------------------------------------------
# 18. Metadata consistency for the 1,200 shared articles
# ------------------------------------------------------------
metadata_columns = [
    column
    for column in [
        "source",
        "source_batch",
        "title",
    ]
    if (
        column in shared_a.columns
        and column in shared_b.columns
    )
]


metadata_a = shared_a[
    [
        KEY,
        *metadata_columns,
    ]
].copy()

metadata_b = shared_b[
    [
        KEY,
        *metadata_columns,
    ]
].copy()


shared_metadata = (
    metadata_a.merge(
        metadata_b,
        on=KEY,
        how="outer",
        suffixes=(
            "_A",
            "_B",
        ),
        indicator=True,
        validate="one_to_one",
    )
)


metadata_mismatch = (
    ~shared_metadata[
        "_merge"
    ].eq("both")
)


for column in metadata_columns:

    column_a = (
        f"{column}_A"
    )

    column_b = (
        f"{column}_B"
    )

    metadata_mismatch = (
        metadata_mismatch
        | (
            shared_metadata[
                column_a
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
            .ne(
                shared_metadata[
                    column_b
                ]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
            )
        )
    )


n_metadata_mismatch = int(
    metadata_mismatch.sum()
)


add_check(
    qa_records,
    "Metadata consistency",
    "Shared A/B article metadata",
    n_metadata_mismatch,
    0,
    n_metadata_mismatch == 0,
)


if n_metadata_mismatch > 0:

    issue_tables[
        "Shared metadata mismatches"
    ] = shared_metadata.loc[
        metadata_mismatch
    ]


# ------------------------------------------------------------
# 19. Optional evidence and note information
# ------------------------------------------------------------
# These counts are informational only.
# They do not affect PASS/FAIL.

optional_text_records = []


for name, frame in (
    datasets.items()
):

    if (
        "evidence_text"
        in frame.columns
    ):
        missing_evidence = int(
            missing_mask(
                frame[
                    "evidence_text"
                ]
            ).sum()
        )
    else:
        missing_evidence = np.nan

    if (
        "annotator_note"
        in frame.columns
    ):
        missing_note = int(
            missing_mask(
                frame[
                    "annotator_note"
                ]
            ).sum()
        )
    else:
        missing_note = np.nan

    optional_text_records.append(
        {
            "file": name,
            "missing_evidence_text": (
                missing_evidence
            ),
            "missing_annotator_note": (
                missing_note
            ),
            "affects_QA_status": False,
        }
    )


optional_text_summary = pd.DataFrame(
    optional_text_records
)


# ------------------------------------------------------------
# 20. Label-distribution diagnostics
# ------------------------------------------------------------
distribution_records = []


for name, frame in (
    datasets.items()
):

    for task in TASKS:

        if task in VALID_BINARY_LABELS:

            values = normalize_binary(
                frame[
                    task
                ]
            )

        else:

            values = normalize_text(
                frame[
                    task
                ]
            )

        counts = (
            values
            .fillna(
                "__MISSING__"
            )
            .value_counts(
                dropna=False
            )
        )

        for label, count in (
            counts.items()
        ):

            distribution_records.append(
                {
                    "file": name,
                    "task": task,
                    "label": label,
                    "n_articles": int(
                        count
                    ),
                    "share": (
                        float(count)
                        / len(frame)
                    ),
                }
            )


label_distribution = pd.DataFrame(
    distribution_records
)


# ------------------------------------------------------------
# 21. Compile QA outputs
# ------------------------------------------------------------
qa_summary = pd.DataFrame(
    qa_records
)


qa_summary = (
    qa_summary
    .sort_values(
        [
            "status",
            "check_group",
            "check_name",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


status_summary = (
    qa_summary[
        "status"
    ]
    .value_counts()
    .rename_axis(
        "status"
    )
    .reset_index(
        name="n_checks"
    )
)


failed_checks = qa_summary.loc[
    qa_summary[
        "status"
    ].eq("FAIL")
].copy()


# ------------------------------------------------------------
# 22. Display results
# ------------------------------------------------------------
print(
    "===== OVERALL QA STATUS ====="
)

display(
    status_summary
)


print(
    "===== COMPLETE QA SUMMARY ====="
)

display(
    qa_summary
)


print(
    "===== LABEL COMPLETENESS ====="
)

display(
    label_missing_summary
)


print(
    "===== LABEL PROVENANCE ====="
)

display(
    provenance_summary
)


print(
    "===== LABEL DISTRIBUTIONS ====="
)

display(
    label_distribution
)


print(
    "===== OPTIONAL TEXT FIELDS ====="
)

print(
    "Evidence and notes do not affect "
    "the QA result."
)

display(
    optional_text_summary
)


# ------------------------------------------------------------
# 23. Display detailed issue tables
# ------------------------------------------------------------
if issue_tables:

    print(
        "===== DETAILED ISSUE TABLES ====="
    )

    for issue_name, issue_frame in (
        issue_tables.items()
    ):

        print()
        print(
            issue_name
        )

        print(
            "Number of rows:",
            len(issue_frame),
        )

        display(
            issue_frame.head(200)
        )


# ------------------------------------------------------------
# 24. Final result
# ------------------------------------------------------------
n_pass = int(
    qa_summary[
        "status"
    ].eq("PASS").sum()
)

n_fail = int(
    qa_summary[
        "status"
    ].eq("FAIL").sum()
)


print()
print(
    "Total QA checks:",
    len(
        qa_summary
    ),
)

print(
    "PASS:",
    n_pass,
)

print(
    "FAIL:",
    n_fail,
)


if failed_checks.empty:

    print()
    print(
        "FULL DATA QA: PASS"
    )

    print(
        "The three files are structurally "
        "ready for reliability analysis."
    )

    print(
        "ESG/non_ESG were treated as "
        "relevant/not_relevant."
    )

    print(
        "Evidence and annotator notes were "
        "excluded from PASS/FAIL."
    )

    print(
        "No agreement or Cohen's kappa was "
        "calculated in this cell."
    )

else:

    print()
    print(
        "FULL DATA QA: FAIL"
    )

    display(
        failed_checks
    )

    raise ValueError(
        f"{len(failed_checks)} QA checks failed. "
        "Review the displayed issue tables."
    )

# === Notebook cell 8 ===
# ============================================================
# EXPERT-TO-FINAL LABEL CONCORDANCE
#
# Compare only direct new annotations:
#     Expert A: 600 articles
#     Expert B: 600 articles
#
# Do not compare all 2,000 rows because 1,400 rows in each
# expert master already contain pre-existing final labels.
# ============================================================

import warnings

import numpy as np
import pandas as pd

from IPython.display import display

from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
KEY = "article_key"

TASKS = [
    "esg_relevance",
    "sentiment",
    "environmental",
    "social",
    "governance",
]

EXPECTED_DIRECT_PER_EXPERT = 600

EXPECTED_DIRECT_SOURCE_COUNTS = {
    "cafef": 400,
    "vietstock": 200,
}

BOOTSTRAP_ITERATIONS = 500
BOOTSTRAP_SEED = 42


TASK_LABEL_ORDER = {
    "esg_relevance": [
        "not_relevant",
        "relevant",
        "uncertain",
    ],
    "sentiment": [
        "negative",
        "mixed",
        "neutral",
        "positive",
    ],
    "environmental": [
        "0",
        "1",
    ],
    "social": [
        "0",
        "1",
    ],
    "governance": [
        "0",
        "1",
    ],
}


POSITIVE_LABELS = {
    "esg_relevance": "relevant",
    "environmental": "1",
    "social": "1",
    "governance": "1",
}


# ------------------------------------------------------------
# 2. Normalization helpers
# ------------------------------------------------------------
def normalize_text(
    series,
):
    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def missing_mask(
    series,
):
    normalized = normalize_text(
        series
    )

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


def normalize_boolean(
    series,
):
    normalized = normalize_text(
        series
    )

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


def normalize_task_labels(
    task,
    series,
):
    normalized = normalize_text(
        series
    )

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


def get_observed_labels(
    task,
    expert_values,
    final_values,
):
    observed = set(
        pd.concat(
            [
                expert_values,
                final_values,
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
    )

    preferred_order = (
        TASK_LABEL_ORDER.get(
            task,
            [],
        )
    )

    ordered = [
        label
        for label in preferred_order
        if label in observed
    ]

    remaining = sorted(
        observed
        - set(ordered)
    )

    return (
        ordered
        + remaining
    )


# ------------------------------------------------------------
# 3. Wilson confidence interval for agreement
# ------------------------------------------------------------
def wilson_interval(
    successes,
    total,
    z=1.959963984540054,
):
    if total == 0:
        return (
            np.nan,
            np.nan,
        )

    proportion = (
        successes
        / total
    )

    denominator = (
        1
        + (
            z ** 2
            / total
        )
    )

    centre = (
        proportion
        + (
            z ** 2
            / (
                2
                * total
            )
        )
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            (
                proportion
                * (
                    1
                    - proportion
                )
                / total
            )
            + (
                z ** 2
                / (
                    4
                    * total ** 2
                )
            )
        )
        / denominator
    )

    return (
        max(
            0,
            centre
            - half_width,
        ),
        min(
            1,
            centre
            + half_width,
        ),
    )


# ------------------------------------------------------------
# 4. Bootstrap confidence interval for Cohen's kappa
# ------------------------------------------------------------
def bootstrap_kappa_interval(
    expert_values,
    final_values,
    n_iterations=500,
    seed=42,
):
    expert_values = np.asarray(
        expert_values
    )

    final_values = np.asarray(
        final_values
    )

    n = len(
        expert_values
    )

    if n < 2:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_values = []

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        for _ in range(
            n_iterations
        ):

            sample_indices = rng.integers(
                0,
                n,
                size=n,
            )

            sampled_expert = (
                expert_values[
                    sample_indices
                ]
            )

            sampled_final = (
                final_values[
                    sample_indices
                ]
            )

            try:
                value = cohen_kappa_score(
                    sampled_final,
                    sampled_expert,
                )
            except Exception:
                value = np.nan

            if np.isfinite(
                value
            ):
                bootstrap_values.append(
                    float(value)
                )

    if not bootstrap_values:
        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            np.percentile(
                bootstrap_values,
                2.5,
            )
        ),
        float(
            np.percentile(
                bootstrap_values,
                97.5,
            )
        ),
    )


# ------------------------------------------------------------
# 5. Descriptive interpretation of kappa
# ------------------------------------------------------------
def interpret_kappa(
    value,
):
    if pd.isna(
        value
    ):
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


# ------------------------------------------------------------
# 6. Prepare the final-label lookup
# ------------------------------------------------------------
required_final_columns = [
    KEY,
    "source",
    *TASKS,
]

missing_final_columns = [
    column
    for column in required_final_columns
    if column not in F.columns
]

if missing_final_columns:
    raise ValueError(
        "Final thiếu cột: "
        f"{missing_final_columns}"
    )


final_lookup_columns = [
    column
    for column in [
        KEY,
        "source",
        "source_batch",
        "title",
        *TASKS,
    ]
    if column in F.columns
]


final_lookup = (
    F[
        final_lookup_columns
    ]
    .copy()
)


final_lookup[
    KEY
] = (
    final_lookup[
        KEY
    ]
    .astype("string")
    .str.strip()
)


if final_lookup[
    KEY
].duplicated().any():
    raise ValueError(
        "Final contains duplicate article_key."
    )


final_rename = {
    task: (
        f"final_{task}"
    )
    for task in TASKS
}

final_lookup = (
    final_lookup.rename(
        columns=final_rename
    )
)


# ------------------------------------------------------------
# 7. Extract each expert's 600 direct annotations
# ------------------------------------------------------------
expert_inputs = {
    "Expert A": A.copy(),
    "Expert B": B.copy(),
}

expert_direct_frames = {}


for expert_name, frame in (
    expert_inputs.items()
):

    required_expert_columns = [
        KEY,
        "source",
        "is_direct_new_annotation",
        *TASKS,
    ]

    missing_expert_columns = [
        column
        for column in required_expert_columns
        if column not in frame.columns
    ]

    if missing_expert_columns:
        raise ValueError(
            f"{expert_name} thiếu cột: "
            f"{missing_expert_columns}"
        )

    frame[
        KEY
    ] = (
        frame[
            KEY
        ]
        .astype("string")
        .str.strip()
    )

    frame[
        "is_direct_new_annotation"
    ] = normalize_boolean(
        frame[
            "is_direct_new_annotation"
        ]
    )

    direct = (
        frame.loc[
            frame[
                "is_direct_new_annotation"
            ]
        ]
        .copy()
    )

    if (
        len(direct)
# COLAB COMMAND:         != EXPECTED_DIRECT_PER_EXPERT
    ):
        raise ValueError(
            f"{expert_name} có "
            f"{len(direct)} nhãn trực tiếp; "
            f"cần đúng "
            f"{EXPECTED_DIRECT_PER_EXPERT}."
        )

    if direct[
        KEY
    ].duplicated().any():
        raise ValueError(
            f"{expert_name} có article_key "
            "trùng trong 600 nhãn trực tiếp."
        )

    source_counts = (
        normalize_text(
            direct[
                "source"
            ]
        )
        .value_counts()
        .to_dict()
    )

    if (
        source_counts
# COLAB COMMAND:         != EXPECTED_DIRECT_SOURCE_COUNTS
    ):
        raise ValueError(
            f"{expert_name} có phân bổ nguồn "
            f"{source_counts}; cần "
            f"{EXPECTED_DIRECT_SOURCE_COUNTS}."
        )

    expert_rename = {
        task: (
            f"expert_{task}"
        )
        for task in TASKS
    }

    direct = direct.rename(
        columns=expert_rename
    )

    expert_direct_frames[
        expert_name
    ] = direct


# ------------------------------------------------------------
# 8. Merge each expert's direct labels with Final
# ------------------------------------------------------------
comparison_frames = {}


for expert_name, direct in (
    expert_direct_frames.items()
):

    expert_columns = [
        column
        for column in [
            KEY,
            "source",
            "source_batch",
            "title",
            *[
                f"expert_{task}"
                for task in TASKS
            ],
        ]
        if column in direct.columns
    ]

    comparison = (
        direct[
            expert_columns
        ]
        .merge(
            final_lookup,
            on=KEY,
            how="left",
            suffixes=(
                "_expert_meta",
                "_final_meta",
            ),
            validate="one_to_one",
            indicator=True,
        )
    )

    unmatched = (
        ~comparison[
            "_merge"
        ].eq("both")
    )

    if unmatched.any():

        display(
            comparison.loc[
                unmatched
            ]
        )

        raise ValueError(
            f"{expert_name} có "
            f"{int(unmatched.sum())} bài "
            "không ghép được với Final."
        )

    comparison = comparison.drop(
        columns=[
            "_merge",
        ]
    )

    # Obtain one standard source column.
    if (
        "source_expert_meta"
        in comparison.columns
    ):
        comparison[
            "comparison_source"
        ] = normalize_text(
            comparison[
                "source_expert_meta"
            ]
        )

    elif "source" in comparison.columns:
        comparison[
            "comparison_source"
        ] = normalize_text(
            comparison[
                "source"
            ]
        )

    elif (
        "source_final_meta"
        in comparison.columns
    ):
        comparison[
            "comparison_source"
        ] = normalize_text(
            comparison[
                "source_final_meta"
            ]
        )

    else:
        raise KeyError(
            "Could not identify source after merge."
        )

    # Normalize task labels.
    for task in TASKS:

        expert_column = (
            f"expert_{task}"
        )

        final_column = (
            f"final_{task}"
        )

        comparison[
            expert_column
        ] = normalize_task_labels(
            task,
            comparison[
                expert_column
            ],
        )

        comparison[
            final_column
        ] = normalize_task_labels(
            task,
            comparison[
                final_column
            ],
        )

    comparison_frames[
        expert_name
    ] = comparison


# ------------------------------------------------------------
# 9. QA the comparison sample
# ------------------------------------------------------------
comparison_qa_records = []


for expert_name, comparison in (
    comparison_frames.items()
):

    comparison_qa_records.append(
        {
            "expert": expert_name,
            "n_direct_articles": len(
                comparison
            ),
            "cafef": int(
                comparison[
                    "comparison_source"
                ].eq(
                    "cafef"
                ).sum()
            ),
            "vietstock": int(
                comparison[
                    "comparison_source"
                ].eq(
                    "vietstock"
                ).sum()
            ),
            "unique_keys": (
                comparison[
                    KEY
                ].nunique()
            ),
        }
    )


comparison_sample_qa = pd.DataFrame(
    comparison_qa_records
)


print(
    "===== COMPARISON SAMPLE QA ====="
)

display(
    comparison_sample_qa
)


# ------------------------------------------------------------
# 10. Metric calculation function
# ------------------------------------------------------------
def calculate_task_metrics(
    expert_name,
    comparison,
    task,
    source_group,
):
    if source_group == "all":

        subset = comparison.copy()

    else:

        subset = comparison.loc[
            comparison[
                "comparison_source"
            ].eq(
                source_group
            )
        ].copy()

    expert_column = (
        f"expert_{task}"
    )

    final_column = (
        f"final_{task}"
    )

    valid_mask = (
        ~missing_mask(
            subset[
                expert_column
            ]
        )
        & ~missing_mask(
            subset[
                final_column
            ]
        )
    )

    valid = subset.loc[
        valid_mask
    ].copy()

    expert_values = (
        valid[
            expert_column
        ]
        .astype(str)
        .to_numpy()
    )

    final_values = (
        valid[
            final_column
        ]
        .astype(str)
        .to_numpy()
    )

    n_total = len(
        subset
    )

    n_compared = len(
        valid
    )

    n_missing_pairs = (
        n_total
        - n_compared
    )

    if n_compared == 0:

        return {
            "expert": expert_name,
            "source": source_group,
            "task": task,
            "n_total": n_total,
            "n_compared": 0,
            "n_missing_pairs": (
                n_missing_pairs
            ),
            "n_agree": 0,
            "n_disagree": 0,
            "agreement": np.nan,
            "agreement_ci_low": np.nan,
            "agreement_ci_high": np.nan,
            "cohen_kappa": np.nan,
            "kappa_ci_low": np.nan,
            "kappa_ci_high": np.nan,
            "kappa_interpretation": (
                "not_estimable"
            ),
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "balanced_accuracy": np.nan,
            "matthews_correlation": np.nan,
        }

    agreement_vector = (
        expert_values
        == final_values
    )

    n_agree = int(
        agreement_vector.sum()
    )

    n_disagree = int(
        (
            ~agreement_vector
        ).sum()
    )

    agreement = (
        n_agree
        / n_compared
    )

    agreement_ci_low, agreement_ci_high = (
        wilson_interval(
            n_agree,
            n_compared,
        )
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        try:
            kappa = (
                cohen_kappa_score(
                    final_values,
                    expert_values,
                )
            )
        except Exception:
            kappa = np.nan

        try:
            macro_f1 = f1_score(
                final_values,
                expert_values,
                average="macro",
                zero_division=0,
            )
        except Exception:
            macro_f1 = np.nan

        try:
            weighted_f1 = f1_score(
                final_values,
                expert_values,
                average="weighted",
                zero_division=0,
            )
        except Exception:
            weighted_f1 = np.nan

        try:
            balanced_accuracy = (
                balanced_accuracy_score(
                    final_values,
                    expert_values,
                )
            )
        except Exception:
            balanced_accuracy = np.nan

        try:
            matthews = (
                matthews_corrcoef(
                    final_values,
                    expert_values,
                )
            )
        except Exception:
            matthews = np.nan

    kappa_ci_low, kappa_ci_high = (
        bootstrap_kappa_interval(
            expert_values,
            final_values,
            n_iterations=(
                BOOTSTRAP_ITERATIONS
            ),
            seed=(
                BOOTSTRAP_SEED
            ),
        )
    )

    return {
        "expert": expert_name,
        "source": source_group,
        "task": task,
        "n_total": n_total,
        "n_compared": n_compared,
        "n_missing_pairs": (
            n_missing_pairs
        ),
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement": agreement,
        "agreement_ci_low": (
            agreement_ci_low
        ),
        "agreement_ci_high": (
            agreement_ci_high
        ),
        "cohen_kappa": kappa,
        "kappa_ci_low": (
            kappa_ci_low
        ),
        "kappa_ci_high": (
            kappa_ci_high
        ),
        "kappa_interpretation": (
            interpret_kappa(
                kappa
            )
        ),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "matthews_correlation": (
            matthews
        ),
    }


# ------------------------------------------------------------
# 11. Calculate overall and source-specific metrics
# ------------------------------------------------------------
metric_records = []


for expert_name, comparison in (
    comparison_frames.items()
):

    for source_group in [
        "all",
        "cafef",
        "vietstock",
    ]:

        for task in TASKS:

            metric_records.append(
                calculate_task_metrics(
                    expert_name=(
                        expert_name
                    ),
                    comparison=(
                        comparison
                    ),
                    task=task,
                    source_group=(
                        source_group
                    ),
                )
            )


expert_final_metrics = (
    pd.DataFrame(
        metric_records
    )
    .sort_values(
        [
            "task",
            "source",
            "expert",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "===== EXPERT-TO-FINAL METRICS ====="
)

display(
    expert_final_metrics
)


# ------------------------------------------------------------
# 12. Compact overall comparison
# ------------------------------------------------------------
overall_metrics = (
    expert_final_metrics.loc[
        expert_final_metrics[
            "source"
        ].eq("all")
    ]
    .copy()
)


overall_comparison = (
    overall_metrics[
        [
            "expert",
            "task",
            "n_compared",
            "n_disagree",
            "agreement",
            "agreement_ci_low",
            "agreement_ci_high",
            "cohen_kappa",
            "kappa_ci_low",
            "kappa_ci_high",
            "kappa_interpretation",
            "macro_f1",
            "balanced_accuracy",
            "matthews_correlation",
        ]
    ]
    .sort_values(
        [
            "task",
            "expert",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "===== OVERALL EXPERT COMPARISON ====="
)

display(
    overall_comparison
)


# ------------------------------------------------------------
# 13. Difference between Expert A and Expert B metrics
# ------------------------------------------------------------
metric_difference = (
    overall_metrics.pivot(
        index="task",
        columns="expert",
        values=[
            "agreement",
            "cohen_kappa",
            "macro_f1",
            "balanced_accuracy",
            "matthews_correlation",
        ],
    )
)


metric_difference.columns = [
    (
        metric_name
        + "_"
        + expert_name.replace(
            " ",
            "_",
        )
    )
    for metric_name, expert_name
    in metric_difference.columns
]


metric_difference = (
    metric_difference.reset_index()
)


for metric_name in [
    "agreement",
    "cohen_kappa",
    "macro_f1",
    "balanced_accuracy",
    "matthews_correlation",
]:

    column_a = (
        f"{metric_name}_Expert_A"
    )

    column_b = (
        f"{metric_name}_Expert_B"
    )

    if (
        column_a
        in metric_difference.columns
        and column_b
        in metric_difference.columns
    ):

        metric_difference[
            f"{metric_name}_A_minus_B"
        ] = (
            metric_difference[
                column_a
            ]
            - metric_difference[
                column_b
            ]
        )


print(
    "===== EXPERT A MINUS EXPERT B ====="
)

display(
    metric_difference
)


# ------------------------------------------------------------
# 14. Binary-task diagnostics
# ------------------------------------------------------------
binary_metric_records = []


for expert_name, comparison in (
    comparison_frames.items()
):

    for source_group in [
        "all",
        "cafef",
        "vietstock",
    ]:

        if source_group == "all":
            subset = comparison.copy()
        else:
            subset = comparison.loc[
                comparison[
                    "comparison_source"
                ].eq(
                    source_group
                )
            ].copy()

        for task, positive_label in (
            POSITIVE_LABELS.items()
        ):

            expert_column = (
                f"expert_{task}"
            )

            final_column = (
                f"final_{task}"
            )

            valid_mask = (
                ~missing_mask(
                    subset[
                        expert_column
                    ]
                )
                & ~missing_mask(
                    subset[
                        final_column
                    ]
                )
            )

            valid = subset.loc[
                valid_mask
            ].copy()

            expert_values = (
                valid[
                    expert_column
                ]
                .astype(str)
                .to_numpy()
            )

            final_values = (
                valid[
                    final_column
                ]
                .astype(str)
                .to_numpy()
            )

            if len(valid) == 0:
                continue

            expert_positive = (
                expert_values
                == positive_label
            )

            final_positive = (
                final_values
                == positive_label
            )

            tp = int(
                (
                    expert_positive
                    & final_positive
                ).sum()
            )

            tn = int(
                (
                    ~expert_positive
                    & ~final_positive
                ).sum()
            )

            fp = int(
                (
                    expert_positive
                    & ~final_positive
                ).sum()
            )

            fn = int(
                (
                    ~expert_positive
                    & final_positive
                ).sum()
            )

            precision = (
                tp
                / (
                    tp
                    + fp
                )
                if (
                    tp
                    + fp
                )
                > 0
                else np.nan
            )

            recall = (
                tp
                / (
                    tp
                    + fn
                )
                if (
                    tp
                    + fn
                )
                > 0
                else np.nan
            )

            specificity = (
                tn
                / (
                    tn
                    + fp
                )
                if (
                    tn
                    + fp
                )
                > 0
                else np.nan
            )

            f1 = (
                (
                    2
                    * precision
                    * recall
                    / (
                        precision
                        + recall
                    )
                )
                if (
                    pd.notna(
                        precision
                    )
                    and pd.notna(
                        recall
                    )
                    and (
                        precision
                        + recall
                    )
                    > 0
                )
                else np.nan
            )

            # McNemar discordant cells:
            # fp = expert positive, final negative
            # fn = expert negative, final positive
            n_discordant = (
                fp
                + fn
            )

            if n_discordant > 0:

                mcnemar_p = (
                    binomtest(
                        k=min(
                            fp,
                            fn,
                        ),
                        n=n_discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )

            else:

                mcnemar_p = 1.0

            expert_positive_rate = (
                float(
                    expert_positive.mean()
                )
            )

            final_positive_rate = (
                float(
                    final_positive.mean()
                )
            )

            binary_metric_records.append(
                {
                    "expert": expert_name,
                    "source": source_group,
                    "task": task,
                    "positive_label": (
                        positive_label
                    ),
                    "n_compared": len(
                        valid
                    ),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "precision_vs_final": (
                        precision
                    ),
                    "recall_vs_final": recall,
                    "specificity_vs_final": (
                        specificity
                    ),
                    "f1_vs_final": f1,
                    "expert_positive_rate": (
                        expert_positive_rate
                    ),
                    "final_positive_rate": (
                        final_positive_rate
                    ),
                    "positive_rate_difference": (
                        expert_positive_rate
                        - final_positive_rate
                    ),
                    "mcnemar_discordant_pairs": (
                        n_discordant
                    ),
                    "mcnemar_exact_p": (
                        mcnemar_p
                    ),
                }
            )


binary_task_metrics = (
    pd.DataFrame(
        binary_metric_records
    )
    .sort_values(
        [
            "task",
            "source",
            "expert",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "===== BINARY TASK DIAGNOSTICS ====="
)

display(
    binary_task_metrics
)


# ------------------------------------------------------------
# 15. Per-class precision, recall, and F1
# ------------------------------------------------------------
per_class_records = []


for expert_name, comparison in (
    comparison_frames.items()
):

    for source_group in [
        "all",
        "cafef",
        "vietstock",
    ]:

        if source_group == "all":
            subset = comparison.copy()
        else:
            subset = comparison.loc[
                comparison[
                    "comparison_source"
                ].eq(
                    source_group
                )
            ].copy()

        for task in TASKS:

            expert_column = (
                f"expert_{task}"
            )

            final_column = (
                f"final_{task}"
            )

            valid_mask = (
                ~missing_mask(
                    subset[
                        expert_column
                    ]
                )
                & ~missing_mask(
                    subset[
                        final_column
                    ]
                )
            )

            valid = subset.loc[
                valid_mask
            ]

            if valid.empty:
                continue

            expert_values = (
                valid[
                    expert_column
                ].astype(str)
            )

            final_values = (
                valid[
                    final_column
                ].astype(str)
            )

            labels = (
                get_observed_labels(
                    task,
                    expert_values,
                    final_values,
                )
            )

            (
                precision_values,
                recall_values,
                f1_values,
                support_values,
            ) = precision_recall_fscore_support(
                final_values,
                expert_values,
                labels=labels,
                zero_division=0,
            )

            expert_counts = (
                expert_values
                .value_counts()
                .to_dict()
            )

            final_counts = (
                final_values
                .value_counts()
                .to_dict()
            )

            for (
                label,
                precision_value,
                recall_value,
                f1_value,
                support_value,
            ) in zip(
                labels,
                precision_values,
                recall_values,
                f1_values,
                support_values,
            ):

                per_class_records.append(
                    {
                        "expert": expert_name,
                        "source": source_group,
                        "task": task,
                        "label": label,
                        "final_support": int(
                            support_value
                        ),
                        "expert_count": int(
                            expert_counts.get(
                                label,
                                0,
                            )
                        ),
                        "final_count": int(
                            final_counts.get(
                                label,
                                0,
                            )
                        ),
                        "precision_vs_final": (
                            float(
                                precision_value
                            )
                        ),
                        "recall_vs_final": (
                            float(
                                recall_value
                            )
                        ),
                        "f1_vs_final": float(
                            f1_value
                        ),
                    }
                )


per_class_metrics = (
    pd.DataFrame(
        per_class_records
    )
    .sort_values(
        [
            "task",
            "source",
            "label",
            "expert",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "===== PER-CLASS METRICS ====="
)

display(
    per_class_metrics
)


# ------------------------------------------------------------
# 16. Confusion matrices
# ------------------------------------------------------------
confusion_matrices = {}


for expert_name, comparison in (
    comparison_frames.items()
):

    for task in TASKS:

        expert_column = (
            f"expert_{task}"
        )

        final_column = (
            f"final_{task}"
        )

        valid_mask = (
            ~missing_mask(
                comparison[
                    expert_column
                ]
            )
            & ~missing_mask(
                comparison[
                    final_column
                ]
            )
        )

        valid = comparison.loc[
            valid_mask
        ]

        expert_values = (
            valid[
                expert_column
            ].astype(str)
        )

        final_values = (
            valid[
                final_column
            ].astype(str)
        )

        labels = get_observed_labels(
            task,
            expert_values,
            final_values,
        )

        matrix = confusion_matrix(
            final_values,
            expert_values,
            labels=labels,
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                f"Final={label}"
                for label in labels
            ],
            columns=[
                f"Expert={label}"
                for label in labels
            ],
        )

        confusion_matrices[
            (
                expert_name,
                task,
            )
        ] = matrix_df


for (
    expert_name,
    task,
), matrix_df in (
    confusion_matrices.items()
):

    print()
    print(
        f"{expert_name} vs Final — {task}"
    )

    display(
        matrix_df
    )


# ------------------------------------------------------------
# 17. Detailed disagreement records
# ------------------------------------------------------------
disagreement_records = []


for expert_name, comparison in (
    comparison_frames.items()
):

    for task in TASKS:

        expert_column = (
            f"expert_{task}"
        )

        final_column = (
            f"final_{task}"
        )

        valid_mask = (
            ~missing_mask(
                comparison[
                    expert_column
                ]
            )
            & ~missing_mask(
                comparison[
                    final_column
                ]
            )
        )

        disagreement_mask = (
            valid_mask
            & comparison[
                expert_column
            ].ne(
                comparison[
                    final_column
                ]
            )
        )

        disagreement_subset = (
            comparison.loc[
                disagreement_mask
            ]
        )

        for _, row in (
            disagreement_subset.iterrows()
        ):

            disagreement_records.append(
                {
                    "expert": expert_name,
                    "article_key": (
                        row[
                            KEY
                        ]
                    ),
                    "source": (
                        row[
                            "comparison_source"
                        ]
                    ),
                    "source_batch": (
                        row.get(
                            "source_batch_expert_meta",
                            row.get(
                                "source_batch",
                                np.nan,
                            ),
                        )
                    ),
                    "title": (
                        row.get(
                            "title_expert_meta",
                            row.get(
                                "title",
                                np.nan,
                            ),
                        )
                    ),
                    "task": task,
                    "expert_label": (
                        row[
                            expert_column
                        ]
                    ),
                    "final_label": (
                        row[
                            final_column
                        ]
                    ),
                    "transition": (
                        str(
                            row[
                                "final_"
                                + task
                            ]
                        )
                        + " → "
                        + str(
                            row[
                                "expert_"
                                + task
                            ]
                        )
                    ),
                }
            )


expert_final_disagreements = (
    pd.DataFrame(
        disagreement_records
    )
)


if not expert_final_disagreements.empty:

    expert_final_disagreements = (
        expert_final_disagreements
        .sort_values(
            [
                "task",
                "expert",
                "source",
                "source_batch",
                "article_key",
            ]
        )
        .reset_index(
            drop=True
        )
    )


print(
    "===== DISAGREEMENT COUNTS ====="
)


if expert_final_disagreements.empty:

    disagreement_summary = pd.DataFrame(
        columns=[
            "expert",
            "source",
            "task",
            "n_disagreements",
        ]
    )

else:

    disagreement_summary = (
        expert_final_disagreements
        .groupby(
            [
                "expert",
                "source",
                "task",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "n_disagreements"
        )
        .reset_index()
        .sort_values(
            [
                "task",
                "source",
                "expert",
            ]
        )
        .reset_index(
            drop=True
        )
    )


display(
    disagreement_summary
)


print(
    "===== DETAILED DISAGREEMENTS ====="
)

display(
    expert_final_disagreements.head(
        200
    )
)


# ------------------------------------------------------------
# 18. Transition summary
# ------------------------------------------------------------
if expert_final_disagreements.empty:

    disagreement_transitions = (
        pd.DataFrame(
            columns=[
                "expert",
                "task",
                "transition",
                "n_articles",
            ]
        )
    )

else:

    disagreement_transitions = (
        expert_final_disagreements
        .groupby(
            [
                "expert",
                "task",
                "transition",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "n_articles"
        )
        .reset_index()
        .sort_values(
            [
                "task",
                "expert",
                "n_articles",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


print(
    "===== DISAGREEMENT TRANSITIONS ====="
)

display(
    disagreement_transitions
)


# ------------------------------------------------------------
# 19. Final summary
# ------------------------------------------------------------
print()
print(
    "EXPERT-TO-FINAL COMPARISON: COMPLETE"
)

print(
    "Expert A direct comparisons:",
    len(
        comparison_frames[
            "Expert A"
        ]
    ),
)

print(
    "Expert B direct comparisons:",
    len(
        comparison_frames[
            "Expert B"
        ]
    ),
)

print(
    "Only direct new annotations were compared."
)

print(
    "The final labels were treated as a "
    "reference, not as an adjudicated gold standard."
)

# === Notebook cell 10 ===
