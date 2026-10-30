# Original notebook code exported for archival review. Paths/environment require adaptation.


# === Cell 2 ===
# Cell 1 — Cài thư viện
# COLAB COMMAND: !pip -q install -U "transformers>=5.0,<6" "datasets>=4.0,<5" "accelerate>=1.6,<2" \
    "scikit-learn>=1.5,<2" "pandas>=2.0,<3" openpyxl seaborn pyarrow


# === Cell 3 ===
# Cell 2 — Import và kiểm tra GPU
import os, re, json, math, random, hashlib, shutil, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Chưa có GPU. XLM-RoBERTa-base sẽ rất chậm; hãy bật GPU trong Runtime settings.")


# === Cell 5 ===
# Cell 3 — Nạp annotation từ cả Vietstock và CafeF
DATA_SOURCE = "drive"  # "drive" hoặc "upload"

VIETSTOCK_DATA_GLOBS = [
    "/content/gdrive/MyDrive/Project_Fin_Data_Crawling/Crawling_news/ESG_annotation_sampling/03_annotation_batches/Done/annotation_batch_*.xlsx",
]

# SỬA các đường dẫn dưới đây theo đúng nơi lưu annotation CafeF của bạn.
# Có thể khai báo nhiều pattern nếu CafeF được chia thành nhiều batch/thư mục.
CAFEF_DATA_GLOBS = [
    "/content/gdrive/MyDrive/Project_Fin_Data_Crawling/Crawling_news/ESG_annotation_sampling/ESG_annotation_CafeF_800/02_annotation_batches_41_56/Done/annotation_batch_*.xlsx",
]


def identify_source_hint(path):
    value = str(path).lower()
    if "cafef" in value or "cafe_f" in value or "cafe-f" in value:
        return "cafef"
    if "vietstock" in value:
        return "vietstock"
    return None


DATA_FILE_SPECS = []

if DATA_SOURCE == "upload":
    from google.colab import files

    uploaded = files.upload()
    for name in sorted(uploaded):
        path = Path("/content") / name
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            DATA_FILE_SPECS.append({"path": path, "source_hint": identify_source_hint(path)})
elif DATA_SOURCE == "drive":
    from google.colab import drive

    drive.mount("/content/gdrive")
    seen_paths = set()
    for source_name, patterns in [
        ("vietstock", VIETSTOCK_DATA_GLOBS),
        ("cafef", CAFEF_DATA_GLOBS),
    ]:
        matches = []
        for pattern in patterns:
            matches.extend(sorted(Path("/").glob(pattern.lstrip("/"))))

        unique_matches = []
        for path in matches:
            resolved = str(path.resolve())
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                unique_matches.append(path)
                DATA_FILE_SPECS.append({"path": path, "source_hint": source_name})

        print(f"{source_name}: {len(unique_matches)} file")
        for path in unique_matches:
            print(" -", path)
else:
    raise ValueError("DATA_SOURCE phải là 'drive' hoặc 'upload'.")

assert DATA_FILE_SPECS, "Không tìm thấy file annotation. Hãy kiểm tra các pattern Google Drive."
if DATA_SOURCE == "drive":
    configured_sources = {item["source_hint"] for item in DATA_FILE_SPECS}
    missing_sources = {"vietstock", "cafef"} - configured_sources
    if missing_sources:
        raise FileNotFoundError(
            "Thiếu annotation của nguồn: " + ", ".join(sorted(missing_sources))
            + ". Hãy sửa VIETSTOCK_DATA_GLOBS / CAFEF_DATA_GLOBS đúng thư mục thực tế."
        )

# Giữ biến này để tương thích với những cell cũ hoặc cell người dùng bổ sung.
DATA_FILES = [item["path"] for item in DATA_FILE_SPECS]
print(f"Tổng cộng {len(DATA_FILES)} file annotation từ hai nguồn.")


# === Cell 6 ===
# Cell 4 — Cấu hình mô hình four-view:
# title + sapo / head / continuation / tail

from pathlib import Path


# ============================================================
# 1. Thư mục dữ liệu và kết quả
# ============================================================

OUTPUT_ROOT = Path(
    "/content/gdrive/MyDrive/"
    "Project_Fin_Data_Crawling/"
    "Crawling_news/"
    "ESG_annotation_sampling"
)


# ============================================================
# 2. Thiết kế dữ liệu
# ============================================================

EXPERIMENT_MODE = (
    "cross_source_target_validation"
)

SOURCE_TRAIN = "vietstock"

SOURCE_TARGET = "cafef"


# None = dùng toàn bộ Vietstock hợp lệ.

TRAIN_SAMPLE_SIZE = None


TARGET_VALIDATION_SIZE = 400

TARGET_TEST_SIZE = 400


# Nếu chuyển sang cross_source_strict.

STRICT_SOURCE_VALIDATION_SIZE = 300


# Cho phép 798 CafeF được chia thành 399 / 399.

REQUIRE_EXACT_TARGET_COUNTS = False

MAX_TARGET_SHORTFALL_RATIO = 0.05


REMOVE_CROSS_SOURCE_GROUP_OVERLAP = True

GROUP_SPLIT_SEARCH_TRIALS = 512


# ============================================================
# 3. Thư mục kết quả và đối chứng PhoBERT
# ============================================================

# Phiên bản three-view gốc:
# title + sapo / head / tail.

THREE_VIEW_TAIL_RESULTS_DIR = (
    OUTPUT_ROOT
    / (
        f"esg_nlp_results_"
        f"{EXPERIMENT_MODE}_"
        f"{SOURCE_TRAIN}_to_"
        f"{SOURCE_TARGET}"
    )
)


# Phiên bản three-view continuation,
# nếu bạn đã chạy trước đó.

THREE_VIEW_CONTINUATION_RESULTS_DIR = (
    OUTPUT_ROOT
    / (
        f"esg_nlp_results_"
        f"{EXPERIMENT_MODE}_"
        f"{SOURCE_TRAIN}_to_"
        f"{SOURCE_TARGET}_"
        f"three_view_continuation"
    )
)


# Kết quả PhoBERT four-view dùng làm đối chứng công bằng.

PHOBERT_RESULTS_DIR = (
    OUTPUT_ROOT
    / (
        f"esg_nlp_results_"
        f"{EXPERIMENT_MODE}_"
        f"{SOURCE_TRAIN}_to_"
        f"{SOURCE_TARGET}_"
        f"four_view"
    )
)


# Kết quả XLM-RoBERTa four-view (tách biệt để không ghi đè PhoBERT).

OUTPUT_DIR = (
    OUTPUT_ROOT
    / (
        f"esg_nlp_results_"
        f"{EXPERIMENT_MODE}_"
        f"{SOURCE_TRAIN}_to_"
        f"{SOURCE_TARGET}_"
        f"xlm_roberta_four_view"
    )
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VIEW_ARCHITECTURE = "four_view"


VIEW_NAMES = [
    "title_sapo",
    "head",
    "continuation",
    "tail",
]

# Tương thích tạm thời với code cũ còn sót lại.
# Có thể xóa sau khi đã thay toàn bộ THIRD_VIEW_MODE trong Cell 16.
THIRD_VIEW_MODE = VIEW_ARCHITECTURE
# ============================================================
# 4. Cấu hình xử lý dữ liệu
# ============================================================

TEXT_COLUMNS_PRIORITY = [
    "title",
    "description",
    "summary",
    "sapo",
    "content",
]


COMPLETED_VALUES = {
    "completed",
    "complete",
    "done",
    "adjudicated",
    "1",
    "true",
}


DROP_MARKED_DUPLICATES = True

MAX_BASELINE_CHARS = 12000


MIN_TOTAL_FOR_TASK = 80

MIN_CLASS_FOR_TASK = 10

MIN_VALIDATION_CLASSES = 2

MIN_TEST_CLASSES = 2


# ============================================================
# 5. Baseline
# ============================================================

BASELINE_LOGREG_C_GRID = [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]


BASELINE_SVM_C_GRID = [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]


# ============================================================
# 6. Cấu hình XLM-RoBERTa
# ============================================================

RESEARCH_TASKS = [
    "esg_relevance",
    "sentiment_all",
    "sentiment_esg",
    "environmental",
    "social",
    "governance",
]

TRANSFORMER_TASKS = list(RESEARCH_TASKS)

CONTINUE_ON_TASK_ERROR = True

SKIP_COMPLETED_TASKS = True


MODEL_NAME = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 256

USE_VNCORENLP = False

NUM_EPOCHS = 6


# Four-view cần nhiều bộ nhớ hơn.
# Effective batch size vẫn là:
#
# 2 × 8 = 16. XLM-RoBERTa-base lớn hơn PhoBERT nên giảm micro-batch
# để hạn chế lỗi hết bộ nhớ mà vẫn giữ effective batch size.

TRAIN_BATCH_SIZE = 2

GRADIENT_ACCUMULATION = 8

EVAL_BATCH_SIZE = 2


LEARNING_RATE = 2e-5

WARMUP_RATIO = 0.08

EARLY_STOPPING_PATIENCE = 2

BOOTSTRAP_ITERATIONS = 2000


# ============================================================
# 7. Trọng số mặc định của bốn view
# ============================================================

DEFAULT_VIEW_WEIGHTS = {
    "title_sapo": 0.20,
    "head": 0.35,
    "continuation": 0.25,
    "tail": 0.20,
}


# ============================================================
# 8. Các cấu hình thử trên validation
# ============================================================

VIEW_WEIGHT_GRID = [

    {
        "title_sapo": 0.20,
        "head": 0.35,
        "continuation": 0.25,
        "tail": 0.20,
    },

    {
        "title_sapo": 0.20,
        "head": 0.40,
        "continuation": 0.30,
        "tail": 0.10,
    },

    {
        "title_sapo": 0.30,
        "head": 0.35,
        "continuation": 0.25,
        "tail": 0.10,
    },

    {
        "title_sapo": 0.30,
        "head": 0.40,
        "continuation": 0.20,
        "tail": 0.10,
    },

    {
        "title_sapo": 0.20,
        "head": 0.30,
        "continuation": 0.30,
        "tail": 0.20,
    },

    {
        "title_sapo": 0.40,
        "head": 0.30,
        "continuation": 0.20,
        "tail": 0.10,
    },

    # Không dùng tail.

    {
        "title_sapo": 0.20,
        "head": 0.40,
        "continuation": 0.40,
        "tail": 0.00,
    },

    {
        "title_sapo": 0.30,
        "head": 0.40,
        "continuation": 0.30,
        "tail": 0.00,
    },

    # Không dùng continuation.

    {
        "title_sapo": 0.20,
        "head": 0.50,
        "continuation": 0.00,
        "tail": 0.30,
    },

    {
        "title_sapo": 0.30,
        "head": 0.40,
        "continuation": 0.00,
        "tail": 0.30,
    },

    # Chỉ title + sapo và head.

    {
        "title_sapo": 0.40,
        "head": 0.60,
        "continuation": 0.00,
        "tail": 0.00,
    },

    {
        "title_sapo": 0.50,
        "head": 0.50,
        "continuation": 0.00,
        "tail": 0.00,
    },
]


# ============================================================
# 9. Kiểm tra cấu hình
# ============================================================

assert EXPERIMENT_MODE in {
    "cross_source_target_validation",
    "cross_source_strict",
    "within_source_random",
}


assert set(
    DEFAULT_VIEW_WEIGHTS.keys()
) == set(
    VIEW_NAMES
)


assert abs(
    sum(
        DEFAULT_VIEW_WEIGHTS.values()
    )
    - 1.0
) < 1e-9


for index, weights in enumerate(
    VIEW_WEIGHT_GRID,
    start=1,
):

    assert set(
        weights.keys()
    ) == set(
        VIEW_NAMES
    ), (
        f"Grid {index} không có đủ bốn view."
    )

    assert abs(
        sum(
            weights.values()
        )
        - 1.0
    ) < 1e-9, (
        f"Grid {index} không có tổng bằng 1."
    )

    # Bài ngắn có thể không có continuation/tail,
    # nên title + head phải còn ít nhất một trọng số.

    assert (
        weights["title_sapo"]
        + weights["head"]
    ) > 0, (
        f"Grid {index} không hợp lệ với bài ngắn."
    )


print(
    "Architecture:",
    VIEW_ARCHITECTURE,
)

print(
    "Views:",
    VIEW_NAMES,
)

print(
    "Train source:",
    SOURCE_TRAIN,
)

print(
    "Target source:",
    SOURCE_TARGET,
)

print(
    "Batch size:",
    TRAIN_BATCH_SIZE,
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION,
)

print(
    "Effective batch size:",
    (
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION
    ),
)

print(
    "Three-view tail results:",
    THREE_VIEW_TAIL_RESULTS_DIR,
)

print(
    "Three-view continuation results:",
    THREE_VIEW_CONTINUATION_RESULTS_DIR,
)

print(
    "PhoBERT reference:",
    PHOBERT_RESULTS_DIR,
)

print(
    "XLM-RoBERTa four-view output:",
    OUTPUT_DIR,
)


# === Cell 7 ===
# Cell 5 — Đọc batch, xác định nguồn và thống nhất schema annotation
warnings.filterwarnings(
    "ignore",
    message="Data Validation extension is not supported and will be removed",
    category=UserWarning,
    module=r"openpyxl\.worksheet\._reader",
)


def snake_case(name):
    name = str(name).strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"[^0-9A-Za-zÀ-ỹ]+", "_", name)
    return name.strip("_").lower()


def canonical_source(value):
    if pd.isna(value):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    if "cafef" in normalized:
        return "cafef"
    if "vietstock" in normalized:
        return "vietstock"
    return None


def infer_row_source(row, source_hint, path):
    detected = []
    for column in ["source", "news_source", "publisher", "website", "url", "link", "article_url"]:
        if column in row.index:
            value = canonical_source(row[column])
            if value is not None:
                detected.append(value)

    if len(set(detected)) > 1:
        raise ValueError(f"Một dòng có dấu hiệu nguồn mâu thuẫn trong {path.name}: {detected}")

    row_source = detected[0] if detected else None
    if source_hint is not None and row_source is not None and row_source != source_hint:
        raise ValueError(
            f"Nguồn trong file ({row_source}) mâu thuẫn với thư mục ({source_hint}): {path}"
        )

    return row_source or source_hint or identify_source_hint(path)


def read_annotation_file(path, source_hint=None):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        sheet_used = "csv"
    else:
        workbook = pd.ExcelFile(path)
        sheet_lookup = {str(name).strip().lower(): name for name in workbook.sheet_names}
        sheet_used = sheet_lookup.get("annotation", workbook.sheet_names[0])
        frame = pd.read_excel(path, sheet_name=sheet_used)

    frame.columns = [snake_case(column) for column in frame.columns]
    frame["source"] = frame.apply(infer_row_source, axis=1, args=(source_hint, path))

    unidentified = frame["source"].isna()
    if unidentified.any():
        raise ValueError(
            f"Không xác định được nguồn cho {int(unidentified.sum())} dòng trong {path.name}. "
            "Hãy thêm cột source = cafef/vietstock hoặc đặt file trong thư mục nguồn riêng."
        )

    frame["_source_file"] = path.name
    frame["_source_sheet"] = str(sheet_used)
    frame["_source_path"] = str(path)
    return frame


frames = []
load_log = []
for item in DATA_FILE_SPECS:
    path = item["path"]
    try:
        part = read_annotation_file(path, item.get("source_hint"))
        frames.append(part)
        load_log.append({
            "file": path.name,
            "source_hint": item.get("source_hint"),
            "rows": len(part),
            "sources_found": ",".join(sorted(part["source"].unique())),
            "status": "ok",
        })
    except Exception as error:
        load_log.append({
            "file": path.name, "source_hint": item.get("source_hint"),
            "rows": 0, "sources_found": "", "status": f"ERROR: {error}",
        })

load_log_frame = pd.DataFrame(load_log)
display(load_log_frame)
load_log_frame.to_csv(OUTPUT_DIR / "annotation_file_load_log.csv", index=False)

failed_files = load_log_frame.loc[load_log_frame["status"].str.startswith("ERROR:")]
if not failed_files.empty:
    raise RuntimeError(
        "Có file annotation đọc lỗi hoặc nhận diện nguồn sai. Sửa các dòng ERROR trước khi train."
    )

assert frames, "Không đọc được file annotation nào."
raw = pd.concat(frames, ignore_index=True, sort=False)
source_load_summary = raw.groupby("source", dropna=False).size().rename("raw_articles").reset_index()
display(source_load_summary)
source_load_summary.to_csv(OUTPUT_DIR / "source_raw_article_counts.csv", index=False)
print("Tổng số dòng ban đầu:", len(raw))
print("Số cột:", len(raw.columns))


# === Cell 8 ===
# Cell 6 — Làm sạch annotation, loại trùng đúng phạm vi nguồn và tạo nhóm chống leakage
def clean_scalar(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else np.nan


def norm_label(value):
    value = clean_scalar(value)
    if pd.isna(value):
        return np.nan
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


df = raw.copy()
df["source"] = df["source"].map(canonical_source)
if df["source"].isna().any():
    raise ValueError("Còn dòng chưa xác định được source sau bước nạp dữ liệu.")

if "annotation_status" in df.columns:
    status_norm = df["annotation_status"].map(norm_label)
    keep = status_norm.isin(COMPLETED_VALUES)
    print(f"Completed/adjudicated: {keep.sum():,}/{len(df):,}")
    df = df.loc[keep].copy()

if DROP_MARKED_DUPLICATES and "is_duplicate" in df.columns:
    duplicate_mark = df["is_duplicate"].map(norm_label).isin({"1", "true", "yes", "duplicate"})
    print("Dòng Is_Duplicate=1 bị loại:", int(duplicate_mark.sum()))
    df = df.loc[~duplicate_mark].copy()

available_text_cols = [column for column in TEXT_COLUMNS_PRIORITY if column in df.columns]
assert available_text_cols, "Không có title/description/content để huấn luyện."
for column in available_text_cols:
    df[column] = df[column].map(clean_scalar).fillna("")


def compose_text(row):
    pieces = []
    for column in available_text_cols:
        value = row[column].strip()
        if value and value not in pieces:
            pieces.append(value)
    return " [SEP] ".join(pieces)


df["model_text"] = df.apply(compose_text, axis=1).str.slice(0, MAX_BASELINE_CHARS)
df = df.loc[df["model_text"].str.len() >= 20].copy()

id_col = "article_id" if "article_id" in df.columns else (
    "news_id" if "news_id" in df.columns else None
)
label_candidates = [
    "esg_relevance", "sentiment", "content_polarity", "environmental", "social", "governance",
    "esg_impact", "esg_information_type", "article_type", "event_status", "severity",
    "materiality", "information_novelty", "company_role", "corporate_response",
]

conflict_rows = []
if id_col:
    valid_id_mask = df[id_col].map(clean_scalar).notna()
    id_parts = df.loc[valid_id_mask].copy()
    duplicate_mask = id_parts.duplicated(subset=["source", id_col], keep=False)

    for (source_name, article_id), group in id_parts.loc[duplicate_mask].groupby(
        ["source", id_col], dropna=False
    ):
        conflicting = [
            column for column in label_candidates
            if column in group.columns and group[column].dropna().map(norm_label).nunique() > 1
        ]
        if conflicting:
            conflict_rows.append({
                "source": source_name,
                "article_id": article_id,
                "conflict_columns": ";".join(conflicting),
            })

    before_dedup = len(df)
    identified = df.loc[valid_id_mask].drop_duplicates(subset=["source", id_col], keep="last")
    unidentified = df.loc[~valid_id_mask]
    df = pd.concat([identified, unidentified], ignore_index=True, sort=False)
    print("Dòng trùng article_id bên trong cùng nguồn bị loại:", before_dedup - len(df))

pd.DataFrame(conflict_rows, columns=["source", "article_id", "conflict_columns"]).to_csv(
    OUTPUT_DIR / "duplicate_label_conflicts.csv", index=False
)


def title_group(value):
    value = re.sub(r"\W+", " ", str(value).lower(), flags=re.UNICODE).strip()
    return hashlib.md5(value.encode("utf-8")).hexdigest()


df["_title_group"] = df.get("title", df["model_text"]).map(title_group)
if "event_id" in df.columns and df["event_id"].notna().mean() >= 0.5:
    event_values = df["event_id"].map(clean_scalar)
    df["split_group"] = event_values.fillna(df["_title_group"]).astype(str)
else:
    df["split_group"] = df["_title_group"]

df = df.reset_index(drop=True)
df["_row_key"] = np.arange(len(df), dtype=np.int64)

source_clean_summary = df.groupby("source").agg(
    clean_articles=("_row_key", "size"),
    unique_groups=("split_group", "nunique"),
).reset_index()
source_clean_summary.to_csv(OUTPUT_DIR / "source_clean_article_counts.csv", index=False)
display(source_clean_summary)
print(f"Dữ liệu sạch: {len(df):,} bài; {df['split_group'].nunique():,} groups")
display(df[[column for column in [id_col, "source", "title", "_source_file"] if column and column in df]].head())


# === Cell 9 ===
# Cell 7 — Chuẩn hóa nhãn và tạo ba task E/S/G nhị phân
# Không còn sử dụng esg_pillar = E/S/G/Mixed làm task dự đoán.

LABEL_MAPS = {
    "esg_relevance": {
        "relevant": "ESG",
        "esg": "ESG",
        "not_relevant": "Non_ESG",
        "non_esg": "Non_ESG",
        "nonesg": "Non_ESG",
        "uncertain": "Uncertain",
    },

    # Sentiment còn ba lớp:
    # positive, negative, mixed
    "sentiment": {
        "positive": "positive",
        "negative": "negative",
        "neutral": "mixed",
        "mixed": "mixed",
    },

    "content_polarity": {
        "negative": "Negative",
        "positive": "Positive",
        "mixed": "Mixed",
    },
}


def canonicalize_column(
    series,
    column,
):
    normalized = series.map(
        norm_label
    )

    mapping = LABEL_MAPS.get(
        column,
        {},
    )

    return normalized.map(
        lambda value: (
            mapping.get(value, value)
            if pd.notna(value)
            else np.nan
        )
    )


# ============================================================
# 1. Chuẩn hóa các nhãn thông thường
# ============================================================

if "esg_relevance" in df.columns:
    df["esg_relevance"] = canonicalize_column(
        df["esg_relevance"],
        "esg_relevance",
    )


for column in [
    "sentiment",
    "content_polarity",
    "esg_impact",
    "esg_information_type",
    "article_type",
    "event_status",
    "severity",
    "materiality",
    "information_novelty",
    "company_role",
    "corporate_response",
    "promotion_label",
    "target_type",
]:
    if column in df.columns:
        df[column] = canonicalize_column(
            df[column],
            column,
        )


# ============================================================
# 2. Hàm chuyển giá trị E/S/G thành 0/1
# ============================================================

BINARY_POSITIVE_VALUES = {
    "1",
    "1.0",
    "true",
    "yes",
    "y",
    "co",
    "có",
    "present",
    "positive",
}

BINARY_NEGATIVE_VALUES = {
    "0",
    "0.0",
    "false",
    "no",
    "n",
    "khong",
    "không",
    "absent",
    "negative",
}


def to_binary(value):
    """
    Chuyển nhãn về:
    1 = trụ cột xuất hiện
    0 = trụ cột không xuất hiện
    NA = thiếu hoặc không xác định
    """

    value = norm_label(value)

    if pd.isna(value):
        return pd.NA

    if value in BINARY_POSITIVE_VALUES:
        return 1

    if value in BINARY_NEGATIVE_VALUES:
        return 0

    return pd.NA


PILLAR_BINARY_COLUMNS = [
    "environmental",
    "social",
    "governance",
]


# Tạo cột nếu file chưa có.
for column in PILLAR_BINARY_COLUMNS:

    if column not in df.columns:
        df[column] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Int64",
        )

    else:
        df[column] = (
            df[column]
            .map(to_binary)
            .astype("Int64")
        )


# ============================================================
# 3. Hỗ trợ dữ liệu cũ chỉ có esg_pillar
# ============================================================
# Chỉ suy ra được các trường hợp đơn nghĩa E, S hoặc G.
# Không suy diễn Mixed thành E=S=G=1.

LEGACY_SINGLE_PILLAR_MAP = {
    "e": (1, 0, 0),
    "environmental": (1, 0, 0),

    "s": (0, 1, 0),
    "social": (0, 1, 0),

    "g": (0, 0, 1),
    "governance": (0, 0, 1),
}


if "esg_pillar" in df.columns:

    legacy_pillar = (
        df["esg_pillar"]
        .map(norm_label)
    )

    legacy_flags = legacy_pillar.map(
        LEGACY_SINGLE_PILLAR_MAP
    )

    for pillar_index, column in enumerate(
        PILLAR_BINARY_COLUMNS
    ):
        inferred_values = legacy_flags.map(
            lambda value: (
                value[pillar_index]
                if isinstance(value, tuple)
                else pd.NA
            )
        ).astype("Int64")

        # Chỉ điền những ô đang thiếu.
        fill_mask = (
            df[column].isna()
            & inferred_values.notna()
        )

        df.loc[
            fill_mask,
            column,
        ] = inferred_values.loc[
            fill_mask
        ]

    # Mixed cũ không cho biết chính xác trụ cột nào xuất hiện.
    legacy_mixed_mask = legacy_pillar.isin({
        "mixed",
        "mixed_esg",
        "esg_mixed",
    })

    unresolved_mixed_mask = (
        legacy_mixed_mask
        & df[PILLAR_BINARY_COLUMNS]
        .isna()
        .all(axis=1)
    )

    print(
        "Số bài Mixed cũ không thể tự chuyển thành E/S/G:",
        int(unresolved_mixed_mask.sum()),
    )

    if unresolved_mixed_mask.any():

        display_columns = [
            column
            for column in [
                id_col,
                "title",
                "esg_pillar",
                "environmental",
                "social",
                "governance",
                "_source_file",
            ]
            if column and column in df.columns
        ]

        display(
            df.loc[
                unresolved_mixed_mask,
                display_columns,
            ].head(30)
        )


# Bảo đảm ba cột có kiểu số nguyên nullable 0/1.
for column in PILLAR_BINARY_COLUMNS:
    df[column] = (
        df[column]
        .astype("Int64")
    )

    invalid_mask = (
        df[column].notna()
        & ~df[column].isin([0, 1])
    )

    assert not invalid_mask.any(), (
        f"Cột {column} còn giá trị ngoài 0/1"
    )


# ============================================================
# 4. Kiểm tra phân bố E/S/G trong các bài ESG
# ============================================================

if "esg_relevance" in df.columns:
    esg_check = df.loc[
        df["esg_relevance"].eq("ESG")
    ].copy()
else:
    esg_check = df.copy()


pillar_distribution_rows = []

for column in PILLAR_BINARY_COLUMNS:

    counts = (
        esg_check[column]
        .value_counts(dropna=False)
    )

    n_valid = int(
        esg_check[column].notna().sum()
    )

    n_positive = int(
        esg_check[column].eq(1).sum()
    )

    n_negative = int(
        esg_check[column].eq(0).sum()
    )

    n_missing = int(
        esg_check[column].isna().sum()
    )

    positive_rate = (
        n_positive / n_valid
        if n_valid > 0
        else np.nan
    )

    pillar_distribution_rows.append({
        "pillar_task": column,
        "n_valid": n_valid,
        "n_positive_1": n_positive,
        "n_negative_0": n_negative,
        "n_missing": n_missing,
        "positive_rate": positive_rate,
    })


pillar_binary_distribution = pd.DataFrame(
    pillar_distribution_rows
)

print(
    "\nPhân bố ba nhãn E/S/G "
    "trong tập bài ESG:"
)

display(
    pillar_binary_distribution
)


# Kiểm tra số lượng trụ cột cùng xuất hiện trong mỗi bài.
esg_check["n_active_pillars"] = (
    esg_check[PILLAR_BINARY_COLUMNS]
    .fillna(0)
    .sum(axis=1)
    .astype(int)
)

print(
    "\nSố trụ cột ESG xuất hiện đồng thời:"
)

display(
    esg_check["n_active_pillars"]
    .value_counts(dropna=False)
    .sort_index()
    .rename("n_articles")
    .to_frame()
)


# ============================================================
# 5. Kiểm tra sentiment
# ============================================================

if "sentiment" in df.columns:

    valid_sentiment = {
        "positive",
        "negative",
        "mixed",
    }

    invalid_sentiment_mask = (
        df["sentiment"].notna()
        & ~df["sentiment"].isin(
            valid_sentiment
        )
    )

    print(
        "\nNhãn sentiment không hợp lệ:",
        int(invalid_sentiment_mask.sum()),
    )

    if invalid_sentiment_mask.any():

        display_columns = [
            column
            for column in [
                id_col,
                "title",
                "sentiment",
                "_source_file",
            ]
            if column and column in df.columns
        ]

        display(
            df.loc[
                invalid_sentiment_mask,
                display_columns,
            ]
        )

    print(
        "\nPhân bố sentiment sau khi "
        "gộp neutral → mixed:"
    )

    display(
        df["sentiment"]
        .value_counts(dropna=False)
        .rename("n")
        .to_frame()
    )


print(
    "\nĐã chuẩn hóa nhãn. "
    "Không còn sử dụng esg_pillar "
    "làm task dự đoán."
)


# === Cell 10 ===
# Cell 8 — Khai báo task và kiểm tra phân bố lớp
# E, S, G là ba task nhị phân độc lập.

VALID_SENTIMENT_LABELS = {
    "positive",
    "negative",
    "mixed",
}

VALID_BINARY_LABELS = {
    "0",
    "1",
}


TASK_SPECS = {
    "esg_relevance": {
        "column": "esg_relevance",
        "filter_esg": False,
        "valid_labels": {"ESG", "Non_ESG"},
    },

    "sentiment_all": {
        "column": "sentiment",
        "filter_esg": False,
        "valid_labels": VALID_SENTIMENT_LABELS,
    },

    "sentiment_esg": {
        "column": "sentiment",
        "filter_esg": True,
        "valid_labels": VALID_SENTIMENT_LABELS,
    },

    "content_polarity": {
        "column": "content_polarity",
        "filter_esg": True,
    },

    # Ba task ESG nhị phân độc lập
    "environmental": {
        "column": "environmental",
        "filter_esg": True,
        "task_type": "binary",
        "valid_labels": VALID_BINARY_LABELS,
        "positive_label": "1",
    },

    "social": {
        "column": "social",
        "filter_esg": True,
        "task_type": "binary",
        "valid_labels": VALID_BINARY_LABELS,
        "positive_label": "1",
    },

    "governance": {
        "column": "governance",
        "filter_esg": True,
        "task_type": "binary",
        "valid_labels": VALID_BINARY_LABELS,
        "positive_label": "1",
    },

    "esg_impact": {
        "column": "esg_impact",
        "filter_esg": True,
    },

    "esg_information_type": {
        "column": "esg_information_type",
        "filter_esg": True,
    },

    "article_type": {
        "column": "article_type",
        "filter_esg": True,
    },

    "event_status": {
        "column": "event_status",
        "filter_esg": True,
    },

    "severity": {
        "column": "severity",
        "filter_esg": True,
    },

    "materiality": {
        "column": "materiality",
        "filter_esg": True,
    },

    "information_novelty": {
        "column": "information_novelty",
        "filter_esg": True,
    },

    "company_role": {
        "column": "company_role",
        "filter_esg": True,
    },

    "corporate_response": {
        "column": "corporate_response",
        "filter_esg": True,
    },

    "promotion_label": {
        "column": "promotion_label",
        "filter_esg": False,
    },

    "target_type": {
        "column": "target_type",
        "filter_esg": False,
    },
}


NA_LABELS = {
    "",
    "nan",
    "none",
    "n/a",
    "na",
    "not_applicable",
    "unknown",
}


def get_task_frame(
    task_name,
    verbose=True,
):
    spec = TASK_SPECS[task_name]
    column = spec["column"]

    if column not in df.columns:
        raise KeyError(
            f"Không có cột {column}"
        )

    work = df.copy()

    # Các task E/S/G chỉ chạy trên bài đã xác định là ESG.
    if spec["filter_esg"]:

        if "esg_relevance" not in work.columns:
            raise KeyError(
                "Task ESG cần cột "
                "esg_relevance để lọc"
            )

        work = work.loc[
            work["esg_relevance"].eq("ESG")
        ].copy()

    # Xử lý riêng task nhị phân.
    if spec.get("task_type") == "binary":

        binary_values = (
            work[column]
            .map(to_binary)
            .astype("Int64")
        )

        # Chuyển thành chuỗi "0"/"1" để LabelEncoder
        # và các mô hình dùng thống nhất.
        work["label"] = (
            binary_values
            .astype("string")
        )

    else:
        work["label"] = (
            work[column]
            .map(clean_scalar)
        )

    # Loại giá trị thiếu.
    work = work.loc[
        work["label"].notna()
    ].copy()

    # Loại các ký hiệu NA dạng chuỗi.
    work = work.loc[
        ~work["label"]
        .astype(str)
        .str.lower()
        .isin(NA_LABELS)
    ].copy()

    # Chuyển label thành chuỗi thống nhất.
    work["label"] = (
        work["label"]
        .astype(str)
        .str.strip()
    )

    valid_labels = spec.get(
        "valid_labels"
    )

    if valid_labels is not None:

        invalid_label_mask = (
            ~work["label"]
            .isin(valid_labels)
        )

        if invalid_label_mask.any():

            print(
                f"{task_name}: loại "
                f"{int(invalid_label_mask.sum())} "
                "nhãn không hợp lệ"
            )

            work = work.loc[
                ~invalid_label_mask
            ].copy()

    counts = (
        work["label"]
        .value_counts()
    )

    if verbose:

        print(
            task_name,
            "— n =",
            len(work),
            "— số lớp =",
            len(counts),
        )

        display(
            counts
            .rename("n")
            .to_frame()
        )

        if spec.get("task_type") == "binary":

            n_positive = int(
                work["label"]
                .eq("1")
                .sum()
            )

            positive_rate = (
                n_positive / len(work)
                if len(work) > 0
                else np.nan
            )

            print(
                "Positive rate:",
                round(
                    positive_rate,
                    4,
                ),
            )

    return work


# ============================================================
# Tổng hợp quy mô task
# ============================================================

task_rows = []

for task_name, spec in TASK_SPECS.items():

    if spec["column"] not in df.columns:
        continue

    try:
        work = get_task_frame(
            task_name,
            verbose=False,
        )

        counts = (
            work["label"]
            .value_counts()
        )

        available = (
            len(work) >= MIN_TOTAL_FOR_TASK
            and len(counts) >= 2
            and counts.min()
            >= MIN_CLASS_FOR_TASK
        )

        is_binary = (
            spec.get("task_type")
            == "binary"
        )

        task_rows.append({
            "task": task_name,
            "column": spec["column"],
            "task_type": (
                "binary"
                if is_binary
                else "multiclass"
            ),
            "filter_esg": spec["filter_esg"],
            "n": len(work),
            "n_classes": len(counts),
            "min_class": (
                int(counts.min())
                if len(counts)
                else 0
            ),
            "n_negative_0": (
                int(counts.get("0", 0))
                if is_binary
                else np.nan
            ),
            "n_positive_1": (
                int(counts.get("1", 0))
                if is_binary
                else np.nan
            ),
            "positive_rate": (
                float(counts.get("1", 0) / len(work))
                if is_binary and len(work) > 0
                else np.nan
            ),
            "available_for_pilot": available,
            "distribution": dict(counts),
        })

    except Exception as exc:

        task_rows.append({
            "task": task_name,
            "column": spec["column"],
            "task_type": spec.get(
                "task_type",
                "multiclass",
            ),
            "filter_esg": spec["filter_esg"],
            "n": 0,
            "n_classes": 0,
            "min_class": 0,
            "n_negative_0": np.nan,
            "n_positive_1": np.nan,
            "positive_rate": np.nan,
            "available_for_pilot": False,
            "distribution": {},
            "error": str(exc),
        })


task_summary = pd.DataFrame(
    task_rows
)

display(
    task_summary
)

task_summary.to_csv(
    OUTPUT_DIR
    / "task_label_summary.csv",
    index=False,
)


AVAILABLE_TASKS = (
    task_summary.loc[
        task_summary[
            "available_for_pilot"
        ].eq(True),
        "task",
    ]
    .tolist()
)

print(
    "\nAVAILABLE_TASKS =",
    AVAILABLE_TASKS,
)

assert AVAILABLE_TASKS, (
    "Chưa có task nào đủ tối thiểu "
    "để chạy thử."
)


# ============================================================
# Kiểm tra riêng ba task E/S/G
# ============================================================

for pillar_task in [
    "environmental",
    "social",
    "governance",
]:
    print(
        f"\nPhân bố {pillar_task}:"
    )

    if pillar_task in TASK_SPECS:
        get_task_frame(
            pillar_task,
            verbose=True,
        )


# Kiểm tra esg_pillar đã bị loại khỏi danh sách task.
assert "esg_pillar" not in TASK_SPECS
assert "esg_pillar" not in AVAILABLE_TASKS

print(
    "\nĐã thay esg_pillar bằng ba task "
    "nhị phân độc lập: "
    "environmental, social, governance."
)


# === Cell 11 ===
# Điều chỉnh số lượng validation/test theo số bài CafeF thực tế

if EXPERIMENT_MODE == "cross_source_target_validation":

    cafef_available = int(
        df["source"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(SOURCE_TARGET.lower())
        .sum()
    )

    requested_validation = int(
        TARGET_VALIDATION_SIZE
    )

    requested_test = int(
        TARGET_TEST_SIZE
    )

    requested_total = (
        requested_validation
        + requested_test
    )

    if cafef_available < 2:

        raise ValueError(
            "CafeF phải có ít nhất 2 bài hợp lệ "
            "để chia validation và test."
        )

    if cafef_available < requested_total:

        test_ratio = (
            requested_test
            / requested_total
        )

        TARGET_TEST_SIZE = round(
            cafef_available
            * test_ratio
        )

        TARGET_TEST_SIZE = max(
            1,
            min(
                TARGET_TEST_SIZE,
                cafef_available - 1,
            ),
        )

        TARGET_VALIDATION_SIZE = (
            cafef_available
            - TARGET_TEST_SIZE
        )

        print(
            f"CafeF hợp lệ: {cafef_available:,} bài"
        )

        print(
            f"Mục tiêu ban đầu: "
            f"{requested_validation:,} validation + "
            f"{requested_test:,} test"
        )

        print(
            f"Điều chỉnh thành: "
            f"{TARGET_VALIDATION_SIZE:,} validation + "
            f"{TARGET_TEST_SIZE:,} test"
        )

    else:

        print(
            f"CafeF hợp lệ: {cafef_available:,} bài."
        )

        print(
            f"Giữ nguyên: "
            f"{TARGET_VALIDATION_SIZE:,} validation + "
            f"{TARGET_TEST_SIZE:,} test."
        )


# === Cell 13 ===
# Cell 9 — Khóa một split manifest chung và kiểm tra feasibility cho từng task
from sklearn.model_selection import GroupShuffleSplit


def build_strata(frame):
    if "esg_relevance" not in frame.columns:
        return None

    esg = frame["esg_relevance"].fillna("missing").astype(str)
    if "sentiment" in frame.columns:
        combined = esg + "|" + frame["sentiment"].fillna("missing").astype(str)
        if combined.value_counts().min() >= 2:
            return combined

    if esg.nunique() >= 2 and esg.value_counts().min() >= 2:
        return esg
    return None


def partition_frame(frame, requested_size, seed, purpose):
    frame = frame.reset_index(drop=True).copy()
    requested_size = int(requested_size)
    if requested_size <= 0 or requested_size >= len(frame):
        raise ValueError(f"{purpose}: requested_size={requested_size}; available={len(frame)}")

    groups = frame["split_group"].astype(str)
    if groups.nunique() == len(frame):
        strata = build_strata(frame)
        try:
            remaining, selected = train_test_split(
                frame,
                test_size=requested_size,
                stratify=strata,
                random_state=seed,
            )
        except ValueError:
            remaining, selected = train_test_split(
                frame,
                test_size=requested_size,
                stratify=None,
                random_state=seed,
            )
        return remaining.reset_index(drop=True), selected.reset_index(drop=True)

    strata = build_strata(frame)
    reference = None if strata is None else strata.value_counts(normalize=True)
    splitter = GroupShuffleSplit(
        n_splits=GROUP_SPLIT_SEARCH_TRIALS,
        test_size=requested_size / len(frame),
        random_state=seed,
    )
    best = None
    for remainder_indices, chosen_indices in splitter.split(frame, groups=groups):
        chosen = frame.iloc[chosen_indices]
        count_difference = abs(len(chosen) - requested_size)
        distribution_difference = 0.0
        if reference is not None:
            observed = build_strata(chosen)
            if observed is not None:
                observed = observed.value_counts(normalize=True)
                distribution_difference = float(
                    reference.sub(observed, fill_value=0).abs().sum()
                )
        candidate = (count_difference, distribution_difference, remainder_indices, chosen_indices)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
            if count_difference == 0 and distribution_difference < 0.01:
                break

    count_difference, _, remainder_indices, chosen_indices = best
    if count_difference and REQUIRE_EXACT_TARGET_COUNTS:
        raise ValueError(
            f"{purpose}: không thể chọn đúng {requested_size} bài mà vẫn giữ nguyên nhóm sự kiện. "
            f"Kích thước gần nhất là {len(chosen_indices)}. Kiểm tra duplicate/event groups hoặc điều chỉnh ngưỡng."
        )
    if count_difference:
        warnings.warn(f"{purpose}: chọn {len(chosen_indices)} thay vì {requested_size} để tránh group leakage.")

    return (
        frame.iloc[remainder_indices].reset_index(drop=True),
        frame.iloc[chosen_indices].reset_index(drop=True),
    )


def limit_training_rows(frame, requested_size, seed):
    if requested_size is None:
        return frame.reset_index(drop=True)
    requested_size = int(requested_size)
    if len(frame) < requested_size:
        raise ValueError(
            f"Vietstock hợp lệ chỉ còn {len(frame)} bài, thấp hơn TRAIN_SAMPLE_SIZE={requested_size}."
        )
    if len(frame) == requested_size:
        return frame.reset_index(drop=True)
    _, selected = partition_frame(frame, requested_size, seed, "fixed training budget")
    return selected


def verify_split_overlap(parts, context):
    keys = {name: set(part["_row_key"].astype(int)) for name, part in parts.items()}
    groups = {name: set(part["split_group"].astype(str)) for name, part in parts.items()}
    titles = {name: set(part["_title_group"].astype(str)) for name, part in parts.items()}
    diagnostics = {}
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        diagnostics[f"{first}_{second}_rows"] = len(keys[first] & keys[second])
        diagnostics[f"{first}_{second}_groups"] = len(groups[first] & groups[second])
        diagnostics[f"{first}_{second}_titles"] = len(titles[first] & titles[second])
    if any(diagnostics.values()):
        raise AssertionError(f"{context}: phát hiện article/group/title leakage: {diagnostics}")
    return diagnostics


source_train_pool = df.loc[df["source"].eq(SOURCE_TRAIN)].copy()
source_target_pool = df.loc[df["source"].eq(SOURCE_TARGET)].copy()
if source_train_pool.empty or source_target_pool.empty:
    raise ValueError(f"Thiếu dữ liệu nguồn train hoặc target: {dict(df['source'].value_counts())}")

overlap_groups = set(source_train_pool["split_group"].astype(str)) & set(
    source_target_pool["split_group"].astype(str)
)
overlap_titles = set(source_train_pool["_title_group"].astype(str)) & set(
    source_target_pool["_title_group"].astype(str)
)

cross_source_overlap_frame = source_train_pool.loc[
    source_train_pool["split_group"].astype(str).isin(overlap_groups)
    | source_train_pool["_title_group"].astype(str).isin(overlap_titles)
].copy()
cross_source_overlap_frame.to_csv(OUTPUT_DIR / "cross_source_train_overlap_removed.csv", index=False)

if not cross_source_overlap_frame.empty:
    if not REMOVE_CROSS_SOURCE_GROUP_OVERLAP:
        raise ValueError(
            f"Có {len(cross_source_overlap_frame)} bài Vietstock trùng nhóm/title với CafeF; "
            "bật REMOVE_CROSS_SOURCE_GROUP_OVERLAP trước khi đánh giá liên nguồn."
        )
    source_train_pool = source_train_pool.loc[
        ~source_train_pool["_row_key"].isin(cross_source_overlap_frame["_row_key"])
    ].copy()
    print("Bài Vietstock loại để bảo vệ holdout CafeF:", len(cross_source_overlap_frame))

if EXPERIMENT_MODE == "cross_source_target_validation":
    needed_target = TARGET_VALIDATION_SIZE + TARGET_TEST_SIZE
    if len(source_target_pool) < needed_target:
        raise ValueError(
            f"CafeF hợp lệ có {len(source_target_pool)} bài nhưng cần {needed_target} "
            f"({TARGET_VALIDATION_SIZE} validation + {TARGET_TEST_SIZE} test)."
        )

    if len(source_target_pool) > needed_target:
        _, target_selected = partition_frame(
            source_target_pool, needed_target, SEED + 11, "CafeF evaluation budget"
        )
    else:
        target_selected = source_target_pool.reset_index(drop=True)

    global_validation, global_test = partition_frame(
        target_selected, TARGET_TEST_SIZE, SEED + 12, "CafeF test holdout"
    )
    global_train = limit_training_rows(source_train_pool, TRAIN_SAMPLE_SIZE, SEED + 13)

elif EXPERIMENT_MODE == "cross_source_strict":
    train_remainder, global_validation = partition_frame(
        source_train_pool, STRICT_SOURCE_VALIDATION_SIZE, SEED + 21, "Vietstock validation"
    )
    global_train = limit_training_rows(train_remainder, TRAIN_SAMPLE_SIZE, SEED + 22)
    if len(source_target_pool) < TARGET_TEST_SIZE:
        raise ValueError(f"CafeF chỉ có {len(source_target_pool)} bài; cần {TARGET_TEST_SIZE} bài test.")
    if len(source_target_pool) == TARGET_TEST_SIZE:
        global_test = source_target_pool.reset_index(drop=True)
    else:
        _, global_test = partition_frame(source_target_pool, TARGET_TEST_SIZE, SEED + 23, "CafeF test")

else:
    all_articles = pd.concat([source_train_pool, source_target_pool], ignore_index=True)
    train_and_validation, global_test = partition_frame(
        all_articles, max(1, round(len(all_articles) * 0.15)), SEED + 31, "mixed-source test"
    )
    global_train, global_validation = partition_frame(
        train_and_validation,
        max(1, round(len(all_articles) * 0.15)),
        SEED + 32,
        "mixed-source validation",
    )
    global_train = limit_training_rows(global_train, TRAIN_SAMPLE_SIZE, SEED + 33)

GLOBAL_SPLITS = {
    "train": global_train.reset_index(drop=True),
    "validation": global_validation.reset_index(drop=True),
    "test": global_test.reset_index(drop=True),
}
global_overlap_diagnostics = verify_split_overlap(GLOBAL_SPLITS, "global split")

manifest_frames = []
for split_name, part in GLOBAL_SPLITS.items():
    manifest_columns = [
        column for column in ["_row_key", id_col, "source", "split_group", "_title_group", "title", "_source_file"]
        if column and column in part.columns
    ]
    manifest_part = part[manifest_columns].copy()
    manifest_part["split"] = split_name
    manifest_part["experiment_mode"] = EXPERIMENT_MODE
    manifest_frames.append(manifest_part)

split_manifest = pd.concat(manifest_frames, ignore_index=True)
split_manifest.to_csv(OUTPUT_DIR / "global_article_split_manifest.csv", index=False)

global_split_summary = split_manifest.groupby(["split", "source"]).size().rename("n_articles").reset_index()
display(global_split_summary)
global_split_summary.to_csv(OUTPUT_DIR / "global_split_source_summary.csv", index=False)
print("Global leakage checks:", global_overlap_diagnostics)


def make_splits(work, seed=SEED):
    del seed  # Mọi task tham chiếu cùng global manifest; không tạo split ngẫu nhiên mới.
    work = work.copy()
    if work["_row_key"].duplicated().any():
        raise ValueError("Task frame có _row_key trùng; không thể kiểm chứng fairness.")

    task_parts = {}
    for split_name, global_part in GLOBAL_SPLITS.items():
        allowed_keys = set(global_part["_row_key"].astype(int))
        task_parts[split_name] = work.loc[work["_row_key"].isin(allowed_keys)].reset_index(drop=True)

    verify_split_overlap(task_parts, f"task {work['label'].name}")
    train, validation, test = (task_parts[name] for name in ["train", "validation", "test"])

    if any(part.empty for part in [train, validation, test]):
        raise ValueError(
            f"Task không đủ bài trên ít nhất một split: train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}."
        )

    train_labels = set(train["label"].astype(str))
    validation_labels = set(validation["label"].astype(str))
    test_labels = set(test["label"].astype(str))
    unknown_validation = validation_labels - train_labels
    unknown_test = test_labels - train_labels
    if unknown_validation or unknown_test:
        raise ValueError(
            f"CafeF có lớp chưa xuất hiện trong Vietstock train: "
            f"validation={sorted(unknown_validation)}, test={sorted(unknown_test)}."
        )
    if len(train_labels) < 2:
        raise ValueError("Train chỉ còn một lớp sau khi lọc task.")
    if len(validation_labels) < MIN_VALIDATION_CLASSES:
        raise ValueError(f"Validation chỉ có {len(validation_labels)} lớp: {sorted(validation_labels)}.")
    if len(test_labels) < MIN_TEST_CLASSES:
        raise ValueError(f"Test chỉ có {len(test_labels)} lớp: {sorted(test_labels)}.")

    print(EXPERIMENT_MODE, {"train": len(train), "validation": len(validation), "test": len(test)})
    for name, part in task_parts.items():
        print(name, "sources:", dict(part["source"].value_counts()), "labels:", dict(part["label"].value_counts()))
    return train, validation, test


task_split_rows = []
feasible_tasks = []
for task_name in AVAILABLE_TASKS:
    try:
        task_work = get_task_frame(task_name, verbose=False)
        train_part, validation_part, test_part = make_splits(task_work)
        audit_record = {
            "task": task_name,
            "status": "eligible",
            "experiment_mode": EXPERIMENT_MODE,
            "n_train": len(train_part),
            "n_validation": len(validation_part),
            "n_test": len(test_part),
            "train_sources": ";".join(sorted(train_part["source"].unique())),
            "validation_sources": ";".join(sorted(validation_part["source"].unique())),
            "test_sources": ";".join(sorted(test_part["source"].unique())),
            "train_label_counts": json.dumps(
                {str(label): int(count) for label, count in train_part["label"].value_counts().items()},
                ensure_ascii=False,
            ),
            "validation_label_counts": json.dumps(
                {str(label): int(count) for label, count in validation_part["label"].value_counts().items()},
                ensure_ascii=False,
            ),
            "test_label_counts": json.dumps(
                {str(label): int(count) for label, count in test_part["label"].value_counts().items()},
                ensure_ascii=False,
            ),
            "error": "",
        }
        task_split_rows.append(audit_record)
        feasible_tasks.append(task_name)
    except Exception as error:
        task_split_rows.append({
            "task": task_name, "status": "excluded", "experiment_mode": EXPERIMENT_MODE,
            "n_train": 0, "n_validation": 0, "n_test": 0,
            "error": f"{type(error).__name__}: {error}",
        })
        warnings.warn(f"Bỏ qua {task_name}: {error}")

task_split_audit = pd.DataFrame(task_split_rows)
task_split_audit.to_csv(OUTPUT_DIR / "task_split_audit.csv", index=False)
display(task_split_audit)
AVAILABLE_TASKS = feasible_tasks
print("Các task đủ điều kiện trên cả train/validation/test:", AVAILABLE_TASKS)
assert AVAILABLE_TASKS, "Không task nào có đủ dữ liệu hợp lệ ở cả Vietstock train và CafeF evaluation."


# === Cell 14 ===
# Cell 9B — Kiểm tra global split với lần chạy PhoBERT (nếu đã có)
PHOBERT_MANIFEST_PATH = PHOBERT_RESULTS_DIR / "global_article_split_manifest.csv"

if PHOBERT_MANIFEST_PATH.exists():
    phobert_manifest = pd.read_csv(PHOBERT_MANIFEST_PATH)
    required_manifest_columns = {"_row_key", "split"}
    if not required_manifest_columns.issubset(phobert_manifest.columns):
        raise ValueError(
            "Manifest PhoBERT thiếu _row_key/split; không thể xác minh so sánh công bằng."
        )

    current_pairs = set(
        split_manifest[["_row_key", "split"]].astype(str).itertuples(index=False, name=None)
    )
    phobert_pairs = set(
        phobert_manifest[["_row_key", "split"]].astype(str).itertuples(index=False, name=None)
    )
    if current_pairs != phobert_pairs:
        only_xlmr = len(current_pairs - phobert_pairs)
        only_phobert = len(phobert_pairs - current_pairs)
        raise AssertionError(
            "Global split khác PhoBERT: "
            f"{only_xlmr} cặp chỉ có ở XLM-R, {only_phobert} cặp chỉ có ở PhoBERT. "
            "Không huấn luyện trước khi khôi phục đúng dữ liệu/split cũ."
        )
    print("✓ Global article split trùng hoàn toàn với PhoBERT.")
else:
    warnings.warn(
        f"Chưa thấy manifest PhoBERT tại {PHOBERT_MANIFEST_PATH}. "
        "Notebook vẫn chạy split xác định bằng SEED; bước so sánh cuối sẽ kiểm tra "
        "task-level fingerprints khi kết quả PhoBERT xuất hiện."
    )


# === Cell 15 ===
# Cell 10 — Baseline dùng ĐÚNG train của XLM-RoBERTa; tuning/chọn model chỉ trên validation
BASELINE_MODELS = {
    "majority_class": Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, min_df=1, max_features=5000)),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ]),
    "tfidf_logreg": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.98,
            max_features=50000, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2500, class_weight="balanced", C=2.0, random_state=SEED,
        )),
    ]),
    "tfidf_linearsvm": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.98,
            max_features=50000, sublinear_tf=True,
        )),
        ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=SEED)),
    ]),
}

BASELINE_PARAM_GRIDS = {
    "majority_class": [{}],
    "tfidf_logreg": [{"clf__C": value} for value in BASELINE_LOGREG_C_GRID],
    "tfidf_linearsvm": [{"clf__C": value} for value in BASELINE_SVM_C_GRID],
}


def metric_row(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def article_set_fingerprint(frame):
    joined = ",".join(map(str, sorted(frame["_row_key"].astype(int))))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


baseline_rows = []
baseline_tuning_rows = []
baseline_predictions = {}
split_cache = {}

for task_name in RESEARCH_TASKS:
    if task_name not in AVAILABLE_TASKS:
        print(f"Bỏ qua baseline {task_name}: task không đủ ngưỡng.")
        continue
    print("\n" + "=" * 80)
    print("TASK:", task_name)
    work = get_task_frame(task_name, verbose=False)
    train, validation, test = make_splits(work)
    split_cache[task_name] = (train.copy(), validation.copy(), test.copy())

    split_fingerprints = {
        "train_fingerprint": article_set_fingerprint(train),
        "validation_fingerprint": article_set_fingerprint(validation),
        "test_fingerprint": article_set_fingerprint(test),
    }

    for model_name, template in BASELINE_MODELS.items():
        candidates = []
        for params in BASELINE_PARAM_GRIDS[model_name]:
            model = clone(template).set_params(**params)

            # Nguyên tắc cốt lõi: KHÔNG ghép validation vào train.
            model.fit(train["model_text"], train["label"])
            validation_prediction = model.predict(validation["model_text"])
            validation_metrics = metric_row(validation["label"], validation_prediction)
            candidate = {
                "task": task_name,
                "model": model_name,
                "params": json.dumps(params, sort_keys=True),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_f1_macro": validation_metrics["f1_macro"],
                "validation_f1_weighted": validation_metrics["f1_weighted"],
            }
            baseline_tuning_rows.append(candidate)
            candidates.append((candidate, model))

        selected_candidate, selected_model = sorted(
            candidates,
            key=lambda item: (
                item[0]["validation_f1_macro"],
                item[0]["validation_balanced_accuracy"],
                item[0]["validation_accuracy"],
            ),
            reverse=True,
        )[0]

        # Test chỉ dùng sau khi đã khóa cấu hình bằng validation.
        test_prediction = selected_model.predict(test["model_text"])
        row = {
            "task": task_name,
            "model": model_name,
            "experiment_mode": EXPERIMENT_MODE,
            "selected_params": selected_candidate["params"],
            "selection_metric": "validation_f1_macro",
            "validation_f1_macro": selected_candidate["validation_f1_macro"],
            "validation_balanced_accuracy": selected_candidate["validation_balanced_accuracy"],
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            **split_fingerprints,
            **metric_row(test["label"], test_prediction),
        }
        baseline_rows.append(row)
        baseline_predictions[(task_name, model_name)] = (test.copy(), test_prediction, selected_model)
        print(
            model_name,
            "validation Macro-F1=", round(row["validation_f1_macro"], 4),
            "test Macro-F1=", round(row["f1_macro"], 4),
            "n_train=", row["n_train"],
            "params=", row["selected_params"],
        )

baseline_results = pd.DataFrame(baseline_rows).sort_values(
    ["task", "validation_f1_macro", "validation_balanced_accuracy"],
    ascending=[True, False, False],
)
baseline_tuning_results = pd.DataFrame(baseline_tuning_rows)
baseline_results.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
baseline_tuning_results.to_csv(OUTPUT_DIR / "baseline_validation_hyperparameter_search.csv", index=False)

fairness_summary = baseline_results.groupby("task").agg(
    models=("model", "nunique"),
    train_count_variants=("n_train", "nunique"),
    validation_count_variants=("n_validation", "nunique"),
    test_count_variants=("n_test", "nunique"),
    train_set_variants=("train_fingerprint", "nunique"),
    validation_set_variants=("validation_fingerprint", "nunique"),
    test_set_variants=("test_fingerprint", "nunique"),
).reset_index()
fairness_columns = [column for column in fairness_summary.columns if column.endswith("_variants")]
assert fairness_summary[fairness_columns].eq(1).all().all(), "Baseline không dùng cùng article sets."
fairness_summary.to_csv(OUTPUT_DIR / "baseline_training_fairness_audit.csv", index=False)
display(baseline_results)
display(fairness_summary)


# === Cell 16 ===
# Cell 11 — Xem chi tiết baseline của một task
BASELINE_REPORT_TASK = "esg_relevance" if "esg_relevance" in AVAILABLE_TASKS else AVAILABLE_TASKS[0]
best_row = baseline_results.loc[baseline_results["task"].eq(BASELINE_REPORT_TASK)].sort_values(["validation_f1_macro", "validation_balanced_accuracy"], ascending=False).iloc[0]
key = (BASELINE_REPORT_TASK, best_row["model"])
test_base, pred_base, best_baseline_model = baseline_predictions[key]
labels_base = sorted(test_base["label"].unique())

print("Best baseline:", key)
print(classification_report(test_base["label"], pred_base, digits=4, zero_division=0))

cm = confusion_matrix(test_base["label"], pred_base, labels=labels_base)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_base, yticklabels=labels_base)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Baseline — {BASELINE_REPORT_TASK}")
plt.tight_layout(); plt.show()


# === Cell 18 ===
# Cell 12 — Xác định danh sách task XLM-RoBERTa sẽ chạy
if TRANSFORMER_TASKS == "all":
    TRANSFORMER_TASKS_TO_RUN = list(AVAILABLE_TASKS)
elif isinstance(TRANSFORMER_TASKS, str):
    TRANSFORMER_TASKS_TO_RUN = [TRANSFORMER_TASKS]
else:
    TRANSFORMER_TASKS_TO_RUN = list(TRANSFORMER_TASKS)

unknown_tasks = [
    task for task in TRANSFORMER_TASKS_TO_RUN
    if task not in TASK_SPECS
]
if unknown_tasks:
    raise KeyError(f"Task chưa được khai báo trong TASK_SPECS: {unknown_tasks}")

unavailable_tasks = [
    task for task in TRANSFORMER_TASKS_TO_RUN
    if task not in AVAILABLE_TASKS
]
if unavailable_tasks:
    warnings.warn(
        "Các task không đủ ngưỡng và sẽ bị bỏ qua: "
        + ", ".join(unavailable_tasks)
    )

TRANSFORMER_TASKS_TO_RUN = [
    task for task in TRANSFORMER_TASKS_TO_RUN
    if task in AVAILABLE_TASKS
]

assert TRANSFORMER_TASKS_TO_RUN, "Không có task XLM-RoBERTa nào đủ điều kiện để chạy."
print(f"Sẽ chạy {len(TRANSFORMER_TASKS_TO_RUN)} task:")
for index, task in enumerate(TRANSFORMER_TASKS_TO_RUN, start=1):
    print(f"  {index:02d}. {task}")

print(
    "\nLưu ý: thời gian gần tăng tuyến tính theo số task; "
    "mỗi task xử lý bốn view cho mỗi bài."
)


# === Cell 19 ===
# Cell 13 — Chuẩn hóa raw Vietnamese cho XLM-RoBERTa (không word segmentation)
from tqdm.auto import tqdm
tqdm.pandas()

uses_external_word_segmentation = False
if USE_VNCORENLP:
    warnings.warn(
        "XLM-RoBERTa không cần VnCoreNLP. Notebook sẽ bỏ qua USE_VNCORENLP=True "
        "để giữ input đúng thiết kế raw-text."
    )

def prepare_transformer_text(text):
    # Không cắt theo ký tự: tail phải được xác định sau tokenization.
    return "" if pd.isna(text) else re.sub(r"\s+", " ", str(text)).strip()

def first_existing_column(frame, candidates):
    return next((column for column in candidates if column in frame.columns), None)

title_col = first_existing_column(df, ["title"])
sapo_col = first_existing_column(df, ["description", "summary", "sapo"])
body_col = first_existing_column(df, ["content"])

input_sources = [
    ("title_transformer_text", title_col),
    ("sapo_transformer_text", sapo_col),
    ("body_transformer_text", body_col),
]

for output_col, source_col in input_sources:
    if source_col is None:
        if output_col == "body_transformer_text":
            print("Không có content; dùng model_text làm body dự phòng.")
            df[output_col] = df["model_text"].progress_apply(prepare_transformer_text)
        else:
            df[output_col] = ""
    else:
        print(f"Đang chuẩn hóa raw text {source_col} → {output_col} ...")
        df[output_col] = df[source_col].progress_apply(prepare_transformer_text)

input_cache = OUTPUT_DIR / "xlmr_input_corpus_all_tasks.parquet"
input_cache_cols = [
    column for column in [
        "_row_key", id_col, "source", "split_group", "title_transformer_text",
        "sapo_transformer_text", "body_transformer_text",
    ]
    if column and column in df.columns
]
df[input_cache_cols].to_parquet(input_cache, index=False)
print("Đã lưu cache raw text dùng chung:", input_cache)

input_lookup = df[[
    "_row_key", "title_transformer_text", "sapo_transformer_text",
    "body_transformer_text",
]].copy()

for cached_task, cached_parts in list(split_cache.items()):
    refreshed_parts = []
    for cached_part in cached_parts:
        stale_input = [
            column for column in [
                "title_transformer_text", "sapo_transformer_text",
                "body_transformer_text",
            ]
            if column in cached_part.columns
        ]
        cached_part = cached_part.drop(columns=stale_input)
        cached_part = cached_part.merge(
            input_lookup,
            on="_row_key",
            how="left",
            validate="one_to_one",
        )
        refreshed_parts.append(cached_part)
    split_cache[cached_task] = tuple(refreshed_parts)

print(f"Đã cập nhật raw text cho {len(split_cache)} split task.")


# === Cell 20 ===
# Cell 14 — Tokenizer và encoding four-view

from datasets import Dataset
from transformers import AutoTokenizer, default_data_collator


# ============================================================
# 1. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

CONTENT_BUDGET = (
    MAX_LENGTH
    - tokenizer.num_special_tokens_to_add(pair=False)
)

assert CONTENT_BUDGET > 0

print("MAX_LENGTH:", MAX_LENGTH)
print("CONTENT_BUDGET:", CONTENT_BUDGET)


# ============================================================
# 2. Chuẩn hóa text
# ============================================================

def safe_text(value):

    if value is None or pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return value


def ids_without_special_tokens(text):

    text = safe_text(text)

    if not text:
        return []

    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def prepare_fixed_length(token_ids):
    """
    Đóng gói token cho XLM-RoBERTa:

        <s> content </s> <pad> ...

    Không sử dụng tokenizer.prepare_for_model vì phương thức
    này không có trong XLMRobertaTokenizer của Transformers 5.x.
    """

    content_ids = list(
        token_ids[:CONTENT_BUDGET]
    )

    required_special_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    missing_special_ids = [
        name
        for name, value in required_special_ids.items()
        if value is None
    ]

    if missing_special_ids:
        raise ValueError(
            "Tokenizer thiếu special token IDs: "
            f"{missing_special_ids}"
        )

    input_ids = [
        tokenizer.bos_token_id,
        *content_ids,
        tokenizer.eos_token_id,
    ]

    attention_mask = [1] * len(input_ids)

    pad_length = MAX_LENGTH - len(input_ids)

    if pad_length < 0:
        raise AssertionError(
            f"Sequence dài {len(input_ids)} "
            f"vượt MAX_LENGTH={MAX_LENGTH}."
        )

    input_ids.extend(
        [tokenizer.pad_token_id] * pad_length
    )

    attention_mask.extend(
        [0] * pad_length
    )

    assert len(input_ids) == MAX_LENGTH
    assert len(attention_mask) == MAX_LENGTH

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# ============================================================
# 2B. Smoke test tokenizer
# ============================================================

_tokenizer_smoke = prepare_fixed_length(
    ids_without_special_tokens(
        "Kiểm tra XLM-RoBERTa"
    )
)

assert len(
    _tokenizer_smoke["input_ids"]
) == MAX_LENGTH

assert len(
    _tokenizer_smoke["attention_mask"]
) == MAX_LENGTH

assert (
    _tokenizer_smoke["input_ids"][0]
    == tokenizer.bos_token_id
)

assert sum(
    _tokenizer_smoke["attention_mask"]
) >= 2

print(
    "✓ Tokenizer smoke test PASS; "
    "không dùng prepare_for_model."
)


# ============================================================
# 3. Chia thân bài thành head / continuation / tail
# ============================================================

def split_body_into_views(body_ids):

    body_length = len(body_ids)

    # Đoạn đầu bài
    head_ids = body_ids[
        :CONTENT_BUDGET
    ]

    # Đoạn tiếp ngay sau head
    continuation_ids = body_ids[
        CONTENT_BUDGET:
        2 * CONTENT_BUDGET
    ]

    # Tail chỉ tồn tại khi còn token chưa xuất hiện
    # trong head và continuation.
    if body_length > 2 * CONTENT_BUDGET:

        tail_start = max(
            2 * CONTENT_BUDGET,
            body_length - CONTENT_BUDGET,
        )

        tail_ids = body_ids[
            tail_start:
        ]

    else:
        tail_ids = []

    return {
        "head_ids": head_ids,
        "continuation_ids": continuation_ids,
        "tail_ids": tail_ids,
        "continuation_available": int(
            len(continuation_ids) > 0
        ),
        "tail_available": int(
            len(tail_ids) > 0
        ),
    }


# ============================================================
# 4. Encoding bốn view
# ============================================================

def encode_four_views(example):

    title_sapo = " ".join(
        part
        for part in [
            safe_text(
                example.get(
                    "title_transformer_text",
                    "",
                )
            ),
            safe_text(
                example.get(
                    "sapo_transformer_text",
                    "",
                )
            ),
        ]
        if part
    )

    title_ids = ids_without_special_tokens(
        title_sapo
    )

    body_ids = ids_without_special_tokens(
        example.get(
            "body_transformer_text",
            "",
        )
    )

    body_views = split_body_into_views(
        body_ids
    )

    title_encoding = prepare_fixed_length(
        title_ids
    )

    head_encoding = prepare_fixed_length(
        body_views["head_ids"]
    )

    continuation_encoding = prepare_fixed_length(
        body_views["continuation_ids"]
    )

    tail_encoding = prepare_fixed_length(
        body_views["tail_ids"]
    )

    return {
        "title_input_ids":
            title_encoding["input_ids"],

        "title_attention_mask":
            title_encoding["attention_mask"],

        "head_input_ids":
            head_encoding["input_ids"],

        "head_attention_mask":
            head_encoding["attention_mask"],

        "continuation_input_ids":
            continuation_encoding["input_ids"],

        "continuation_attention_mask":
            continuation_encoding["attention_mask"],

        "tail_input_ids":
            tail_encoding["input_ids"],

        "tail_attention_mask":
            tail_encoding["attention_mask"],

        "continuation_available":
            body_views["continuation_available"],

        "tail_available":
            body_views["tail_available"],
    }


# ============================================================
# 5. Tạo Hugging Face Dataset
# ============================================================

def to_hf_dataset(
    frame,
    description,
):

    source_columns = [
        "title_transformer_text",
        "sapo_transformer_text",
        "body_transformer_text",
        "labels",
    ]

    missing_columns = [
        column
        for column in source_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise KeyError(
            "Thiếu cột khi tạo dataset: "
            + ", ".join(missing_columns)
        )

    dataset = Dataset.from_pandas(
        frame[source_columns],
        preserve_index=False,
    )

    return dataset.map(
        encode_four_views,
        batched=False,
        remove_columns=[
            "title_transformer_text",
            "sapo_transformer_text",
            "body_transformer_text",
        ],
        desc=description,
    )


print("Four-view architecture ready:")
print("1. Title + sapo")
print("2. Head")
print("3. Continuation")
print("4. Tail")

print(
    "Head:",
    f"0:{CONTENT_BUDGET}",
)

print(
    "Continuation:",
    f"{CONTENT_BUDGET}:{2 * CONTENT_BUDGET}",
)

print(
    "Tail:",
    "đoạn cuối không chồng với head/continuation",
)

# === Cell 21 ===
# Cell 14B — Kiểm tra số token head,
# continuation và tail theo từng nguồn

view_diagnostics = df[
    [
        column
        for column in [
            "_row_key",
            "source",
            "title",
            "body_transformer_text",
        ]
        if column in df.columns
    ]
].copy()


# ============================================================
# 1. Đếm token thân bài
# ============================================================

view_diagnostics[
    "body_token_count"
] = (
    view_diagnostics[
        "body_transformer_text"
    ]
    .map(
        ids_without_special_tokens
    )
    .map(len)
)


# ============================================================
# 2. Head
# ============================================================

view_diagnostics[
    "head_token_count"
] = (
    view_diagnostics[
        "body_token_count"
    ]
    .clip(
        upper=CONTENT_BUDGET
    )
)


# ============================================================
# 3. Continuation
# ============================================================

view_diagnostics[
    "continuation_token_count"
] = (
    view_diagnostics[
        "body_token_count"
    ]
    .sub(
        CONTENT_BUDGET
    )
    .clip(
        lower=0,
        upper=CONTENT_BUDGET,
    )
)


view_diagnostics[
    "continuation_available"
] = (
    view_diagnostics[
        "continuation_token_count"
    ]
    .gt(0)
)


# ============================================================
# 4. Tail không chồng token
# ============================================================

view_diagnostics[
    "tail_token_count"
] = (
    view_diagnostics[
        "body_token_count"
    ]
    .sub(
        2 * CONTENT_BUDGET
    )
    .clip(
        lower=0,
        upper=CONTENT_BUDGET,
    )
)


view_diagnostics[
    "tail_available"
] = (
    view_diagnostics[
        "tail_token_count"
    ]
    .gt(0)
)


# ============================================================
# 5. Tổng hợp theo nguồn
# ============================================================

view_source_summary = (
    view_diagnostics
    .groupby(
        "source",
        as_index=False,
    )
    .agg(

        Articles=(
            "body_token_count",
            "size",
        ),

        Mean_body_tokens=(
            "body_token_count",
            "mean",
        ),

        Median_body_tokens=(
            "body_token_count",
            "median",
        ),

        Mean_head_tokens=(
            "head_token_count",
            "mean",
        ),

        Mean_continuation_tokens=(
            "continuation_token_count",
            "mean",
        ),

        Mean_tail_tokens=(
            "tail_token_count",
            "mean",
        ),

        Continuation_articles=(
            "continuation_available",
            "sum",
        ),

        Continuation_rate=(
            "continuation_available",
            "mean",
        ),

        Tail_articles=(
            "tail_available",
            "sum",
        ),

        Tail_rate=(
            "tail_available",
            "mean",
        ),
    )
)


display(
    view_source_summary
    .style
    .format({

        "Articles":
            "{:,.0f}",

        "Mean_body_tokens":
            "{:,.1f}",

        "Median_body_tokens":
            "{:,.0f}",

        "Mean_head_tokens":
            "{:,.1f}",

        "Mean_continuation_tokens":
            "{:,.1f}",

        "Mean_tail_tokens":
            "{:,.1f}",

        "Continuation_articles":
            "{:,.0f}",

        "Continuation_rate":
            "{:.2%}",

        "Tail_articles":
            "{:,.0f}",

        "Tail_rate":
            "{:.2%}",
    })
)


view_diagnostics.to_csv(
    OUTPUT_DIR
    / "four_view_token_diagnostics.csv",
    index=False,
)


view_source_summary.to_csv(
    OUTPUT_DIR
    / "four_view_token_summary_by_source.csv",
    index=False,
)


print(
    "Bài có continuation:",
    int(
        view_diagnostics[
            "continuation_available"
        ].sum()
    ),
)

print(
    "Bài có tail độc lập:",
    int(
        view_diagnostics[
            "tail_available"
        ].sum()
    ),
)


# === Cell 22 ===
# Cell 15 — XLM-RoBERTa four-view
# title + sapo / head / continuation / tail

import gc

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)


# ============================================================
# 1. Shared XLM-RoBERTa four-view
# ============================================================

class SharedFourViewClassifier(nn.Module):

    """
    Dùng chung một XLM-RoBERTa encoder cho bốn view:

    1. title + sapo
    2. head
    3. continuation
    4. tail

    Continuation và tail sẽ tự nhận trọng số 0
    nếu bài không đủ dài.
    """

    main_input_name = (
        "title_input_ids"
    )

    def __init__(
        self,
        classifier,
        class_weights,
        default_weights,
    ):

        super().__init__()

        self.base_classifier = (
            classifier
        )

        self.config = (
            classifier.config
        )

        self.register_buffer(
            "class_weights",
            class_weights
            .clone()
            .detach(),
        )

        self.register_buffer(
            "default_view_weights",

            torch.tensor(
                [
                    default_weights[
                        "title_sapo"
                    ],

                    default_weights[
                        "head"
                    ],

                    default_weights[
                        "continuation"
                    ],

                    default_weights[
                        "tail"
                    ],
                ],

                dtype=torch.float32,
            ),
        )


    def _normalized_weights(
        self,
        continuation_available,
        tail_available,
        dtype,
        device,
    ):

        batch_size = (
            continuation_available
            .shape[0]
        )

        weights = (
            self.default_view_weights
            .to(
                device=device,
                dtype=dtype,
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
            .clone()
        )

        # Không có continuation:
        # trọng số continuation = 0.

        weights[:, 2] *= (
            continuation_available
            .to(
                device=device,
                dtype=dtype,
            )
        )

        # Không có tail độc lập:
        # trọng số tail = 0.

        weights[:, 3] *= (
            tail_available
            .to(
                device=device,
                dtype=dtype,
            )
        )

        # Phân bổ lại trọng số
        # cho những view thực sự tồn tại.

        return (
            weights
            / weights
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(
                1e-12
            )
        )


    def forward(
        self,

        title_input_ids=None,
        title_attention_mask=None,

        head_input_ids=None,
        head_attention_mask=None,

        continuation_input_ids=None,
        continuation_attention_mask=None,

        tail_input_ids=None,
        tail_attention_mask=None,

        continuation_available=None,
        tail_available=None,

        labels=None,
    ):

        # ----------------------------------------------------
        # 1. Title + sapo
        # ----------------------------------------------------

        title_logits = (
            self.base_classifier(
                input_ids=
                    title_input_ids,

                attention_mask=
                    title_attention_mask,
            )
            .logits
        )

        # ----------------------------------------------------
        # 2. Head
        # ----------------------------------------------------

        head_logits = (
            self.base_classifier(
                input_ids=
                    head_input_ids,

                attention_mask=
                    head_attention_mask,
            )
            .logits
        )

        # ----------------------------------------------------
        # 3. Continuation
        # ----------------------------------------------------

        continuation_logits = (
            self.base_classifier(
                input_ids=
                    continuation_input_ids,

                attention_mask=
                    continuation_attention_mask,
            )
            .logits
        )

        # ----------------------------------------------------
        # 4. Tail
        # ----------------------------------------------------

        tail_logits = (
            self.base_classifier(
                input_ids=
                    tail_input_ids,

                attention_mask=
                    tail_attention_mask,
            )
            .logits
        )

        # ----------------------------------------------------
        # 5. Ghép logits
        # ----------------------------------------------------

        stacked_logits = torch.stack(
            [
                title_logits,
                head_logits,
                continuation_logits,
                tail_logits,
            ],

            dim=1,
        )

        weights = (
            self._normalized_weights(
                continuation_available=
                    continuation_available,

                tail_available=
                    tail_available,

                dtype=
                    stacked_logits.dtype,

                device=
                    stacked_logits.device,
            )
        )

        fused_logits = (
            stacked_logits
            * weights.unsqueeze(-1)
        ).sum(
            dim=1
        )

        # ----------------------------------------------------
        # 6. Weighted loss
        # ----------------------------------------------------

        loss = None

        if labels is not None:

            per_view_loss = torch.stack(
                [
                    F.cross_entropy(
                        logits,

                        labels,

                        weight=(
                            self.class_weights
                            .to(
                                logits.device
                            )
                        ),

                        reduction="none",
                    )

                    for logits in [
                        title_logits,
                        head_logits,
                        continuation_logits,
                        tail_logits,
                    ]
                ],

                dim=1,
            )

            loss = (
                per_view_loss
                * weights
            ).sum(
                dim=1
            ).mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=fused_logits,
        )


# ============================================================
# 2. Dự đoán xác suất riêng từng view
# ============================================================

@torch.no_grad()
def predict_view_probabilities(
    model,
    dataset,
    batch_size=None,
):

    model.eval()

    device = next(
        model.parameters()
    ).device

    loader = DataLoader(
        dataset,

        batch_size=(
            batch_size
            or EVAL_BATCH_SIZE
        ),

        shuffle=False,

        collate_fn=
            default_data_collator,
    )

    title_probs = []

    head_probs = []

    continuation_probs = []

    tail_probs = []

    continuation_flags = []

    tail_flags = []

    true_labels = []


    for batch in loader:

        labels = batch.pop(
            "labels"
        )

        batch = {
            key:
                value.to(device)

            for key, value
            in batch.items()
        }

        # ----------------------------------------------------
        # Xác suất title + sapo
        # ----------------------------------------------------

        title_logits = (
            model.base_classifier(
                input_ids=
                    batch[
                        "title_input_ids"
                    ],

                attention_mask=
                    batch[
                        "title_attention_mask"
                    ],
            )
            .logits
        )

        # ----------------------------------------------------
        # Xác suất head
        # ----------------------------------------------------

        head_logits = (
            model.base_classifier(
                input_ids=
                    batch[
                        "head_input_ids"
                    ],

                attention_mask=
                    batch[
                        "head_attention_mask"
                    ],
            )
            .logits
        )

        # ----------------------------------------------------
        # Xác suất continuation
        # ----------------------------------------------------

        continuation_logits = (
            model.base_classifier(
                input_ids=
                    batch[
                        "continuation_input_ids"
                    ],

                attention_mask=
                    batch[
                        "continuation_attention_mask"
                    ],
            )
            .logits
        )

        # ----------------------------------------------------
        # Xác suất tail
        # ----------------------------------------------------

        tail_logits = (
            model.base_classifier(
                input_ids=
                    batch[
                        "tail_input_ids"
                    ],

                attention_mask=
                    batch[
                        "tail_attention_mask"
                    ],
            )
            .logits
        )

        title_probs.append(
            torch.softmax(
                title_logits,
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        head_probs.append(
            torch.softmax(
                head_logits,
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        continuation_probs.append(
            torch.softmax(
                continuation_logits,
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        tail_probs.append(
            torch.softmax(
                tail_logits,
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        continuation_flags.append(
            batch[
                "continuation_available"
            ]
            .cpu()
            .numpy()
        )

        tail_flags.append(
            batch[
                "tail_available"
            ]
            .cpu()
            .numpy()
        )

        true_labels.append(
            labels.numpy()
        )


    # Shape:
    #
    # n_articles × 4 views × n_classes

    view_probabilities = np.stack(
        [
            np.concatenate(
                title_probs
            ),

            np.concatenate(
                head_probs
            ),

            np.concatenate(
                continuation_probs
            ),

            np.concatenate(
                tail_probs
            ),
        ],

        axis=1,
    )


    continuation_available = (
        np.concatenate(
            continuation_flags
        )
        .astype(float)
    )

    tail_available = (
        np.concatenate(
            tail_flags
        )
        .astype(float)
    )

    true_labels = (
        np.concatenate(
            true_labels
        )
    )

    availability = {
        "continuation":
            continuation_available,

        "tail":
            tail_available,
    }

    return (
        view_probabilities,
        availability,
        true_labels,
    )


# ============================================================
# 3. Ghép xác suất bốn view
# ============================================================

def combine_view_probabilities(
    view_probabilities,
    availability,
    weights,
):

    weight_array = np.array(
        [
            weights[
                "title_sapo"
            ],

            weights[
                "head"
            ],

            weights[
                "continuation"
            ],

            weights[
                "tail"
            ],
        ],

        dtype=float,
    )

    sample_weights = np.tile(
        weight_array,

        (
            len(
                view_probabilities
            ),
            1,
        ),
    )

    # Continuation không tồn tại:
    # trọng số continuation = 0.

    sample_weights[:, 2] *= (
        availability[
            "continuation"
        ]
    )

    # Tail không tồn tại:
    # trọng số tail = 0.

    sample_weights[:, 3] *= (
        availability[
            "tail"
        ]
    )

    # Chuẩn hóa tổng trọng số bằng 1.

    sample_weights /= np.clip(
        sample_weights.sum(
            axis=1,
            keepdims=True,
        ),

        1e-12,

        None,
    )

    return (
        view_probabilities
        * sample_weights[
            :,
            :,
            None
        ]
    ).sum(
        axis=1
    )


# ============================================================
# 4. Bootstrap confidence interval
# ============================================================

def bootstrap_macro_f1(
    y_true,
    y_pred,
    n_boot=BOOTSTRAP_ITERATIONS,
    seed=SEED,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    rng = np.random.default_rng(
        seed
    )

    scores = []

    n_classes = len(
        np.unique(
            y_true
        )
    )

    for _ in range(
        n_boot
    ):

        indices = rng.integers(
            0,
            len(
                y_true
            ),
            len(
                y_true
            ),
        )

        if (
            len(
                np.unique(
                    y_true[
                        indices
                    ]
                )
            )
            < n_classes
        ):

            continue

        scores.append(
            f1_score(
                y_true[
                    indices
                ],

                y_pred[
                    indices
                ],

                average="macro",
            )
        )

    if not scores:

        return np.array(
            [
                np.nan,
                np.nan,
                np.nan,
            ]
        )

    return np.percentile(
        scores,

        [
            2.5,
            50,
            97.5,
        ],
    )


print(
    "SharedFourViewClassifier ready."
)

print(
    "Views:",
    [
        "title_sapo",
        "head",
        "continuation",
        "tail",
    ],
)


# === Cell 23 ===
# Cell 16 — Huấn luyện XLM-RoBERTa four-view
# cho từng task

def run_one_transformer_task(
    task_name
):

    print(
        "\n" + "#" * 88
    )

    print(
        "XLM-R FOUR-VIEW TASK:",
        task_name,
    )

    print(
        "#" * 88
    )


    # ========================================================
    # 1. Lấy dữ liệu task
    # ========================================================

    task_data = get_task_frame(
        task_name,
        verbose=True,
    )

    if task_name not in split_cache:

        raise KeyError(
            f"Thiếu split_cache cho {task_name}. "
            "Không được tự tạo split khác baseline."
        )

    train_df, val_df, test_df = [
        part.copy()

        for part
        in split_cache[
            task_name
        ]
    ]


    # ========================================================
    # 2. Kiểm tra fairness với baseline
    # ========================================================

    reference_baselines = (
        baseline_results.loc[
            baseline_results[
                "task"
            ].eq(
                task_name
            )
        ]
    )

    if reference_baselines.empty:

        raise ValueError(
            "Không có baseline để đối chiếu "
            f"cho task {task_name}."
        )

    actual_fingerprints = {

        "train_fingerprint":
            article_set_fingerprint(
                train_df
            ),

        "validation_fingerprint":
            article_set_fingerprint(
                val_df
            ),

        "test_fingerprint":
            article_set_fingerprint(
                test_df
            ),
    }

    for column, actual in (
        actual_fingerprints.items()
    ):

        expected = set(
            reference_baselines[
                column
            ].astype(str)
        )

        if expected != {
            actual
        }:

            raise AssertionError(
                f"{task_name}: "
                "four-view và baseline khác "
                f"article set ở {column}."
            )


    expected_sizes = {

        "n_train":
            len(
                train_df
            ),

        "n_validation":
            len(
                val_df
            ),

        "n_test":
            len(
                test_df
            ),
    }

    for column, expected_size in (
        expected_sizes.items()
    ):

        actual_sizes = set(
            reference_baselines[
                column
            ].astype(int)
        )

        if actual_sizes != {
            expected_size
        }:

            raise AssertionError(
                f"{task_name}: "
                "four-view và baseline khác "
                f"kích thước ở {column}."
            )


    # ========================================================
    # 3. Mã hóa nhãn
    # ========================================================

    label_encoder = LabelEncoder()

    label_encoder.fit(
        train_df[
            "label"
        ].astype(str)
    )

    for part in [
        train_df,
        val_df,
        test_df,
    ]:

        part[
            "labels"
        ] = (
            label_encoder
            .transform(
                part[
                    "label"
                ].astype(str)
            )
        )

    print(
        "Label mapping:",

        {
            index:
                label

            for index, label
            in enumerate(
                label_encoder.classes_
            )
        },
    )


    # ========================================================
    # 4. Dataset bốn view
    # ========================================================

    train_ds = to_hf_dataset(
        train_df,
        f"{task_name}: tokenizing train",
    )

    val_ds = to_hf_dataset(
        val_df,
        f"{task_name}: tokenizing validation",
    )

    test_ds = to_hf_dataset(
        test_df,
        f"{task_name}: tokenizing test",
    )


    # ========================================================
    # 5. Tỷ lệ bài có continuation và tail
    # ========================================================

    continuation_rates = {

        "train":
            float(
                np.mean(
                    train_ds[
                        "continuation_available"
                    ]
                )
            ),

        "validation":
            float(
                np.mean(
                    val_ds[
                        "continuation_available"
                    ]
                )
            ),

        "test":
            float(
                np.mean(
                    test_ds[
                        "continuation_available"
                    ]
                )
            ),
    }

    tail_rates = {

        "train":
            float(
                np.mean(
                    train_ds[
                        "tail_available"
                    ]
                )
            ),

        "validation":
            float(
                np.mean(
                    val_ds[
                        "tail_available"
                    ]
                )
            ),

        "test":
            float(
                np.mean(
                    test_ds[
                        "tail_available"
                    ]
                )
            ),
    }

    print(
        "Tỷ lệ bài có continuation:",
        continuation_rates,
    )

    print(
        "Tỷ lệ bài có tail độc lập:",
        tail_rates,
    )


    # ========================================================
    # 6. Khởi tạo XLM-RoBERTa
    # ========================================================

    num_labels = len(
        label_encoder.classes_
    )

    id2label = {

        index:
            str(label)

        for index, label
        in enumerate(
            label_encoder.classes_
        )
    }

    label2id = {

        label:
            index

        for index, label
        in id2label.items()
    }

    base_classifier = (
        AutoModelForSequenceClassification
        .from_pretrained(

            MODEL_NAME,

            num_labels=
                num_labels,

            id2label=
                id2label,

            label2id=
                label2id,
        )
    )


    # ========================================================
    # 7. Class weights
    # ========================================================

    train_counts = np.bincount(
        train_df[
            "labels"
        ],

        minlength=
            num_labels,
    )

    class_weights = (
        len(
            train_df
        )
        / (
            num_labels
            * np.maximum(
                train_counts,
                1,
            )
        )
    )

    class_weights = torch.tensor(
        class_weights,

        dtype=
            torch.float32,
    )

    print(
        "Class weights:",
        class_weights.tolist(),
    )


    # ========================================================
    # 8. Four-view classifier
    # ========================================================

    model = SharedFourViewClassifier(

        classifier=
            base_classifier,

        class_weights=
            class_weights,

        default_weights=
            DEFAULT_VIEW_WEIGHTS,
    )


    def compute_transformer_metrics(
        eval_prediction
    ):

        logits, labels = (
            eval_prediction
        )

        predictions = np.argmax(
            logits,

            axis=
                -1,
        )

        return metric_row(
            labels,
            predictions,
        )


    # ========================================================
    # 9. Thư mục kết quả task
    # ========================================================

    run_dir = (

        OUTPUT_DIR

        / "transformer_tasks"

        / (
            f"{task_name}_"
            f"{MODEL_NAME.split('/')[-1]}_"
            "four_view"
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 10. Training steps
    # ========================================================

    effective_steps_per_epoch = (
        math.ceil(
            len(
                train_ds
            )
            / (
                TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION
            )
        )
    )

    total_training_steps = (
        effective_steps_per_epoch
        * NUM_EPOCHS
    )

    warmup_steps = round(
        total_training_steps
        * WARMUP_RATIO
    )

    print(
        "Training steps:",
        total_training_steps,
    )

    print(
        "Warmup steps:",
        warmup_steps,
    )


    # ========================================================
    # 11. TrainingArguments
    # ========================================================

    training_args = TrainingArguments(

        output_dir=
            str(
                run_dir
                / "checkpoints"
            ),

        learning_rate=
            LEARNING_RATE,

        per_device_train_batch_size=
            TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=
            EVAL_BATCH_SIZE,

        gradient_accumulation_steps=
            GRADIENT_ACCUMULATION,

        num_train_epochs=
            NUM_EPOCHS,

        weight_decay=
            0.01,

        warmup_steps=
            warmup_steps,

        eval_strategy=
            "epoch",

        save_strategy=
            "epoch",

        logging_strategy=
            "steps",

        logging_steps=
            20,

        load_best_model_at_end=
            True,

        metric_for_best_model=
            "f1_macro",

        greater_is_better=
            True,

        save_total_limit=
            1,

        fp16=
            torch.cuda.is_available(),

        report_to=
            "none",

        seed=
            SEED,

        data_seed=
            SEED,

        remove_unused_columns=
            True,
    )


    # ========================================================
    # 12. Trainer
    # ========================================================

    trainer = Trainer(

        model=
            model,

        args=
            training_args,

        train_dataset=
            train_ds,

        eval_dataset=
            val_ds,

        processing_class=
            tokenizer,

        data_collator=
            default_data_collator,

        compute_metrics=
            compute_transformer_metrics,

        callbacks=[

            EarlyStoppingCallback(

                early_stopping_patience=
                    EARLY_STOPPING_PATIENCE,
            )
        ],
    )


    # ========================================================
    # 13. Fine-tune
    # ========================================================

    train_output = trainer.train()


    # ========================================================
    # 14. Lưu model
    # ========================================================

    best_model_dir = (
        run_dir
        / "best_model"
    )

    trainer.model.base_classifier.save_pretrained(
        str(
            best_model_dir
        )
    )

    tokenizer.save_pretrained(
        str(
            best_model_dir
        )
    )


    # ========================================================
    # 15. Dự đoán validation
    # ========================================================

    (
        val_view_probs,

        val_availability,

        val_true_id,

    ) = predict_view_probabilities(

        trainer.model,

        val_ds,
    )


    # ========================================================
    # 16. Chọn trọng số trên validation
    # ========================================================

    weight_rows = []

    for candidate_weights in (
        VIEW_WEIGHT_GRID
    ):

        candidate_probs = (
            combine_view_probabilities(

                val_view_probs,

                val_availability,

                candidate_weights,
            )
        )

        candidate_pred_id = (
            candidate_probs.argmax(
                axis=1
            )
        )

        scores = metric_row(

            val_true_id,

            candidate_pred_id,
        )

        weight_rows.append(
            {
                **candidate_weights,

                **scores,
            }
        )

    view_weight_results = (

        pd.DataFrame(
            weight_rows
        )

        .sort_values(

            [
                "f1_macro",

                "balanced_accuracy",

                "accuracy",
            ],

            ascending=
                False,
        )

        .reset_index(
            drop=True
        )
    )

    view_weight_results.to_csv(

        run_dir

        / "validation_view_weight_search.csv",

        index=
            False,
    )

    best_view_weights = {

        key:
            float(
                view_weight_results.loc[
                    0,
                    key,
                ]
            )

        for key in [

            "title_sapo",

            "head",

            "continuation",

            "tail",
        ]
    }

    print(
        "Trọng số khóa sau validation:",
        best_view_weights,
    )


    # ========================================================
    # 17. Dự đoán test
    # ========================================================

    (
        test_view_probs,

        test_availability,

        test_true_id,

    ) = predict_view_probabilities(

        trainer.model,

        test_ds,
    )

    test_probs = (
        combine_view_probabilities(

            test_view_probs,

            test_availability,

            best_view_weights,
        )
    )

    test_pred_id = (
        test_probs.argmax(
            axis=1
        )
    )

    test_pred = (
        label_encoder
        .inverse_transform(
            test_pred_id
        )
    )

    test_true = (
        label_encoder
        .inverse_transform(
            test_true_id
        )
    )


    # ========================================================
    # 18. Metrics
    # ========================================================

    transformer_metrics = metric_row(

        test_true,

        test_pred,
    )

    report = classification_report(

        test_true,

        test_pred,

        digits=
            4,

        zero_division=
            0,

        output_dict=
            True,
    )

    report_frame = (
        pd.DataFrame(
            report
        ).T
    )

    report_frame.to_csv(

        run_dir

        / "classification_report.csv"
    )


    # ========================================================
    # 19. Confusion matrix
    # ========================================================

    labels_plot = list(
        label_encoder.classes_
    )

    confusion = confusion_matrix(

        test_true,

        test_pred,

        labels=
            labels_plot,
    )

    plt.figure(

        figsize=(
            max(
                7,
                len(
                    labels_plot
                )
                * 1.2,
            ),

            5.5,
        )
    )

    sns.heatmap(

        confusion,

        annot=
            True,

        fmt=
            "d",

        cmap=
            "Oranges",

        xticklabels=
            labels_plot,

        yticklabels=
            labels_plot,
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        f"{MODEL_NAME} four-view — {task_name}"
    )

    plt.tight_layout()

    plt.savefig(

        run_dir

        / "confusion_matrix.png",

        dpi=
            180,

        bbox_inches=
            "tight",
    )

    plt.close()


    # ========================================================
    # 20. Lưu dự đoán và lỗi
    # ========================================================

    confidence = (
        test_probs.max(
            axis=1
        )
    )

    error_columns = [

        column

        for column in [

            "_row_key",

            id_col,

            "source",

            "title",

            "description",

            "evidence_text",

            "annotator_note",

            "_source_file",
        ]

        if column

        and column
        in test_df.columns
    ]

    errors = (
        test_df[
            error_columns
        ].copy()
    )

    errors[
        "true_label"
    ] = test_true

    errors[
        "predicted_label"
    ] = test_pred

    errors[
        "confidence"
    ] = confidence

    errors[
        "continuation_available"
    ] = (
        test_availability[
            "continuation"
        ]
        .astype(int)
    )

    errors[
        "tail_available"
    ] = (
        test_availability[
            "tail"
        ]
        .astype(int)
    )

    errors[
        "is_error"
    ] = (

        errors[
            "true_label"
        ]

# COLAB COMMAND:         != errors[
            "predicted_label"
        ]
    )

    errors = errors.sort_values(

        [
            "is_error",

            "confidence",
        ],

        ascending=[
            False,

            False,
        ],
    )

    errors.to_excel(

        run_dir

        / "test_predictions_and_errors.xlsx",

        index=
            False,
    )


    # ========================================================
    # 21. So sánh baseline
    # ========================================================

    base_for_task = (

        baseline_results

        .loc[
            baseline_results[
                "task"
            ].eq(
                task_name
            )
        ]

        .sort_values(

            [
                "validation_f1_macro",

                "validation_balanced_accuracy",
            ],

            ascending=
                False,
        )
    )

    comparison = pd.concat(

        [

            base_for_task[
                [
                    "model",

                    "accuracy",

                    "balanced_accuracy",

                    "f1_macro",

                    "f1_weighted",
                ]
            ],

            pd.DataFrame(
                [
                    {
                        "model":
                            (
                                f"{MODEL_NAME}"
                                "_four_view"
                            ),

                        **transformer_metrics,
                    }
                ]
            ),
        ],

        ignore_index=
            True,

    ).sort_values(

        "f1_macro",

        ascending=
            False,
    )

    comparison.to_csv(

        run_dir

        / "model_comparison.csv",

        index=
            False,
    )


    # ========================================================
    # 22. Bootstrap CI
    # ========================================================

    ci = bootstrap_macro_f1(

        test_true,

        test_pred,
    )


    # ========================================================
    # 23. Cấu hình model four-view
    # ========================================================

    four_view_config = {

        "task":
            task_name,

        "architecture":
            "four_view",

        "views": [

            "title_sapo",

            "head",

            "continuation",

            "tail",
        ],

        "max_length":
            MAX_LENGTH,

        "content_budget":
            CONTENT_BUDGET,

        "default_view_weights":
            DEFAULT_VIEW_WEIGHTS,

        "weight_grid":
            VIEW_WEIGHT_GRID,

        "selected_view_weights":
            best_view_weights,

        "selection_metric":
            "validation_f1_macro",

        "labels":
            list(
                map(
                    str,

                    label_encoder.classes_,
                )
            ),

        "no_body_view_overlap":
            True,
    }

    with open(

        best_model_dir

        / "four_view_config.json",

        "w",

        encoding=
            "utf-8",

    ) as file:

        json.dump(

            four_view_config,

            file,

            ensure_ascii=
                False,

            indent=
                2,
        )


    # ========================================================
    # 24. Resumen completo
    # ========================================================

    summary = {

        "task":
            task_name,

        "status":
            "completed",

        "architecture":
            "four_view",

        "experiment_mode":
            EXPERIMENT_MODE,

        "train_sources":
            sorted(
                train_df[
                    "source"
                ].unique().tolist()
            ),

        "validation_sources":
            sorted(
                val_df[
                    "source"
                ].unique().tolist()
            ),

        "test_sources":
            sorted(
                test_df[
                    "source"
                ].unique().tolist()
            ),

        "split_fingerprints":
            actual_fingerprints,

        "model_name":
            MODEL_NAME,

        "input_strategy":
            (
                "shared_four_view_"
                "title_sapo_head_"
                "continuation_tail"
            ),

        "labels":
            list(
                map(
                    str,

                    label_encoder.classes_,
                )
            ),

        "split_sizes": {

            "train":
                len(
                    train_df
                ),

            "validation":
                len(
                    val_df
                ),

            "test":
                len(
                    test_df
                ),
        },

        "transformer_metrics": {

            key:
                float(value)

            for key, value
            in transformer_metrics.items()
        },

        "macro_f1_bootstrap_95ci": [

            float(
                ci[0]
            ),

            float(
                ci[2]
            ),
        ],

        "macro_f1_bootstrap_median":
            float(
                ci[1]
            ),

        "selected_view_weights":
            best_view_weights,

        "weight_selection_set":
            "validation",

        "continuation_available_rates":
            continuation_rates,

        "tail_available_rates":
            tail_rates,

        "seed":
            SEED,

        "max_length_per_view":
            MAX_LENGTH,

        "maximum_total_tokens_across_views":
            MAX_LENGTH * 4,

        "word_segmented": False,

        "best_checkpoint":
            trainer.state.best_model_checkpoint,

        "training_loss":
            float(
                train_output.training_loss
            ),
    }

    with open(

        run_dir

        / "run_summary.json",

        "w",

        encoding=
            "utf-8",

    ) as file:

        json.dump(

            summary,

            file,

            ensure_ascii=
                False,

            indent=
                2,
        )


    # ========================================================
    # 25. Dòng tổng hợp cho toàn bộ task
    # ========================================================

    summary_row = {

        "task":
            task_name,

        "status":
            "completed",

        "architecture":
            "four_view",

        "n_train":
            len(
                train_df
            ),

        "n_validation":
            len(
                val_df
            ),

        "n_test":
            len(
                test_df
            ),

        "n_classes":
            num_labels,

        "experiment_mode":
            EXPERIMENT_MODE,

        **actual_fingerprints,

        **{

            key:
                float(value)

            for key, value
            in transformer_metrics.items()
        },

        "macro_f1_ci_low":
            float(
                ci[0]
            ),

        "macro_f1_ci_high":
            float(
                ci[2]
            ),

        "title_sapo_weight":
            best_view_weights[
                "title_sapo"
            ],

        "head_weight":
            best_view_weights[
                "head"
            ],

        "continuation_weight":
            best_view_weights[
                "continuation"
            ],

        "tail_weight":
            best_view_weights[
                "tail"
            ],

        "continuation_available_test_rate":
            continuation_rates[
                "test"
            ],

        "tail_available_test_rate":
            tail_rates[
                "test"
            ],

        "run_dir":
            str(
                run_dir
            ),
    }


    print(

        "Test metrics:",

        {

            key:
                round(
                    value,
                    4,
                )

            for key, value
            in transformer_metrics.items()
        },
    )

    print(
        "Macro-F1 bootstrap 95% CI: "
        f"[{ci[0]:.4f}, {ci[2]:.4f}]"
    )

    return summary_row


# === Cell 24 ===
# Cell 16B — Phục hồi các task đã train xong nhưng lỗi ở bước ghi metadata
import json
from pathlib import Path

RECOVERY_VIEW_KEYS = [
    "title_sapo",
    "head",
    "continuation",
    "tail",
]

transformer_root = OUTPUT_DIR / "transformer_tasks"
transformer_summary_path = OUTPUT_DIR / "all_transformer_task_metrics.csv"

recovered_rows = []
recovery_failures = []

for task_name in TRANSFORMER_TASKS_TO_RUN:
    print("\n" + "=" * 80)
    print("Đang phục hồi:", task_name)

    model_slug = MODEL_NAME.split("/")[-1]

    # Thư mục chuẩn của kiến trúc bốn view
    run_dir = (
        transformer_root
        / f"{task_name}_{model_slug}_four_view"
    )

    # Fallback nếu tên thư mục trong Cell 16 hơi khác
    if not run_dir.exists():
        candidates = [
            path
            for path in transformer_root.glob(
                f"{task_name}_{model_slug}*"
            )
            if (
                path.is_dir()
                and (path / "test_predictions_and_errors.xlsx").exists()
            )
        ]

        candidates = sorted(
            candidates,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if candidates:
            run_dir = candidates[0]

    prediction_path = (
        run_dir / "test_predictions_and_errors.xlsx"
    )
    weight_search_path = (
        run_dir / "validation_view_weight_search.csv"
    )
    best_model_dir = run_dir / "best_model"

    required_paths = [
        prediction_path,
        weight_search_path,
        best_model_dir,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        failure = {
            "task": task_name,
            "reason": "Thiếu artifact",
            "missing": missing_paths,
        }
        recovery_failures.append(failure)

        print("Không thể phục hồi vì thiếu:")
        for path in missing_paths:
            print(" -", path)

        continue

    # Lấy đúng split đã dùng chung với baseline
    if task_name not in split_cache:
        recovery_failures.append({
            "task": task_name,
            "reason": "Không có task trong split_cache",
        })
        print("Không có split_cache:", task_name)
        continue

    train_df, val_df, test_df = [
        frame.copy()
        for frame in split_cache[task_name]
    ]

    actual_fingerprints = {
        "train_fingerprint":
            article_set_fingerprint(train_df),
        "validation_fingerprint":
            article_set_fingerprint(val_df),
        "test_fingerprint":
            article_set_fingerprint(test_df),
    }

    # Kiểm tra vẫn đúng tập bài như baseline
    task_baselines = baseline_results.loc[
        baseline_results["task"].eq(task_name)
    ]

    if task_baselines.empty:
        recovery_failures.append({
            "task": task_name,
            "reason": "Không tìm thấy baseline",
        })
        print("Không tìm thấy baseline:", task_name)
        continue

    fingerprint_valid = True

    for column, actual_value in actual_fingerprints.items():
        baseline_values = set(
            task_baselines[column].astype(str)
        )

        if baseline_values != {actual_value}:
            fingerprint_valid = False
            recovery_failures.append({
                "task": task_name,
                "reason": f"Không khớp {column}",
            })
            print("Không khớp fingerprint:", column)
            break

    if not fingerprint_valid:
        continue

    predictions = pd.read_excel(prediction_path)

    required_prediction_columns = {
        "true_label",
        "predicted_label",
    }

    if not required_prediction_columns.issubset(
        predictions.columns
    ):
        recovery_failures.append({
            "task": task_name,
            "reason": (
                "File prediction thiếu true_label/"
                "predicted_label"
            ),
        })
        print("File prediction thiếu cột nhãn.")
        continue

    if len(predictions) != len(test_df):
        recovery_failures.append({
            "task": task_name,
            "reason": (
                f"Số prediction {len(predictions)} "
                f"khác test {len(test_df)}"
            ),
        })
        print(
            "Số prediction không khớp test:",
            len(predictions),
            "!=",
            len(test_df),
        )
        continue

    y_true = predictions["true_label"].astype(str).to_numpy()
    y_pred = predictions["predicted_label"].astype(str).to_numpy()

    transformer_metrics = metric_row(y_true, y_pred)
    ci_low, ci_median, ci_high = bootstrap_macro_f1(
        y_true,
        y_pred,
    )

    # Khôi phục trọng số đã khóa trên validation
    weight_search = pd.read_csv(weight_search_path)

    weight_search = weight_search.sort_values(
        [
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected_view_weights = {
        key: float(weight_search.loc[0, key])
        for key in RECOVERY_VIEW_KEYS
    }

    # Tỷ lệ view khả dụng nếu đã được ghi vào prediction
    test_availability = {}

    for column in [
        "continuation_available",
        "tail_available",
    ]:
        if column in predictions.columns:
            test_availability[column] = float(
                pd.to_numeric(
                    predictions[column],
                    errors="coerce",
                ).fillna(0).mean()
            )

    label_values = sorted(
        train_df["label"].astype(str).unique().tolist()
    )

    summary = {
        "task": task_name,
        "status": "completed",
        "recovered_without_retraining": True,
        "experiment_mode": EXPERIMENT_MODE,
        "view_architecture": VIEW_ARCHITECTURE,
        "views": list(VIEW_NAMES),
        "model_name": MODEL_NAME,
        "input_strategy": (
            "shared_four_view_"
            "title_sapo_head_continuation_tail"
        ),
        "train_sources": sorted(
            train_df["source"].astype(str).unique().tolist()
        ),
        "validation_sources": sorted(
            val_df["source"].astype(str).unique().tolist()
        ),
        "test_sources": sorted(
            test_df["source"].astype(str).unique().tolist()
        ),
        "split_fingerprints": actual_fingerprints,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "labels": label_values,
        "transformer_metrics": {
            key: float(value)
            for key, value in transformer_metrics.items()
        },
        "macro_f1_bootstrap_95ci": [
            float(ci_low),
            float(ci_high),
        ],
        "macro_f1_bootstrap_median": float(ci_median),
        "selected_view_weights": selected_view_weights,
        "weight_selection_set": "validation",
        "test_view_availability": test_availability,
        "seed": SEED,
        "max_length_per_view": MAX_LENGTH,
        "maximum_total_tokens_across_views":
            MAX_LENGTH * 4,
        "word_segmented":
            bool(uses_external_word_segmentation and USE_VNCORENLP),
        "best_model_dir": str(best_model_dir),
    }

    with open(
        run_dir / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    four_view_config = {
        "task": task_name,
        "view_architecture": VIEW_ARCHITECTURE,
        "views": list(VIEW_NAMES),
        "max_length_per_view": MAX_LENGTH,
        "default_view_weights": DEFAULT_VIEW_WEIGHTS,
        "weight_grid": VIEW_WEIGHT_GRID,
        "selected_view_weights": selected_view_weights,
        "selection_metric": "validation_f1_macro",
        "labels": label_values,
    }

    with open(
        best_model_dir / "four_view_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            four_view_config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    recovered_row = {
        "task": task_name,
        "status": "completed",
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "n_classes": len(label_values),
        "experiment_mode": EXPERIMENT_MODE,
        "view_architecture": VIEW_ARCHITECTURE,
        **actual_fingerprints,
        **{
            key: float(value)
            for key, value in transformer_metrics.items()
        },
        "macro_f1_ci_low": float(ci_low),
        "macro_f1_ci_high": float(ci_high),
        "title_sapo_weight":
            selected_view_weights["title_sapo"],
        "head_weight":
            selected_view_weights["head"],
        "continuation_weight":
            selected_view_weights["continuation"],
        "tail_weight":
            selected_view_weights["tail"],
        "run_dir": str(run_dir),
        "recovered_without_retraining": True,
    }

    recovered_rows.append(recovered_row)

    print("Phục hồi thành công.")
    print("Trọng số:", selected_view_weights)
    print(
        "Macro-F1:",
        round(transformer_metrics["f1_macro"], 4),
    )

transformer_results = pd.DataFrame(recovered_rows)

if not transformer_results.empty:
    transformer_results = transformer_results.sort_values(
        "f1_macro",
        ascending=False,
    ).reset_index(drop=True)

    transformer_results.to_csv(
        transformer_summary_path,
        index=False,
    )

print("\nSố task phục hồi:", len(transformer_results))
print("Số task chưa phục hồi:", len(recovery_failures))

display(transformer_results)

if recovery_failures:
    display(pd.DataFrame(recovery_failures))


# === Cell 25 ===
# Cell 17 — Chạy tuần tự toàn bộ task và tiếp tục nếu một task lỗi
transformer_summary_path = OUTPUT_DIR / "all_transformer_task_metrics.csv"
transformer_rows = []
transformer_failures = []

# Nạp kết quả cũ nếu notebook từng bị ngắt và được chạy lại.
if SKIP_COMPLETED_TASKS and transformer_summary_path.exists():
    previous_results = pd.read_csv(transformer_summary_path)
    required_resume_columns = {
        "task", "train_fingerprint", "validation_fingerprint", "test_fingerprint"
    }
    if not required_resume_columns.issubset(previous_results.columns):
        raise ValueError(
            "Kết quả XLM-RoBERTa cũ thiếu split fingerprints nên không thể xác minh cùng dữ liệu. "
            "Hãy đổi OUTPUT_DIR hoặc đặt SKIP_COMPLETED_TASKS = False."
        )
    for _, previous_row in previous_results.iterrows():
        previous_task = str(previous_row["task"])
        if previous_task not in split_cache:
            continue
        current_train, current_validation, current_test = split_cache[previous_task]
        current_fingerprints = {
            "train_fingerprint": article_set_fingerprint(current_train),
            "validation_fingerprint": article_set_fingerprint(current_validation),
            "test_fingerprint": article_set_fingerprint(current_test),
        }
        for column, expected in current_fingerprints.items():
            if str(previous_row[column]) != expected:
                raise ValueError(
                    f"Kết quả XLM-RoBERTa cũ của {previous_task} dùng {column} khác lần chạy hiện tại. "
                    "Hãy đổi OUTPUT_DIR hoặc đặt SKIP_COMPLETED_TASKS = False."
                )
    transformer_rows = previous_results.to_dict("records")
else:
    previous_results = pd.DataFrame()

completed_tasks = {
    str(row["task"])
    for row in transformer_rows
    if str(row.get("status", "")) == "completed"
}

for task_index, task_name in enumerate(
    TRANSFORMER_TASKS_TO_RUN, start=1
):
    print(
        f"\n===== {task_index}/{len(TRANSFORMER_TASKS_TO_RUN)}: "
        f"{task_name} ====="
    )

    if SKIP_COMPLETED_TASKS and task_name in completed_tasks:
        print("Đã có kết quả hoàn tất; bỏ qua task này.")
        continue

    try:
        summary_row = run_one_transformer_task(task_name)
        transformer_rows = [
            row for row in transformer_rows
            if str(row.get("task")) != task_name
        ]
        transformer_rows.append(summary_row)
        pd.DataFrame(transformer_rows).to_csv(
            transformer_summary_path,
            index=False,
        )
    except Exception as exc:
        failure = {
            "task": task_name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        transformer_failures.append(failure)
        print("TASK FAILED:", failure)
        pd.DataFrame(transformer_failures).to_csv(
            OUTPUT_DIR / "transformer_task_failures.csv",
            index=False,
        )
        if not CONTINUE_ON_TASK_ERROR:
            raise
    finally:
        # Giải phóng model/task trước khi nạp model cho task tiếp theo.
        for variable_name in [
            "trainer", "model", "base_classifier",
            "train_ds", "val_ds", "test_ds",
        ]:
            if variable_name in globals():
                globals().pop(variable_name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

transformer_results = pd.DataFrame(transformer_rows)
if not transformer_results.empty:
    transformer_results = transformer_results.sort_values(
        "f1_macro", ascending=False
    ).reset_index(drop=True)
    transformer_results.to_csv(transformer_summary_path, index=False)

failures_frame = pd.DataFrame(transformer_failures)
print(
    f"\nHoàn tất {len(transformer_results)} task; "
    f"lỗi trong lần chạy này: {len(failures_frame)} task."
)
display(transformer_results)
if not failures_frame.empty:
    display(failures_frame)


# === Cell 26 ===
# Cell 18 — Chọn baseline bằng validation và kiểm chứng fairness trực tiếp với XLM-RoBERTa
if transformer_results.empty:
    raise RuntimeError("Chưa có task XLM-RoBERTa nào hoàn tất. Hãy xem transformer_task_failures.csv.")

best_baseline_all = (
    baseline_results
    .sort_values(
        ["task", "validation_f1_macro", "validation_balanced_accuracy"],
        ascending=[True, False, False],
    )
    .groupby("task", as_index=False)
    .first()
    .rename(columns={
        "model": "best_baseline_model",
        "accuracy": "baseline_accuracy",
        "balanced_accuracy": "baseline_balanced_accuracy",
        "f1_macro": "baseline_f1_macro",
        "f1_weighted": "baseline_f1_weighted",
        "validation_f1_macro": "baseline_selection_validation_f1_macro",
        "n_train": "baseline_n_train",
        "n_validation": "baseline_n_validation",
        "n_test": "baseline_n_test",
        "train_fingerprint": "baseline_train_fingerprint",
        "validation_fingerprint": "baseline_validation_fingerprint",
        "test_fingerprint": "baseline_test_fingerprint",
    })
)

baseline_columns = [
    "task", "best_baseline_model", "baseline_accuracy", "baseline_balanced_accuracy",
    "baseline_f1_macro", "baseline_f1_weighted", "baseline_selection_validation_f1_macro",
    "baseline_n_train", "baseline_n_validation", "baseline_n_test",
    "baseline_train_fingerprint", "baseline_validation_fingerprint", "baseline_test_fingerprint",
]
all_task_comparison = transformer_results.merge(
    best_baseline_all[baseline_columns], on="task", how="left", validate="one_to_one"
)

for transformer_column, baseline_column in [
    ("n_train", "baseline_n_train"),
    ("n_validation", "baseline_n_validation"),
    ("n_test", "baseline_n_test"),
    ("train_fingerprint", "baseline_train_fingerprint"),
    ("validation_fingerprint", "baseline_validation_fingerprint"),
    ("test_fingerprint", "baseline_test_fingerprint"),
]:
    mismatch = all_task_comparison[transformer_column].astype(str).ne(
        all_task_comparison[baseline_column].astype(str)
    )
    if mismatch.any():
        failing = all_task_comparison.loc[mismatch, "task"].tolist()
        raise AssertionError(f"So sánh không công bằng ở {transformer_column}: {failing}")

all_task_comparison["identical_training_articles"] = True
all_task_comparison["identical_validation_articles"] = True
all_task_comparison["identical_test_articles"] = True
all_task_comparison["baseline_selected_on"] = "validation_f1_macro"
all_task_comparison["xlmr_minus_baseline_macro_f1"] = (
    all_task_comparison["f1_macro"] - all_task_comparison["baseline_f1_macro"]
)
all_task_comparison = all_task_comparison.sort_values("f1_macro", ascending=False).reset_index(drop=True)
all_task_comparison.to_csv(OUTPUT_DIR / "all_task_model_comparison.csv", index=False)

display(all_task_comparison[[
    "task", "n_classes", "n_train", "n_validation", "n_test",
    "best_baseline_model", "baseline_selection_validation_f1_macro", "baseline_f1_macro",
    "f1_macro", "macro_f1_ci_low", "macro_f1_ci_high",
    "xlmr_minus_baseline_macro_f1", "identical_training_articles",
]])


# === Cell 27 ===
# Tổng hợp kết quả four-view

master_summary = {

    "architecture":
        "four_view",

    "experiment_mode":
        EXPERIMENT_MODE,

    "train_source":
        SOURCE_TRAIN,

    "target_source":
        SOURCE_TARGET,

    "source_clean_counts": {

        str(source):
            int(count)

        for source, count
        in df[
            "source"
        ]
        .value_counts()
        .items()
    },

    "global_split_sizes": {

        split_name:
            len(part)

        for split_name, part
        in GLOBAL_SPLITS.items()
    },

    "requested_training_budget":
        TRAIN_SAMPLE_SIZE,

    "requested_cafef_validation_size":
        TARGET_VALIDATION_SIZE,

    "requested_cafef_test_size":
        TARGET_TEST_SIZE,

    "cross_source_train_overlap_removed":
        int(
            len(
                cross_source_overlap_frame
            )
        ),

    "baseline_fit_data":
        "train_only",

    "baseline_model_selection_set":
        "validation",

    "identical_article_sets_across_models":
        True,

    "target_domain_used_for_validation":
        (
            EXPERIMENT_MODE
            == "cross_source_target_validation"
        ),

    "n_clean_articles":
        int(
            len(
                df
            )
        ),

    "requested_transformer_tasks":
        TRANSFORMER_TASKS_TO_RUN,

    "completed_transformer_tasks":
        (
            transformer_results[
                "task"
            ].tolist()
        ),

    "failed_transformer_tasks_this_run":

        (
            failures_frame[
                "task"
            ].tolist()

            if not failures_frame.empty

            else []
        ),

    "n_completed_tasks":
        int(
            len(
                transformer_results
            )
        ),

    "n_failed_tasks_this_run":
        int(
            len(
                failures_frame
            )
        ),

    "model_name":
        MODEL_NAME,

    "input_strategy":
        (
            "shared_four_view_"
            "title_sapo_head_"
            "continuation_tail"
        ),

    "views": [

        "title_sapo",

        "head",

        "continuation",

        "tail",
    ],

    "max_length_per_view":
        MAX_LENGTH,

    "maximum_total_tokens_across_views":
        (
            MAX_LENGTH
            * 4
        ),

    "word_segmented": False,

    "sentiment_labels": [

        "positive",

        "negative",

        "mixed",
    ],

    "neutral_merged_into_mixed":
        True,

    "esg_pillar_strategy":
        "three_independent_binary_tasks",

    "esg_pillar_tasks": {

        "environmental": {

            "negative_label":
                "0",

            "positive_label":
                "1",
        },

        "social": {

            "negative_label":
                "0",

            "positive_label":
                "1",
        },

        "governance": {

            "negative_label":
                "0",

            "positive_label":
                "1",
        },
    },

    "mixed_esg_class_used":
        False,

    "pillar_training_scope":
        "articles_with_esg_relevance_ESG",

    "view_weight_selection_set":
        "validation",

    "default_view_weights":
        DEFAULT_VIEW_WEIGHTS,

    "seed":
        SEED,
}


with open(

    OUTPUT_DIR

    / "all_tasks_run_summary.json",

    "w",

    encoding=
        "utf-8",

) as file:

    json.dump(

        master_summary,

        file,

        ensure_ascii=
            False,

        indent=
            2,
    )


print(
    json.dumps(

        master_summary,

        ensure_ascii=
            False,

        indent=
            2,
    )
)


# === Cell 28 ===
# So sánh trực tiếp XLM-RoBERTa với PhoBERT trên cùng article fingerprints
PHOBERT_METRICS_PATH = PHOBERT_RESULTS_DIR / "all_transformer_task_metrics.csv"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "xlmr_vs_phobert_fair_comparison.csv"

if not PHOBERT_METRICS_PATH.exists():
    print(
        "Chưa thấy kết quả PhoBERT để ghép trực tiếp:",
        PHOBERT_METRICS_PATH,
    )
    print("Sau khi file xuất hiện, chạy lại cell này; không cần train lại XLM-RoBERTa.")
else:
    phobert_results = pd.read_csv(PHOBERT_METRICS_PATH)
    phobert_results = phobert_results.loc[
        phobert_results["task"].isin(TRANSFORMER_TASKS_TO_RUN)
    ].copy()

    required = {
        "task", "f1_macro", "accuracy", "balanced_accuracy",
        "train_fingerprint", "validation_fingerprint", "test_fingerprint",
        "n_train", "n_validation", "n_test",
    }
    missing_xlmr = required - set(transformer_results.columns)
    missing_phobert = required - set(phobert_results.columns)
    if missing_xlmr or missing_phobert:
        raise ValueError(
            f"Thiếu cột so sánh — XLM-R: {sorted(missing_xlmr)}; "
            f"PhoBERT: {sorted(missing_phobert)}"
        )

    model_comparison = transformer_results.merge(
        phobert_results,
        on="task",
        how="inner",
        suffixes=("_xlmr", "_phobert"),
        validate="one_to_one",
    )

    if model_comparison.empty:
        raise RuntimeError("Không có task chung giữa kết quả XLM-RoBERTa và PhoBERT.")

    for field in [
        "n_train", "n_validation", "n_test",
        "train_fingerprint", "validation_fingerprint", "test_fingerprint",
    ]:
        mismatch = model_comparison[f"{field}_xlmr"].astype(str).ne(
            model_comparison[f"{field}_phobert"].astype(str)
        )
        if mismatch.any():
            bad_tasks = model_comparison.loc[mismatch, "task"].tolist()
            raise AssertionError(
                f"Không được so sánh: {field} khác nhau ở task {bad_tasks}."
            )

    model_comparison["identical_article_sets"] = True
    for metric in ["f1_macro", "balanced_accuracy", "accuracy"]:
        model_comparison[f"xlmr_minus_phobert_{metric}"] = (
            model_comparison[f"{metric}_xlmr"]
            - model_comparison[f"{metric}_phobert"]
        )

    report_columns = [
        "task", "n_train_xlmr", "n_validation_xlmr", "n_test_xlmr",
        "f1_macro_phobert", "f1_macro_xlmr", "xlmr_minus_phobert_f1_macro",
        "balanced_accuracy_phobert", "balanced_accuracy_xlmr",
        "xlmr_minus_phobert_balanced_accuracy", "identical_article_sets",
    ]
    fair_model_comparison = model_comparison[report_columns].sort_values(
        "xlmr_minus_phobert_f1_macro",
        ascending=False,
    ).reset_index(drop=True)
    fair_model_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    display(fair_model_comparison)
    print("Đã lưu:", MODEL_COMPARISON_PATH)


# === Cell 31 ===


# === Cell 32 ===
from google.colab import runtime
runtime.unassign()

# === Cell 33 ===
