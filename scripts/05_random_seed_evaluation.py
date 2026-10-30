# Original notebook code exported for archival review. Paths/environment require adaptation.


# === Cell 2 ===
# Cell 1 — Cài thư viện
# COLAB COMMAND: %pip install -q -U \
    "transformers>=4.46,<6" \
    datasets \
    accelerate \
    sentencepiece \
    protobuf \
    openpyxl \
    pyarrow \
    scipy \
    seaborn


# === Cell 3 ===
# Cell 2 — Mount Drive, import và cấu hình thí nghiệm

from google.colab import drive
drive.mount("/content/gdrive")

import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import warnings

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from datasets import Dataset

from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

warnings.filterwarnings("ignore")


# =========================================================
# 1. Cấu hình GPU
# =========================================================

CUDA_AVAILABLE = torch.cuda.is_available()

GPU_NAME = "CPU"
GPU_CAPABILITY = None
GPU_SUPPORTS_TF32 = False

USE_BF16 = False
USE_FP16 = False

if CUDA_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_CAPABILITY = torch.cuda.get_device_capability(0)

    gpu_major, gpu_minor = GPU_CAPABILITY

    # TF32 chỉ hỗ trợ chính thức trên Ampere trở lên.
    GPU_SUPPORTS_TF32 = gpu_major >= 8

    torch.backends.cuda.matmul.allow_tf32 = (
        GPU_SUPPORTS_TF32
    )
    torch.backends.cudnn.allow_tf32 = (
        GPU_SUPPORTS_TF32
    )

    torch.set_float32_matmul_precision("high")

    USE_BF16 = bool(
        hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    USE_FP16 = not USE_BF16


print("=" * 72)
print("GPU CONFIGURATION")
print("=" * 72)
print("CUDA available:", CUDA_AVAILABLE)
print("GPU:", GPU_NAME)
print("GPU capability:", GPU_CAPABILITY)
print("TF32:", GPU_SUPPORTS_TF32)
print("BF16:", USE_BF16)
print("FP16:", USE_FP16)


# =========================================================
# 2. Đường dẫn
# =========================================================

OUTPUT_ROOT = Path(
    "/content/gdrive/MyDrive/"
    "Project_Fin_Data_Crawling/"
    "Crawling_news/"
    "ESG_annotation_sampling"
)

# Pipeline four-view chính thức, dùng để lấy:
# - annotation;
# - manifest;
# - segmentation;
# - split train/validation/test.
PRIMARY_DIR = (
    OUTPUT_ROOT
    / (
        "esg_nlp_results_cross_source_target_validation_"
        "vietstock_to_cafef_four_view"
    )
)

# Thư mục kết quả random-seed.
EXPERIMENT_DIR = (
    OUTPUT_ROOT
    / "randomseed_phobert_xlmr_6tasks_locked_weights"
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Checkpoint từng epoch được lưu trên ổ local của Colab.
# Không ghi checkpoint lớn trực tiếp lên Google Drive.
LOCAL_CHECKPOINT_ROOT = Path(
    "/content/esg_randomseed_checkpoints"
)

LOCAL_CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_PATH = (
    EXPERIMENT_DIR
    / "all_model_seed_task_metrics.csv"
)


# =========================================================
# 3. Model
# =========================================================

# MODEL_SPECS chỉ chứa đặc điểm nhận dạng/tokenizer.
MODEL_SPECS = {
    "phobert": {
        "display_name": "PhoBERT-base-v2",
        "checkpoint": "vinai/phobert-base-v2",
        "text_mode": "segmented",
        "use_fast": False,
    },

    "xlmr": {
        "display_name": "XLM-RoBERTa-base",
        "checkpoint": "FacebookAI/xlm-roberta-base",
        "text_mode": "raw",
        "use_fast": True,
    },
}


# Cấu hình huấn luyện thực sự được Cell 8 sử dụng.
#
# Effective batch:
# - PhoBERT: 16 × 1 = 16
# - XLM-R:   16 × 1 = 16
#
# Cell 6 ghép bốn view:
# train batch 16 tương đương 64 sequence/forward.
MODEL_RUNTIME_CONFIG = {
    "phobert": {
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "gradient_accumulation": 1,
        "learning_rate": 2e-5,
    },

    "xlmr": {
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "gradient_accumulation": 1,
        "learning_rate": 2e-5,
    },
}


# =========================================================
# 4. Sáu task báo cáo
# =========================================================

TASKS_TO_RUN = [
    "esg_relevance",
    "sentiment_all",
    "sentiment_esg",
    "environmental",
    "social",
    "governance",
]

TASK_DISPLAY = {
    "esg_relevance": "esg_relevance",
    "sentiment_all": "sentiment_all",
    "sentiment_esg": "sentiment_esg",
    "environmental": "E",
    "social": "S",
    "governance": "G",
}


# =========================================================
# 5. Random seed và hyperparameter chung
# =========================================================

# Ba seed là mức tối thiểu để báo cáo độ ổn định.
MULTI_SEEDS = [
    13,
    42,
    77,
]

MAX_LENGTH = 256
NUM_EPOCHS = 6

LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.08
EARLY_STOPPING_PATIENCE = 2
WEIGHT_DECAY = 0.01

# Biến mặc định để tương thích với các cell cũ.
# Cell 8 mới phải ưu tiên MODEL_RUNTIME_CONFIG.
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1

# DataLoader.
CPU_COUNT = os.cpu_count() or 2

DATALOADER_NUM_WORKERS = max(
    1,
    min(4, CPU_COUNT),
)

# Không giữ checkpoint sau khi đã lấy best model và đánh giá.
KEEP_CHECKPOINTS = False

# True: Cell 9 thực hiện train/resume.
RUN_TRAINING = True

BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260922


# =========================================================
# 6. Four-view
# =========================================================

VIEW_NAMES = [
    "title_sapo",
    "head",
    "continuation",
    "tail",
]

DEFAULT_VIEW_WEIGHTS = {
    "title_sapo": 0.20,
    "head": 0.35,
    "continuation": 0.25,
    "tail": 0.20,
}

# Weight dùng trong loss/fused logits khi train.
TRAINING_VIEW_WEIGHTS = (
    DEFAULT_VIEW_WEIGHTS.copy()
)


# =========================================================
# 7. Locked inference weights theo model × task
# =========================================================

# Các weight này đã được chọn từ validation của lượt chạy
# chính thức trước đó và được khóa qua mọi random seed.
#
# Không lựa chọn lại weight theo từng seed.
LOCKED_INFERENCE_WEIGHTS = {
    "phobert": {
        "esg_relevance": {
            "title_sapo": 0.20,
            "head": 0.50,
            "continuation": 0.00,
            "tail": 0.30,
        },

        "sentiment_all": {
            "title_sapo": 0.40,
            "head": 0.30,
            "continuation": 0.20,
            "tail": 0.10,
        },

        "sentiment_esg": {
            "title_sapo": 0.30,
            "head": 0.40,
            "continuation": 0.00,
            "tail": 0.30,
        },

        "environmental": {
            "title_sapo": 0.20,
            "head": 0.30,
            "continuation": 0.30,
            "tail": 0.20,
        },

        "social": {
            "title_sapo": 0.40,
            "head": 0.60,
            "continuation": 0.00,
            "tail": 0.00,
        },

        "governance": {
            "title_sapo": 0.20,
            "head": 0.50,
            "continuation": 0.00,
            "tail": 0.30,
        },
    },

    "xlmr": {
        "esg_relevance": {
            "title_sapo": 0.30,
            "head": 0.40,
            "continuation": 0.00,
            "tail": 0.30,
        },

        "sentiment_all": {
            "title_sapo": 0.30,
            "head": 0.35,
            "continuation": 0.25,
            "tail": 0.10,
        },

        "sentiment_esg": {
            "title_sapo": 0.30,
            "head": 0.40,
            "continuation": 0.30,
            "tail": 0.00,
        },

        "environmental": {
            "title_sapo": 0.20,
            "head": 0.30,
            "continuation": 0.30,
            "tail": 0.20,
        },

        "social": {
            "title_sapo": 0.30,
            "head": 0.40,
            "continuation": 0.30,
            "tail": 0.00,
        },

        "governance": {
            "title_sapo": 0.20,
            "head": 0.40,
            "continuation": 0.40,
            "tail": 0.00,
        },
    },
}


WEIGHT_REFERENCE = {
    "phobert": (
        "esg_nlp_results_cross_source_target_validation_"
        "vietstock_to_cafef_four_view/"
        "all_transformer_task_metrics.csv"
    ),

    "xlmr": (
        "esg_nlp_results_cross_source_target_validation_"
        "vietstock_to_cafef_xlm_roberta_four_view/"
        "all_transformer_task_metrics.csv"
    ),
}


# =========================================================
# 8. Kiểm tra cấu hình
# =========================================================

assert set(MODEL_RUNTIME_CONFIG) == set(MODEL_SPECS), (
    "MODEL_RUNTIME_CONFIG phải có đủ PhoBERT và XLM-R."
)

assert set(
    LOCKED_INFERENCE_WEIGHTS
) == set(MODEL_SPECS)

for model_key in MODEL_SPECS:
    assert (
        set(LOCKED_INFERENCE_WEIGHTS[model_key])
        == set(TASKS_TO_RUN)
    )

    runtime = MODEL_RUNTIME_CONFIG[model_key]

    assert runtime["train_batch_size"] >= 1
    assert runtime["eval_batch_size"] >= 1
    assert runtime["gradient_accumulation"] >= 1
    assert runtime["learning_rate"] > 0

    for task_name, weights in (
        LOCKED_INFERENCE_WEIGHTS[model_key].items()
    ):
        assert set(weights) == set(VIEW_NAMES)

        assert all(
            value >= 0
            for value in weights.values()
        )

        assert abs(
            sum(weights.values()) - 1.0
        ) < 1e-12


runtime_audit = pd.DataFrame([
    {
        "model_key": model_key,
        "model": MODEL_SPECS[
            model_key
        ]["display_name"],

        "train_batch_size": runtime[
            "train_batch_size"
        ],

        "eval_batch_size": runtime[
            "eval_batch_size"
        ],

        "gradient_accumulation": runtime[
            "gradient_accumulation"
        ],

        "effective_batch_size": (
            runtime["train_batch_size"]
            * runtime["gradient_accumulation"]
        ),

        "sequences_per_train_forward": (
            runtime["train_batch_size"]
            * len(VIEW_NAMES)
        ),

        "learning_rate": runtime[
            "learning_rate"
        ],
    }

    for model_key, runtime
    in MODEL_RUNTIME_CONFIG.items()
])


# Đảm bảo hai model có cùng effective batch.
assert (
    runtime_audit[
        "effective_batch_size"
    ].nunique()
    == 1
), (
    "PhoBERT và XLM-R phải có cùng "
    "effective batch size."
)


locked_weights_audit = pd.DataFrame([
    {
        "model_key": model_key,

        "model": MODEL_SPECS[
            model_key
        ]["display_name"],

        "task": task_name,

        "report_name": TASK_DISPLAY[
            task_name
        ],

        **weights,

        "weight_source": WEIGHT_REFERENCE[
            model_key
        ],

        "selection_split": "validation",
        "locked_across_seeds": True,
    }

    for model_key, task_weights
    in LOCKED_INFERENCE_WEIGHTS.items()

    for task_name, weights
    in task_weights.items()
])


# =========================================================
# 9. Lưu audit cấu hình
# =========================================================

locked_weights_audit.to_csv(
    EXPERIMENT_DIR
    / "locked_inference_weights.csv",
    index=False,
)

runtime_audit.to_csv(
    EXPERIMENT_DIR
    / "model_runtime_configuration.csv",
    index=False,
)


# =========================================================
# 10. Hiển thị cấu hình
# =========================================================

print("\n" + "=" * 72)
print("EXPERIMENT CONFIGURATION")
print("=" * 72)

print("Experiment:", EXPERIMENT_DIR)
print("Temporary checkpoints:", LOCAL_CHECKPOINT_ROOT)

print(
    "Models:",
    {
        key: value["checkpoint"]
        for key, value
        in MODEL_SPECS.items()
    },
)

print(
    "Tasks:",
    [
        TASK_DISPLAY[task]
        for task in TASKS_TO_RUN
    ],
)

print("Seeds:", MULTI_SEEDS)

print(
    "Total planned runs:",
    (
        len(MODEL_SPECS)
        * len(TASKS_TO_RUN)
        * len(MULTI_SEEDS)
    ),
)

print(
    "DataLoader workers:",
    DATALOADER_NUM_WORKERS,
)

print("\nRuntime configuration:")
display(runtime_audit)

print("\nLocked inference weights:")
display(locked_weights_audit)

# === Cell 4 ===
# Cell 3 — Đọc annotation, manifest và segmentation
MANIFEST_PATH = PRIMARY_DIR / "global_article_split_manifest.csv"
SEGMENTED_CORPUS_PATH = PRIMARY_DIR / "segmented_corpus_all_tasks.parquet"

for required_path in [MANIFEST_PATH, SEGMENTED_CORPUS_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(f"Thiếu artifact bắt buộc: {required_path}")

VIETSTOCK_DATA_GLOBS = [
    str(OUTPUT_ROOT / "03_annotation_batches" / "Done" / "annotation_batch_*.xlsx"),
]
CAFEF_DATA_GLOBS = [
    str(
        OUTPUT_ROOT
        / "ESG_annotation_CafeF_800/02_annotation_batches_41_56"
        / "Done/annotation_batch_*.xlsx"
    ),
    str(OUTPUT_ROOT / "CafeF_Done" / "*.xlsx"),
]


def snake_case(name):
    name = str(name).strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"[^0-9A-Za-zÀ-ỹ]+", "_", name)
    return name.strip("_").lower()


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


def normalize_article_id(value):
    value = clean_scalar(value)
    if pd.isna(value):
        return np.nan
    return re.sub(r"\.0$", "", str(value).strip())


def safe_text(value):
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "nan", "none", "<na>"} else value


def read_annotation_file(path, source_name):
    workbook = pd.ExcelFile(path)
    sheet_lookup = {str(s).strip().lower(): s for s in workbook.sheet_names}
    sheet_used = sheet_lookup.get("annotation", workbook.sheet_names[0])
    frame = pd.read_excel(path, sheet_name=sheet_used)
    frame.columns = [snake_case(c) for c in frame.columns]
    frame["source"] = source_name
    frame["_source_file"] = path.name
    return frame


file_specs = []
for source_name, patterns in [
    ("vietstock", VIETSTOCK_DATA_GLOBS),
    ("cafef", CAFEF_DATA_GLOBS),
]:
    matched = []
    for pattern in patterns:
        matched.extend(Path(p) for p in sorted(glob(pattern)))
    unique_paths, seen = [], set()
    for path in matched:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    print(source_name, "files:", len(unique_paths))
    file_specs.extend((p, source_name) for p in unique_paths)

if not file_specs:
    raise FileNotFoundError("Không tìm thấy annotation Vietstock/CafeF.")

raw = pd.concat(
    [read_annotation_file(path, source) for path, source in file_specs],
    ignore_index=True,
    sort=False,
)

if "annotation_status" in raw.columns:
    completed = raw["annotation_status"].map(norm_label).isin(
        {"completed", "complete", "done", "adjudicated", "1", "true"}
    )
    if completed.any():
        raw = raw.loc[completed].copy()

if "is_duplicate" in raw.columns:
    duplicate = raw["is_duplicate"].map(norm_label).isin(
        {"1", "true", "yes", "duplicate"}
    )
    raw = raw.loc[~duplicate].copy()

if "article_id" not in raw.columns:
    raise KeyError("Annotation không có article_id.")

raw["article_id"] = raw["article_id"].map(normalize_article_id)
raw = raw.loc[raw["article_id"].notna()].copy()
raw = raw.drop_duplicates(subset=["source", "article_id"], keep="last")

relevance_map = {
    "relevant": "ESG", "esg": "ESG",
    "not_relevant": "Non_ESG", "non_esg": "Non_ESG", "nonesg": "Non_ESG",
}
raw["esg_relevance"] = raw["esg_relevance"].map(norm_label).map(
    lambda x: relevance_map.get(x, x) if pd.notna(x) else np.nan
)

# Theo thiết kế nghiên cứu đã chốt: neutral được gộp vào mixed.
sentiment_map = {
    "positive": "positive", "negative": "negative",
    "neutral": "mixed", "mixed": "mixed",
}
raw["sentiment"] = raw["sentiment"].map(norm_label).map(
    lambda x: sentiment_map.get(x, x) if pd.notna(x) else np.nan
)

POSITIVE_BINARY = {"1", "1.0", "true", "yes", "y", "co", "có", "present", "positive"}
NEGATIVE_BINARY = {"0", "0.0", "false", "no", "n", "khong", "không", "absent", "negative"}


def to_binary_label(value):
    value = norm_label(value)
    if pd.isna(value):
        return np.nan
    if value in POSITIVE_BINARY:
        return "1"
    if value in NEGATIVE_BINARY:
        return "0"
    return np.nan


for column in ["environmental", "social", "governance"]:
    raw[column] = raw[column].map(to_binary_label)

split_manifest = pd.read_csv(MANIFEST_PATH)
split_manifest["article_id"] = split_manifest["article_id"].map(normalize_article_id)
split_manifest["_row_key"] = split_manifest["_row_key"].astype(int)

manifest_core = split_manifest[["_row_key", "article_id", "source", "split"]].copy()
df = manifest_core.merge(
    raw,
    on=["source", "article_id"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

missing = df.loc[df["_merge"].ne("both"), ["_row_key", "source", "article_id", "split"]]
if not missing.empty:
    display(missing.head(30))
    raise ValueError(f"Không ghép được annotation cho {len(missing)} bài trong manifest.")
df = df.drop(columns="_merge")

segmented = pd.read_parquet(SEGMENTED_CORPUS_PATH)
segment_columns = ["_row_key", "title_segmented", "sapo_segmented", "body_segmented"]
missing_segment_columns = set(segment_columns) - set(segmented.columns)
if missing_segment_columns:
    raise KeyError(f"Cache segmentation thiếu: {sorted(missing_segment_columns)}")

segmented = segmented[segment_columns].drop_duplicates("_row_key", keep="last").copy()
segmented["_row_key"] = segmented["_row_key"].astype(int)
df = df.merge(segmented, on="_row_key", how="left", validate="one_to_one")

for column in ["title_segmented", "sapo_segmented", "body_segmented"]:
    df[column] = df[column].map(safe_text)

# XLM-RoBERTa dùng văn bản nguyên bản; fallback có kiểm soát nếu tên cột khác nhau.
def first_existing_text(frame, candidates):
    output = pd.Series("", index=frame.index, dtype="object")
    for candidate in candidates:
        if candidate in frame.columns:
            values = frame[candidate].map(safe_text)
            output = output.mask(output.eq(""), values)
    return output


df["title_raw"] = first_existing_text(df, ["title", "article_title"])
df["sapo_raw"] = first_existing_text(df, ["sapo", "description", "summary"])
df["body_raw"] = first_existing_text(df, ["content", "body", "article_content"])

if df["body_raw"].eq("").mean() > 0.05:
    raise ValueError("Hơn 5% bài thiếu body_raw; cần kiểm tra tên cột nội dung trước khi train XLM-R.")

assert len(df) == len(split_manifest)
display(df.groupby(["split", "source"]).size().rename("n").reset_index())
print("Articles restored:", len(df))


# === Cell 5 ===
# Cell 4 — Tạo đúng sáu task và khóa tập bài bằng fingerprint
TASK_SPECS = {
    "esg_relevance": {
        "column": "esg_relevance", "filter_esg": False,
        "valid_labels": {"ESG", "Non_ESG"},
    },
    "sentiment_all": {
        "column": "sentiment", "filter_esg": False,
        "valid_labels": {"positive", "negative", "mixed"},
    },
    "sentiment_esg": {
        "column": "sentiment", "filter_esg": True,
        "valid_labels": {"positive", "negative", "mixed"},
    },
    "environmental": {
        "column": "environmental", "filter_esg": True,
        "valid_labels": {"0", "1"},
    },
    "social": {
        "column": "social", "filter_esg": True,
        "valid_labels": {"0", "1"},
    },
    "governance": {
        "column": "governance", "filter_esg": True,
        "valid_labels": {"0", "1"},
    },
}


def article_set_fingerprint(frame):
    keys = sorted(frame["_row_key"].astype(int).tolist())
    return hashlib.sha256(",".join(map(str, keys)).encode("utf-8")).hexdigest()


def get_task_frame(task_name):
    spec = TASK_SPECS[task_name]
    work = df.copy()
    if spec["filter_esg"]:
        work = work.loc[work["esg_relevance"].eq("ESG")].copy()
    work["label"] = work[spec["column"]].map(clean_scalar)
    work = work.loc[work["label"].notna()].copy()
    work["label"] = work["label"].astype(str)
    invalid = sorted(set(work["label"]) - spec["valid_labels"])
    if invalid:
        raise ValueError(f"{task_name} có nhãn ngoài thiết kế: {invalid}")
    parts = tuple(work.loc[work["split"].eq(s)].copy() for s in ["train", "validation", "test"])
    train_labels = set(parts[0]["label"])
    for split_name, part in zip(["train", "validation", "test"], parts):
        if part.empty:
            raise ValueError(f"{task_name}/{split_name} rỗng.")
        unknown = set(part["label"]) - train_labels
        if unknown:
            raise ValueError(f"{task_name}/{split_name} có class không xuất hiện ở train: {unknown}")
    return parts


task_cache = {task: get_task_frame(task) for task in TASKS_TO_RUN}
audit_rows = []
for task, (train_df, val_df, test_df) in task_cache.items():
    row = {
        "task": task,
        "report_name": TASK_DISPLAY[task],
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "train_fingerprint": article_set_fingerprint(train_df),
        "validation_fingerprint": article_set_fingerprint(val_df),
        "test_fingerprint": article_set_fingerprint(test_df),
    }
    audit_rows.append(row)
    print("\n", TASK_DISPLAY[task], row["n_train"], row["n_validation"], row["n_test"])
    display(pd.crosstab(df.loc[df.index.isin(train_df.index), "split"], train_df["label"]))

task_data_audit = pd.DataFrame(audit_rows)
task_data_audit.to_csv(EXPERIMENT_DIR / "task_data_audit.csv", index=False)
display(task_data_audit)


# === Cell 6 ===
# Cell 5 — Tokenizer tương thích PhoBERT và XLM-RoBERTa
TOKENIZER_CACHE = {}


def get_tokenizer(model_key):
    if model_key not in TOKENIZER_CACHE:
        spec = MODEL_SPECS[model_key]

        TOKENIZER_CACHE[model_key] = (
            AutoTokenizer.from_pretrained(
                spec["checkpoint"],
                use_fast=spec["use_fast"],
            )
        )

    return TOKENIZER_CACHE[model_key]


def ids_without_special_tokens(
    tokenizer,
    text,
):
    text = safe_text(text)

    if not text:
        return []

    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )


def prepare_fixed_length(
    tokenizer,
    token_ids,
):
    """
    Đóng gói một sequence mà không sử dụng các API
    tokenizer đã bị loại bỏ trong Transformers 5.x.

    PhoBERT và XLM-RoBERTa đều dùng:

        BOS/CLS + content + EOS/SEP + padding
    """

    special_count = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    if special_count != 2:
        raise ValueError(
            f"{type(tokenizer).__name__}: "
            f"expected 2 special tokens for "
            f"a single sequence, found "
            f"{special_count}."
        )

    # Token mở đầu: ưu tiên BOS, fallback sang CLS.
    start_id = tokenizer.bos_token_id

    if start_id is None:
        start_id = tokenizer.cls_token_id

    # Token kết thúc: ưu tiên EOS, fallback sang SEP.
    end_id = tokenizer.eos_token_id

    if end_id is None:
        end_id = tokenizer.sep_token_id

    if (
        start_id is None
        or end_id is None
    ):
        raise ValueError(
            f"{type(tokenizer).__name__} "
            "thiếu BOS/CLS hoặc EOS/SEP token ID."
        )

    content_budget = (
        MAX_LENGTH - special_count
    )

    content_ids = list(
        token_ids[:content_budget]
    )

    # Không gọi:
    # tokenizer.prepare_for_model(...)
    # tokenizer.build_inputs_with_special_tokens(...)
    input_ids = [
        int(start_id),
        *content_ids,
        int(end_id),
    ]

    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    if pad_id is None:
        raise ValueError(
            "Tokenizer không có pad_token_id "
            "hoặc eos_token_id."
        )

    attention_mask = (
        [1] * len(input_ids)
    )

    padding = (
        MAX_LENGTH - len(input_ids)
    )

    if padding < 0:
        raise AssertionError(
            f"Sequence dài {len(input_ids)} "
            f"vượt MAX_LENGTH={MAX_LENGTH}."
        )

    if tokenizer.padding_side == "left":
        input_ids = (
            [pad_id] * padding
            + input_ids
        )

        attention_mask = (
            [0] * padding
            + attention_mask
        )

    else:
        input_ids = (
            input_ids
            + [pad_id] * padding
        )

        attention_mask = (
            attention_mask
            + [0] * padding
        )

    assert len(input_ids) == MAX_LENGTH
    assert len(attention_mask) == MAX_LENGTH

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def split_body_into_views(
    body_ids,
    content_budget,
):
    body_length = len(body_ids)

    head = body_ids[
        :content_budget
    ]

    continuation = body_ids[
        content_budget:
        2 * content_budget
    ]

    # Tail chỉ tồn tại nếu còn token chưa xuất hiện
    # trong head và continuation.
    if body_length > 2 * content_budget:
        tail_start = max(
            2 * content_budget,
            body_length - content_budget,
        )

        tail = body_ids[
            tail_start:
        ]

    else:
        tail = []

    return {
        "head": head,
        "continuation": continuation,
        "tail": tail,
        "continuation_available":
            int(bool(continuation)),
        "tail_available":
            int(bool(tail)),
    }


def encode_four_views(
    example,
    tokenizer,
    text_mode,
    label_to_id,
):
    suffix = (
        "segmented"
        if text_mode == "segmented"
        else "raw"
    )

    title = safe_text(
        example.get(
            f"title_{suffix}",
            "",
        )
    )

    sapo = safe_text(
        example.get(
            f"sapo_{suffix}",
            "",
        )
    )

    body = safe_text(
        example.get(
            f"body_{suffix}",
            "",
        )
    )

    title_sapo = " ".join(
        value
        for value in [title, sapo]
        if value
    )

    special_count = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    content_budget = (
        MAX_LENGTH - special_count
    )

    title_sapo_ids = (
        ids_without_special_tokens(
            tokenizer,
            title_sapo,
        )
    )

    body_ids = (
        ids_without_special_tokens(
            tokenizer,
            body,
        )
    )

    views = split_body_into_views(
        body_ids,
        content_budget,
    )

    encoded = {}

    view_token_ids = {
        "title_sapo":
            title_sapo_ids,
        "head":
            views["head"],
        "continuation":
            views["continuation"],
        "tail":
            views["tail"],
    }

    for (
        view_name,
        view_ids,
    ) in view_token_ids.items():

        part = prepare_fixed_length(
            tokenizer,
            view_ids,
        )

        encoded[
            f"{view_name}_input_ids"
        ] = part["input_ids"]

        encoded[
            f"{view_name}_attention_mask"
        ] = part["attention_mask"]

    encoded[
        "continuation_available"
    ] = views[
        "continuation_available"
    ]

    encoded[
        "tail_available"
    ] = views[
        "tail_available"
    ]

    encoded["labels"] = (
        label_to_id[
            str(example["label"])
        ]
    )

    return encoded


MODEL_COLUMNS = [
    *[
        f"{view}_{field}"
        for view in VIEW_NAMES
        for field in [
            "input_ids",
            "attention_mask",
        ]
    ],
    "continuation_available",
    "tail_available",
    "labels",
]


def to_hf_dataset(
    frame,
    model_key,
    label_to_id,
    description,
):
    tokenizer = get_tokenizer(
        model_key
    )

    text_mode = (
        MODEL_SPECS[
            model_key
        ]["text_mode"]
    )

    keep = [
        "label",
        "title_segmented",
        "sapo_segmented",
        "body_segmented",
        "title_raw",
        "sapo_raw",
        "body_raw",
    ]

    dataset = Dataset.from_pandas(
        frame[keep].copy(),
        preserve_index=False,
    )

    original_columns = (
        dataset.column_names
    )

    dataset = dataset.map(
        lambda example: encode_four_views(
            example,
            tokenizer,
            text_mode,
            label_to_id,
        ),
        remove_columns=
            original_columns,
        desc=description,
    )

    dataset.set_format(
        type="torch",
        columns=MODEL_COLUMNS,
    )

    return dataset


# ============================================================
# Smoke test cho cả hai tokenizer
# ============================================================

for model_key in MODEL_SPECS:
    tok = get_tokenizer(
        model_key
    )

    probe_ids = (
        ids_without_special_tokens(
            tok,
            "Kiểm tra tokenizer tiếng Việt.",
        )
    )

    probe = prepare_fixed_length(
        tok,
        probe_ids,
    )

    assert (
        len(probe["input_ids"])
        == MAX_LENGTH
    )

    assert (
        len(probe["attention_mask"])
        == MAX_LENGTH
    )

    real_positions = np.flatnonzero(
        np.asarray(
            probe["attention_mask"]
        )
    )

    assert len(real_positions) >= 2

    first_real = int(
        real_positions[0]
    )

    last_real = int(
        real_positions[-1]
    )

    valid_start_ids = {
        value
        for value in [
            tok.bos_token_id,
            tok.cls_token_id,
        ]
        if value is not None
    }

    valid_end_ids = {
        value
        for value in [
            tok.eos_token_id,
            tok.sep_token_id,
        ]
        if value is not None
    }

    assert (
        probe["input_ids"][first_real]
        in valid_start_ids
    )

    assert (
        probe["input_ids"][last_real]
        in valid_end_ids
    )

    print(
        model_key,
        "PASS",
        type(tok).__name__,
        "special_tokens=",
        tok.num_special_tokens_to_add(
            pair=False
        ),
        "start_id=",
        probe["input_ids"][first_real],
        "end_id=",
        probe["input_ids"][last_real],
        "pad_id=",
        tok.pad_token_id,
    )

# === Cell 7 ===
# Cell 6 — Mô hình four-view tối ưu và các hàm đánh giá

class SharedFourViewClassifier(nn.Module):
    """
    Four-view classifier dùng chung một base transformer.

    Bốn view:
        1. title_sapo
        2. head
        3. continuation
        4. tail

    Thay vì gọi transformer bốn lần, cell này ghép bốn view
    thành một tensor [4 × batch_size, sequence_length] và chỉ
    gọi base classifier một lần.
    """

    def __init__(
        self,
        classifier,
        class_weights,
        default_weights,
    ):
        super().__init__()

        self.base_classifier = classifier
        self.config = classifier.config

        self.register_buffer(
            "class_weights",
            class_weights.float(),
        )

        self.register_buffer(
            "view_weights",
            torch.tensor(
                [
                    default_weights[view_name]
                    for view_name in VIEW_NAMES
                ],
                dtype=torch.float32,
            ),
        )

    def normalized_weights(
        self,
        batch_size,
        continuation_available,
        tail_available,
        device,
    ):
        """
        Chuẩn hóa view weights cho từng bài.

        Nếu bài không có continuation hoặc tail, weight của view
        tương ứng được đặt về 0 và các weight còn lại được chuẩn hóa.
        """

        base_weights = (
            self.view_weights
            .to(device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        availability = torch.stack(
            [
                torch.ones(
                    batch_size,
                    device=device,
                ),
                torch.ones(
                    batch_size,
                    device=device,
                ),
                continuation_available
                .float()
                .to(device),
                tail_available
                .float()
                .to(device),
            ],
            dim=1,
        )

        effective_weights = (
            base_weights
            * availability
        )

        denominator = (
            effective_weights
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(1e-12)
        )

        return (
            effective_weights
            / denominator
        )

    def forward(
        self,
        title_sapo_input_ids=None,
        title_sapo_attention_mask=None,
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
        # ====================================================
        # 1. Ghép bốn view thành một transformer batch
        # ====================================================

        batch_size = (
            title_sapo_input_ids.shape[0]
        )

        all_input_ids = torch.cat(
            [
                title_sapo_input_ids,
                head_input_ids,
                continuation_input_ids,
                tail_input_ids,
            ],
            dim=0,
        )

        all_attention_mask = torch.cat(
            [
                title_sapo_attention_mask,
                head_attention_mask,
                continuation_attention_mask,
                tail_attention_mask,
            ],
            dim=0,
        )

        # all_input_ids:
        # [4 × batch_size, MAX_LENGTH]

        all_logits = (
            self.base_classifier(
                input_ids=all_input_ids,
                attention_mask=
                    all_attention_mask,
            )
            .logits
        )

        num_labels = (
            all_logits.shape[-1]
        )

        # Thứ tự ban đầu:
        # title batch → head batch →
        # continuation batch → tail batch
        #
        # Chuyển thành:
        # [batch_size, 4 views, num_labels]

        stacked_logits = (
            all_logits
            .reshape(
                len(VIEW_NAMES),
                batch_size,
                num_labels,
            )
            .permute(1, 0, 2)
            .contiguous()
        )

        # ====================================================
        # 2. Chuẩn hóa weight theo view khả dụng
        # ====================================================

        normalized_weights = (
            self.normalized_weights(
                batch_size=
                    batch_size,
                continuation_available=
                    continuation_available,
                tail_available=
                    tail_available,
                device=
                    stacked_logits.device,
            )
        )

        # ====================================================
        # 3. Hợp nhất logits của bốn view
        # ====================================================

        fused_logits = (
            stacked_logits
            * normalized_weights
            .unsqueeze(-1)
        ).sum(dim=1)

        # ====================================================
        # 4. Loss có class weight và view weight
        # ====================================================

        loss = None

        if labels is not None:
            per_view_losses = []

            for view_index in range(
                len(VIEW_NAMES)
            ):
                view_loss = (
                    F.cross_entropy(
                        stacked_logits[
                            :,
                            view_index,
                            :
                        ],
                        labels,
                        weight=
                            self.class_weights,
                        reduction="none",
                    )
                )

                per_view_losses.append(
                    view_loss
                )

            # [batch_size, 4 views]

            per_view_losses = (
                torch.stack(
                    per_view_losses,
                    dim=1,
                )
            )

            loss = (
                per_view_losses
                * normalized_weights
            ).sum(dim=1).mean()

        return {
            "loss": loss,
            "logits": fused_logits,
        }


# ============================================================
# Metric dùng thống nhất cho validation và test
# ============================================================

def metric_row(
    y_true,
    y_pred,
):
    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "f1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "f1_weighted":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
    }


# ============================================================
# Khóa toàn bộ nguồn random
# ============================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


# ============================================================
# Dự đoán xác suất riêng của từng view
# ============================================================

@torch.no_grad()
def predict_view_probabilities(
    model,
    dataset,
    batch_size,
):
    """
    Trả về xác suất riêng của bốn view.

    Hàm này cũng ghép bốn view thành một forward để tăng
    GPU utilization trong validation và test.
    """

    device = next(
        model.parameters()
    ).device

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=
            default_data_collator,
        pin_memory=
            torch.cuda.is_available(),
        num_workers=2,
    )

    model.eval()

    all_view_probabilities = []
    all_true_ids = []
    all_continuation_available = []
    all_tail_available = []

    use_bf16 = bool(
        globals().get(
            "USE_BF16",
            False,
        )
    )

    amp_enabled = (
        device.type == "cuda"
    )

    amp_dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    for batch in loader:
        labels = batch.pop(
            "labels"
        )

        continuation_available = (
            batch.pop(
                "continuation_available"
            )
        )

        tail_available = (
            batch.pop(
                "tail_available"
            )
        )

        batch = {
            key: value.to(
                device,
                non_blocking=True,
            )
            for key, value
            in batch.items()
        }

        current_batch_size = (
            batch[
                "title_sapo_input_ids"
            ].shape[0]
        )

        # ====================================================
        # Ghép bốn view
        # ====================================================

        all_input_ids = torch.cat(
            [
                batch[
                    "title_sapo_input_ids"
                ],
                batch[
                    "head_input_ids"
                ],
                batch[
                    "continuation_input_ids"
                ],
                batch[
                    "tail_input_ids"
                ],
            ],
            dim=0,
        )

        all_attention_mask = torch.cat(
            [
                batch[
                    "title_sapo_attention_mask"
                ],
                batch[
                    "head_attention_mask"
                ],
                batch[
                    "continuation_attention_mask"
                ],
                batch[
                    "tail_attention_mask"
                ],
            ],
            dim=0,
        )

        # Mixed precision cho validation/test.

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            all_logits = (
                model.base_classifier(
                    input_ids=
                        all_input_ids,
                    attention_mask=
                        all_attention_mask,
                )
                .logits
            )

        num_labels = (
            all_logits.shape[-1]
        )

        view_logits = (
            all_logits
            .reshape(
                len(VIEW_NAMES),
                current_batch_size,
                num_labels,
            )
            .permute(1, 0, 2)
            .contiguous()
        )

        view_probabilities = (
            torch.softmax(
                view_logits.float(),
                dim=-1,
            )
        )

        all_view_probabilities.append(
            view_probabilities
            .cpu()
            .numpy()
        )

        all_true_ids.append(
            labels.numpy()
        )

        all_continuation_available.append(
            continuation_available.numpy()
        )

        all_tail_available.append(
            tail_available.numpy()
        )

    return {
        "view_probabilities":
            np.concatenate(
                all_view_probabilities,
                axis=0,
            ),

        "true_ids":
            np.concatenate(
                all_true_ids,
                axis=0,
            ),

        "continuation_available":
            np.concatenate(
                all_continuation_available,
                axis=0,
            ).astype(float),

        "tail_available":
            np.concatenate(
                all_tail_available,
                axis=0,
            ).astype(float),
    }


# ============================================================
# Hợp nhất xác suất bằng locked inference weights
# ============================================================

def combine_view_probabilities(
    view_probabilities,
    continuation_available,
    tail_available,
    weights,
):
    """
    view_probabilities:
        [n_articles, 4 views, n_classes]

    Nếu continuation hoặc tail không tồn tại, weight của view
    tương ứng được đặt về 0 và các weight còn lại được chuẩn hóa.
    """

    base_weights = np.array(
        [
            weights[view_name]
            for view_name
            in VIEW_NAMES
        ],
        dtype=float,
    )

    availability = np.column_stack(
        [
            np.ones(
                len(
                    view_probabilities
                )
            ),

            np.ones(
                len(
                    view_probabilities
                )
            ),

            continuation_available,

            tail_available,
        ]
    )

    effective_weights = (
        availability
        * base_weights[None, :]
    )

    denominators = (
        effective_weights
        .sum(
            axis=1,
            keepdims=True,
        )
    )

    # Trường hợp bảo vệ, dù title và head luôn có weight.

    zero_rows = (
        denominators[:, 0]
        <= 0
    )

    if zero_rows.any():
        effective_weights[
            zero_rows,
            0,
        ] = 1.0

        denominators = (
            effective_weights
            .sum(
                axis=1,
                keepdims=True,
            )
        )

    normalized_weights = (
        effective_weights
        / denominators
    )

    combined_probabilities = (
        view_probabilities
        * normalized_weights[
            :,
            :,
            None,
        ]
    ).sum(axis=1)

    # Chuẩn hóa lại để tránh sai số số học nhỏ.

    combined_probabilities = (
        combined_probabilities
        / combined_probabilities
        .sum(
            axis=1,
            keepdims=True,
        )
    )

    return combined_probabilities


print(
    "Cell 6 ready — four views use "
    "one transformer forward pass."
)

# === Cell 8 ===
# Cell 7 — Chuẩn bị dataset một lần cho mỗi model × task
def prepare_task_bundle(model_key, task_name):
    train_df, val_df, test_df = [x.copy() for x in task_cache[task_name]]
    class_names = sorted(train_df["label"].astype(str).unique().tolist())
    label_to_id = {label: i for i, label in enumerate(class_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    datasets = {
        "train": to_hf_dataset(train_df, model_key, label_to_id, f"{model_key}/{task_name}/train"),
        "validation": to_hf_dataset(val_df, model_key, label_to_id, f"{model_key}/{task_name}/validation"),
        "test": to_hf_dataset(test_df, model_key, label_to_id, f"{model_key}/{task_name}/test"),
    }
    train_ids = train_df["label"].astype(str).map(label_to_id).to_numpy()
    counts = np.bincount(train_ids, minlength=len(class_names))
    weights = len(train_df) / (len(class_names) * np.maximum(counts, 1))
    return {
        "frames": {"train": train_df, "validation": val_df, "test": test_df},
        "datasets": datasets,
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "class_weights": torch.tensor(weights, dtype=torch.float32),
    }


def training_arguments_compatible(**kwargs):
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    accepted = {k: v for k, v in kwargs.items() if k in signature}
    return TrainingArguments(**accepted)


def trainer_compatible(**kwargs):
    signature = inspect.signature(Trainer.__init__).parameters
    tokenizer = kwargs.pop("tokenizer_object")
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


# === Cell 9 ===
# Cell 8 — Train một model × task × seed và lưu dự đoán test

# Cấu hình dự phòng nếu Cell 2 chưa khai báo MODEL_RUNTIME_CONFIG.
# Effective batch size của cả hai model đều bằng 16:
# PhoBERT: 8 × 2 = 16
# XLM-R:   4 × 4 = 16
if "MODEL_RUNTIME_CONFIG" not in globals():
    MODEL_RUNTIME_CONFIG = {
        "phobert": {
            "train_batch_size": 8,
            "eval_batch_size": 16,
            "gradient_accumulation": 2,
            "learning_rate": 2e-5,
        },
        "xlmr": {
            "train_batch_size": 4,
            "eval_batch_size": 8,
            "gradient_accumulation": 4,
            "learning_rate": 2e-5,
        },
    }


def upsert_metric_row(row):
    """Thêm hoặc cập nhật kết quả của một model × task × seed."""
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
    else:
        metrics = pd.DataFrame()

    new_row = pd.DataFrame([row])

    if not metrics.empty:
        same = (
            metrics["model_key"].astype(str).eq(str(row["model_key"]))
            & metrics["task"].astype(str).eq(str(row["task"]))
            & metrics["seed"].astype(int).eq(int(row["seed"]))
        )
        metrics = metrics.loc[~same].copy()

    metrics = pd.concat(
        [metrics, new_row],
        ignore_index=True,
        sort=False,
    )

    metrics = (
        metrics
        .sort_values(["model_key", "task", "seed"])
        .reset_index(drop=True)
    )

    metrics.to_csv(METRICS_PATH, index=False)


def get_amp_settings():
    """
    Ưu tiên BF16 nếu GPU hỗ trợ.
    Nếu không hỗ trợ BF16 thì dùng FP16.
    """
    if not torch.cuda.is_available():
        return {
            "bf16": False,
            "fp16": False,
            "amp_name": "float32",
        }

    bf16_supported = bool(
        hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    return {
        "bf16": bf16_supported,
        "fp16": not bf16_supported,
        "amp_name": "bfloat16" if bf16_supported else "float16",
    }


def run_one_seed_task(model_key, task_name, seed, bundle):
    spec = MODEL_SPECS[model_key]
    runtime = MODEL_RUNTIME_CONFIG[model_key]
    amp = get_amp_settings()

    train_batch_size = int(runtime["train_batch_size"])
    eval_batch_size = int(runtime["eval_batch_size"])
    gradient_accumulation = int(runtime["gradient_accumulation"])
    learning_rate = float(runtime.get("learning_rate", LEARNING_RATE))

    effective_batch_size = (
        train_batch_size * gradient_accumulation
    )

    print("\n" + "#" * 88)
    print(
        spec["display_name"],
        "|",
        TASK_DISPLAY[task_name],
        "| seed",
        seed,
    )
    print("#" * 88)

    print(
        f"Train batch={train_batch_size} | "
        f"Gradient accumulation={gradient_accumulation} | "
        f"Effective batch={effective_batch_size}"
    )
    print(
        f"Eval batch={eval_batch_size} | "
        f"Precision={amp['amp_name']} | "
        f"Learning rate={learning_rate}"
    )

    set_all_seeds(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    class_names = bundle["class_names"]

    train_ds = bundle["datasets"]["train"]
    val_ds = bundle["datasets"]["validation"]
    test_ds = bundle["datasets"]["test"]

    train_df = bundle["frames"]["train"]
    val_df = bundle["frames"]["validation"]
    test_df = bundle["frames"]["test"]

    base_classifier = AutoModelForSequenceClassification.from_pretrained(
        spec["checkpoint"],
        num_labels=len(class_names),
        id2label=bundle["id_to_label"],
        label2id=bundle["label_to_id"],
    )

    model = SharedFourViewClassifier(
        classifier=base_classifier,
        class_weights=bundle["class_weights"],
        default_weights=TRAINING_VIEW_WEIGHTS,
    )

    run_dir = (
        EXPERIMENT_DIR
        / "runs"
        / model_key
        / f"seed_{seed}"
        / task_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Số optimizer step trong mỗi epoch.
    batches_per_epoch = math.ceil(
        len(train_ds) / train_batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation
    )
    total_optimizer_steps = (
        optimizer_steps_per_epoch * NUM_EPOCHS
    )
    warmup_steps = round(
        total_optimizer_steps * WARMUP_RATIO
    )

    training_kwargs = {
        "output_dir": str(run_dir / "checkpoints"),

        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,

        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": warmup_steps,

        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 20,

        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 1,

        "bf16": amp["bf16"],
        "fp16": amp["fp16"],
        "tf32": bool(torch.cuda.is_available()),

        # Tối ưu đường truyền CPU → GPU.
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": bool(torch.cuda.is_available()),
        "dataloader_persistent_workers": True,

        # Không bật gradient checkpointing vì GPU còn đủ bộ nhớ.
        "gradient_checkpointing": False,

        # Fused AdamW nhanh hơn trên CUDA.
        "optim": (
            "adamw_torch_fused"
            if torch.cuda.is_available()
            else "adamw_torch"
        ),

        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "remove_unused_columns": True,
    }

    args = training_arguments_compatible(
        **training_kwargs
    )

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        pred_ids = np.argmax(logits, axis=-1)
        return metric_row(labels, pred_ids)

    trainer = trainer_compatible(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer_object=get_tokenizer(model_key),
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    train_output = trainer.train()

    # ---------------------------------------------------------
    # Validation audit với weight đã khóa
    # ---------------------------------------------------------
    # Weight được chọn từ validation của lượt chạy chính thức
    # và được giữ nguyên qua tất cả random seed.
    val_outputs = predict_view_probabilities(
        trainer.model,
        val_ds,
        batch_size=eval_batch_size,
    )

    selected_weights = (
        LOCKED_INFERENCE_WEIGHTS[model_key][task_name].copy()
    )

    val_probabilities = combine_view_probabilities(
        view_probabilities=val_outputs["view_probabilities"],
        continuation_available=val_outputs["continuation_available"],
        tail_available=val_outputs["tail_available"],
        weights=selected_weights,
    )

    val_pred_ids = val_probabilities.argmax(axis=1)

    validation_audit = pd.DataFrame([
        {
            "model_key": model_key,
            "model": spec["display_name"],
            "task": task_name,
            "report_name": TASK_DISPLAY[task_name],
            "seed": seed,

            **{
                f"{view}_weight": selected_weights[view]
                for view in VIEW_NAMES
            },

            **metric_row(
                val_outputs["true_ids"],
                val_pred_ids,
            ),

            "weight_source": WEIGHT_REFERENCE[model_key],
            "weights_locked_across_seeds": True,
            "weights_reselected_for_seed": False,
        }
    ])

    validation_audit.to_csv(
        run_dir / "locked_weight_validation_audit.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Đánh giá test
    # ---------------------------------------------------------
    test_outputs = predict_view_probabilities(
        trainer.model,
        test_ds,
        batch_size=eval_batch_size,
    )

    test_probabilities = combine_view_probabilities(
        view_probabilities=test_outputs["view_probabilities"],
        continuation_available=test_outputs["continuation_available"],
        tail_available=test_outputs["tail_available"],
        weights=selected_weights,
    )

    true_ids = test_outputs["true_ids"].astype(int)
    pred_ids = test_probabilities.argmax(axis=1)

    scores = metric_row(
        true_ids,
        pred_ids,
    )

    predictions = pd.DataFrame({
        "_row_key": (
            test_df["_row_key"]
            .astype(int)
            .to_numpy()
        ),
        "article_id": (
            test_df["article_id"]
            .astype(str)
            .to_numpy()
        ),
        "true_label": [
            class_names[i]
            for i in true_ids
        ],
        "predicted_label": [
            class_names[i]
            for i in pred_ids
        ],
        "confidence": test_probabilities.max(axis=1),
        "is_error": pred_ids != true_ids,
    })

    for class_id, class_name in enumerate(class_names):
        safe_name = re.sub(
            r"[^0-9A-Za-zÀ-ỹ]+",
            "_",
            str(class_name),
        ).strip("_")

        predictions[f"prob_{safe_name}"] = (
            test_probabilities[:, class_id]
        )

    predictions.to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Theo dõi mức sử dụng GPU
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        peak_gpu_memory_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )
        peak_gpu_reserved_gb = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )
    else:
        peak_gpu_memory_gb = 0.0
        peak_gpu_reserved_gb = 0.0

    print(
        f"Test macro-F1: {scores['f1_macro']:.4f} | "
        f"Accuracy: {scores['accuracy']:.4f}"
    )

    if torch.cuda.is_available():
        print(
            f"Peak GPU allocated: "
            f"{peak_gpu_memory_gb:.2f} GB | "
            f"reserved: {peak_gpu_reserved_gb:.2f} GB"
        )

    # ---------------------------------------------------------
    # Lưu checkpoint nếu được yêu cầu
    # ---------------------------------------------------------
    if KEEP_CHECKPOINTS:
        model_dir = run_dir / "best_model"

        trainer.model.base_classifier.save_pretrained(
            str(model_dir)
        )
        get_tokenizer(model_key).save_pretrained(
            str(model_dir)
        )

    row = {
        "model_key": model_key,
        "model": spec["display_name"],
        "checkpoint": spec["checkpoint"],
        "text_mode": spec["text_mode"],

        "task": task_name,
        "report_name": TASK_DISPLAY[task_name],
        "seed": seed,
        "status": "completed",

        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "n_classes": len(class_names),

        "train_fingerprint": article_set_fingerprint(train_df),
        "validation_fingerprint": article_set_fingerprint(val_df),
        "test_fingerprint": article_set_fingerprint(test_df),

        **scores,

        **{
            f"{view}_weight": selected_weights[view]
            for view in VIEW_NAMES
        },

        "weight_source": WEIGHT_REFERENCE[model_key],
        "weights_locked_across_seeds": True,
        "weights_reselected_for_seed": False,

        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "precision": amp["amp_name"],

        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "warmup_steps": warmup_steps,

        "peak_gpu_memory_gb": round(
            peak_gpu_memory_gb,
            4,
        ),
        "peak_gpu_reserved_gb": round(
            peak_gpu_reserved_gb,
            4,
        ),

        "training_loss": float(
            train_output.training_loss
        ),
        "run_dir": str(run_dir),
    }

    with open(
        run_dir / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            row,
            file,
            ensure_ascii=False,
            indent=2,
        )

    upsert_metric_row(row)

    # Xóa checkpoint trung gian để tránh đầy Google Drive.
    if not KEEP_CHECKPOINTS:
        checkpoint_dir = run_dir / "checkpoints"

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

    del trainer
    del model
    del base_classifier

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row

# === Cell 10 ===
# Cell 9 — Chạy/resume toàn bộ 36 lượt theo thứ tự tiết kiệm thời gian tokenize
if METRICS_PATH.exists():
    existing_metrics = pd.read_csv(METRICS_PATH)
else:
    existing_metrics = pd.DataFrame(columns=["model_key", "task", "seed", "status"])

completed = {
    (str(r.model_key), str(r.task), int(r.seed))
    for r in existing_metrics.itertuples()
    if str(getattr(r, "status", "")) == "completed"
}

plan_rows = []
for model_key in MODEL_SPECS:
    for task_name in TASKS_TO_RUN:
        for seed in MULTI_SEEDS:
            plan_rows.append({
                "model_key": model_key,
                "task": task_name,
                "report_name": TASK_DISPLAY[task_name],
                "seed": seed,
                "status": "completed" if (model_key, task_name, seed) in completed else "pending",
            })

run_plan = pd.DataFrame(plan_rows)
display(run_plan.groupby(["model_key", "status"]).size().rename("n").reset_index())

failures = []
if RUN_TRAINING:
    for model_key in MODEL_SPECS:
        for task_name in TASKS_TO_RUN:
            pending_seeds = [
                seed for seed in MULTI_SEEDS
                if (model_key, task_name, seed) not in completed
            ]
            if not pending_seeds:
                print("SKIP completed:", model_key, task_name)
                continue

            bundle = prepare_task_bundle(model_key, task_name)
            for seed in pending_seeds:
                try:
                    run_one_seed_task(model_key, task_name, seed, bundle)
                    completed.add((model_key, task_name, seed))
                except Exception as exc:
                    failure = {
                        "model_key": model_key,
                        "task": task_name,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    pd.DataFrame(failures).to_csv(EXPERIMENT_DIR / "failures.csv", index=False)
                    print("FAILED:", failure)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del bundle
            gc.collect()

all_metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
display(all_metrics[["model", "report_name", "seed", "f1_macro", "accuracy"]])


# === Cell 11 ===
# Cell 10 — Tóm tắt mean, SD, min–max đúng sáu task
all_metrics = pd.read_csv(METRICS_PATH)
expected_runs = len(MODEL_SPECS) * len(TASKS_TO_RUN) * len(MULTI_SEEDS)
actual_keys = all_metrics[["model_key", "task", "seed"]].drop_duplicates()
print(f"Completed unique runs: {len(actual_keys)}/{expected_runs}")

summary = (
    all_metrics
    .groupby(["model_key", "model", "task", "report_name"], as_index=False)
    .agg(
        n_seeds=("seed", "nunique"),
        macro_f1_mean=("f1_macro", "mean"),
        macro_f1_sd=("f1_macro", "std"),
        macro_f1_min=("f1_macro", "min"),
        macro_f1_max=("f1_macro", "max"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
    )
)
summary["macro_f1_mean_sd"] = summary.apply(
    lambda r: f"{r.macro_f1_mean:.4f} ± {r.macro_f1_sd:.4f}", axis=1
)
summary.to_csv(EXPERIMENT_DIR / "summary_model_task.csv", index=False)

report_order = {name: i for i, name in enumerate(TASK_DISPLAY.values())}
summary["_order"] = summary["report_name"].map(report_order)
summary = summary.sort_values(["_order", "model_key"]).drop(columns="_order")
display(summary[[
    "report_name", "model", "n_seeds", "macro_f1_mean_sd",
    "macro_f1_min", "macro_f1_max", "accuracy_mean",
]])


# === Cell 12 ===
# Cell 11 — So sánh paired theo seed và paired bootstrap theo từng test article
def paired_bootstrap_macro_f1(y_true, pred_phobert, pred_xlmr, iterations, seed):
    y_true = np.asarray(y_true)
    pred_phobert = np.asarray(pred_phobert)
    pred_xlmr = np.asarray(pred_xlmr)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        f1_p = f1_score(y_true[idx], pred_phobert[idx], average="macro", zero_division=0)
        f1_x = f1_score(y_true[idx], pred_xlmr[idx], average="macro", zero_division=0)
        deltas[i] = f1_x - f1_p
    return {
        "delta_xlmr_minus_phobert": float(
            f1_score(y_true, pred_xlmr, average="macro", zero_division=0)
            - f1_score(y_true, pred_phobert, average="macro", zero_division=0)
        ),
        "bootstrap_ci_low": float(np.quantile(deltas, 0.025)),
        "bootstrap_ci_high": float(np.quantile(deltas, 0.975)),
        "prob_delta_gt_zero": float(np.mean(deltas > 0)),
    }


paired_seed = (
    all_metrics.pivot_table(
        index=["task", "report_name", "seed"],
        columns="model_key",
        values="f1_macro",
        aggfunc="first",
    )
    .dropna(subset=["phobert", "xlmr"])
    .reset_index()
)
paired_seed["delta_xlmr_minus_phobert"] = paired_seed["xlmr"] - paired_seed["phobert"]
paired_seed.to_csv(EXPERIMENT_DIR / "paired_seed_macro_f1.csv", index=False)

bootstrap_rows = []
for task_name in TASKS_TO_RUN:
    for seed in MULTI_SEEDS:
        paths = {
            model_key: EXPERIMENT_DIR / "runs" / model_key / f"seed_{seed}" / task_name / "test_predictions.csv"
            for model_key in ["phobert", "xlmr"]
        }
        if not all(path.exists() for path in paths.values()):
            continue
        p = pd.read_csv(paths["phobert"])
        x = pd.read_csv(paths["xlmr"])
        paired = p[["_row_key", "true_label", "predicted_label"]].merge(
            x[["_row_key", "true_label", "predicted_label"]],
            on="_row_key",
            suffixes=("_phobert", "_xlmr"),
            validate="one_to_one",
        )
        if not paired["true_label_phobert"].equals(paired["true_label_xlmr"]):
            raise AssertionError(f"True labels lệch giữa hai model: {task_name}, seed {seed}")

        labels = sorted(set(paired["true_label_phobert"].astype(str)))
        label_to_id = {label: i for i, label in enumerate(labels)}
        y = paired["true_label_phobert"].astype(str).map(label_to_id).to_numpy()
        pp = paired["predicted_label_phobert"].astype(str).map(label_to_id).to_numpy()
        px = paired["predicted_label_xlmr"].astype(str).map(label_to_id).to_numpy()
        if np.isnan(pp).any() or np.isnan(px).any():
            raise ValueError("Prediction chứa label ngoài class set.")
        pp, px = pp.astype(int), px.astype(int)

        result = paired_bootstrap_macro_f1(
            y, pp, px, BOOTSTRAP_ITERATIONS,
            BOOTSTRAP_SEED + seed + 1000 * TASKS_TO_RUN.index(task_name),
        )
        correct_p, correct_x = pp == y, px == y
        p_only = int(np.sum(correct_p & ~correct_x))
        x_only = int(np.sum(~correct_p & correct_x))
        discordant = p_only + x_only
        mcnemar_p = 1.0 if discordant == 0 else binomtest(
            min(p_only, x_only), n=discordant, p=0.5, alternative="two-sided"
        ).pvalue
        bootstrap_rows.append({
            "task": task_name,
            "report_name": TASK_DISPLAY[task_name],
            "seed": seed,
            "n_test": len(paired),
            **result,
            "phobert_only_correct": p_only,
            "xlmr_only_correct": x_only,
            "mcnemar_exact_p": mcnemar_p,
        })

bootstrap_results = pd.DataFrame(bootstrap_rows)
bootstrap_results.to_csv(EXPERIMENT_DIR / "paired_bootstrap_by_task_seed.csv", index=False)

comparison = (
    paired_seed.groupby(["task", "report_name"], as_index=False)
    .agg(
        n_paired_seeds=("seed", "nunique"),
        phobert_macro_f1_mean=("phobert", "mean"),
        phobert_macro_f1_sd=("phobert", "std"),
        xlmr_macro_f1_mean=("xlmr", "mean"),
        xlmr_macro_f1_sd=("xlmr", "std"),
        delta_mean=("delta_xlmr_minus_phobert", "mean"),
        delta_sd=("delta_xlmr_minus_phobert", "std"),
        xlmr_seed_wins=("delta_xlmr_minus_phobert", lambda x: int((x > 0).sum())),
        phobert_seed_wins=("delta_xlmr_minus_phobert", lambda x: int((x < 0).sum())),
        ties=("delta_xlmr_minus_phobert", lambda x: int((x == 0).sum())),
    )
)
comparison["winner_by_mean"] = np.select(
    [comparison["delta_mean"] > 0, comparison["delta_mean"] < 0],
    ["XLM-RoBERTa", "PhoBERT"],
    default="Tie",
)
comparison.to_csv(EXPERIMENT_DIR / "final_6task_model_comparison.csv", index=False)
display(comparison)
display(bootstrap_results)


# === Cell 13 ===
# Cell 12 — Xuất workbook báo cáo và biểu đồ tập trung vào sáu task
report_path = EXPERIMENT_DIR / "Randomseed_PhoBERT_XLMR_6tasks_report.xlsx"
with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
    comparison.to_excel(writer, sheet_name="Final comparison", index=False)
    summary.to_excel(writer, sheet_name="Model task summary", index=False)
    paired_seed.to_excel(writer, sheet_name="Paired seeds", index=False)
    bootstrap_results.to_excel(writer, sheet_name="Paired bootstrap", index=False)
    all_metrics.to_excel(writer, sheet_name="All runs", index=False)
    task_data_audit.to_excel(writer, sheet_name="Data audit", index=False)
    locked_weights_audit.to_excel(writer, sheet_name="Locked weights", index=False)

plot_data = summary.copy()
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
sns.barplot(
    data=plot_data,
    x="report_name", y="macro_f1_mean", hue="model_key",
    order=list(TASK_DISPLAY.values()), ax=axes[0],
)
axes[0].set_title("Mean Macro-F1 across seeds")
axes[0].set_xlabel("")
axes[0].set_ylabel("Macro-F1")
axes[0].set_ylim(0, 1)
axes[0].legend(title="Model")

delta_plot = comparison.copy()
delta_plot["color"] = np.where(delta_plot["delta_mean"] >= 0, "#2B6CB0", "#C05621")
axes[1].bar(delta_plot["report_name"], delta_plot["delta_mean"], color=delta_plot["color"])
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_title("Mean Macro-F1 difference")
axes[1].set_xlabel("")
axes[1].set_ylabel("XLM-RoBERTa − PhoBERT")
for axis in axes:
    axis.tick_params(axis="x", rotation=25)
plt.tight_layout()
figure_path = EXPERIMENT_DIR / "final_6task_model_comparison.png"
plt.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()

completion = pd.DataFrame([
    {"check": "All planned runs completed", "value": len(actual_keys), "target": expected_runs,
     "status": "PASS" if len(actual_keys) == expected_runs else "INCOMPLETE"},
    {"check": "Six requested tasks only", "value": all_metrics["task"].nunique(), "target": 6,
     "status": "PASS" if set(all_metrics["task"]) == set(TASKS_TO_RUN) else "FAIL"},
    {"check": "Both models present", "value": all_metrics["model_key"].nunique(), "target": 2,
     "status": "PASS" if set(all_metrics["model_key"]) == set(MODEL_SPECS) else "FAIL"},
    {"check": "Weights locked by model and task", "value": len(locked_weights_audit), "target": 12,
     "status": "PASS" if len(locked_weights_audit) == 12 else "FAIL"},
])
completion.to_csv(EXPERIMENT_DIR / "completion_audit.csv", index=False)
display(completion)
print("Report:", report_path)
print("Figure:", figure_path)


# === Cell 14 ===
