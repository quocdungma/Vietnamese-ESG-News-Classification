# Original notebook code exported for archival review. Paths/environment require adaptation.


# === Cell 1 ===
# COLAB COMMAND: !pip -q install beautifulsoup4 lxml pyarrow tqdm requests

# === Cell 2 ===
from google.colab import drive

drive.mount("/content/gdrive", force_remount=True)

# === Cell 3 ===
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import hashlib
import json
import random
import re
import time

import pandas as pd
import requests

from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_ROOT = Path(
    "/content/gdrive/My Drive/Project_Fin_Data_Crawling"
)

DATA_DIR = PROJECT_ROOT / "Crawling_news" / "Data"
UPDATE_DIR = DATA_DIR / "incremental_updates"
CHECKPOINT_DIR = UPDATE_DIR / "checkpoints"

UPDATE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# File dữ liệu cũ dùng để xác định ngày và URL đã có
# Đổi đường dẫn này nếu file master của bạn có tên khác.
MASTER_FILE = DATA_DIR / "combined_clean_030725.csv"


# Ngày cập nhật
CRAWL_TO = pd.Timestamp.today().normalize()

# Lùi lại 7 ngày so với ngày cuối cùng của từng nguồn
OVERLAP_DAYS = 7

# Số luồng vừa phải để hạn chế bị chặn
DETAIL_WORKERS = 12

# Lưu checkpoint sau mỗi số bài này
CHECKPOINT_EVERY = 100

# Giới hạn an toàn số trang
CAFEF_MAX_PAGES = 500
VIETSTOCK_MAX_PAGES = 500

RUN_DATE = CRAWL_TO.strftime("%Y%m%d")
BATCH_ID = f"news_update_{RUN_DATE}_{datetime.now():%H%M%S}"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("MASTER_FILE:", MASTER_FILE)
print("CRAWL_TO:", CRAWL_TO.date())
print("BATCH_ID:", BATCH_ID)

# === Cell 4 ===
def crawl_one_cafef_category(
    category,
    content_id,
    start_date,
    end_date,
    max_pages=None,
    max_empty_pages=2,
    max_repeated_pages=3,
):
    """
    Cào một chuyên mục CafeF.

    max_pages=None:
        Không giới hạn cứng số trang. Crawler tự dừng theo ngày,
        trang trống hoặc trang bị lặp.
    """
    session = create_session()

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    category_links = []
    page = 1
    empty_pages = 0
    repeated_pages = 0
    previous_page_urls = None

    while True:
        # Chỉ áp dụng giới hạn nếu người dùng truyền max_pages
        if max_pages is not None and page > max_pages:
            print(
                f"CafeF — {category}: đạt giới hạn "
                f"{max_pages} trang."
            )
            break

        articles, status = fetch_cafef_listing_page(
            session=session,
            content_id=content_id,
            category=category,
            page=page,
        )

        if not articles:
            empty_pages += 1

            print(
                f"CafeF — {category} — trang {page}: "
                f"không có dữ liệu ({status}); "
                f"empty={empty_pages}/{max_empty_pages}"
            )

            if empty_pages >= max_empty_pages:
                break

            page += 1
            time.sleep(random.uniform(0.8, 1.5))
            continue

        empty_pages = 0

        # --------------------------------------------------------
        # Kiểm tra CafeF có trả lặp lại cùng một trang hay không
        # --------------------------------------------------------
        current_page_urls = tuple(
            sorted(
                row["url_normalized"]
                for row in articles
                if row.get("url_normalized")
            )
        )

        if current_page_urls == previous_page_urls:
            repeated_pages += 1
        else:
            repeated_pages = 0

        previous_page_urls = current_page_urls

        if repeated_pages >= max_repeated_pages:
            print(
                f"CafeF — {category}: nội dung trang bị lặp "
                f"{max_repeated_pages} lần; dừng tại trang {page}."
            )
            break

        # --------------------------------------------------------
        # Xác định ngày của trang hiện tại
        # --------------------------------------------------------
        valid_dates = [
            pd.Timestamp(row["published_at"])
            for row in articles
            if not pd.isna(row.get("published_at"))
        ]

        articles_in_range = [
            row
            for row in articles
            if (
                pd.isna(row.get("published_at"))
                or (
                    start_date
                    <= pd.Timestamp(
                        row["published_at"]
                    ).normalize()
                    <= end_date
                )
            )
        ]

        category_links.extend(articles_in_range)

        oldest_date = (
            min(valid_dates)
            if valid_dates
            else pd.NaT
        )

        newest_date = (
            max(valid_dates)
            if valid_dates
            else pd.NaT
        )

        print(
            f"CafeF — {category} — trang {page}: "
            f"tìm thấy={len(articles)}, "
            f"giữ={len(articles_in_range)}, "
            f"mới nhất={newest_date}, "
            f"cũ nhất={oldest_date}"
        )

        # Timeline CafeF đi từ mới đến cũ.
        # Khi bài cũ nhất của trang đã nhỏ hơn start_date,
        # không cần tải thêm trang.
        if (
            not pd.isna(oldest_date)
            and oldest_date.normalize() < start_date
        ):
            print(
                f"CafeF — {category}: đã đi qua "
                f"start_date={start_date.date()}; dừng."
            )
            break

        page += 1
        time.sleep(random.uniform(0.4, 0.9))

    session.close()

    return {
        "category": category,
        "content_id": content_id,
        "links": category_links,
        "last_page": page,
    }


def crawl_cafef_links_by_date(
    start_date,
    end_date,
    max_pages=None,
    category_workers=4,
):
    """
    Cào đồng thời các chuyên mục CafeF.

    max_pages=None:
        Không giới hạn cứng số trang.

    category_workers:
        Số chuyên mục được cào đồng thời.
        Nên dùng 3–5 để tránh gửi quá nhiều request.
    """
    all_links = []
    category_summaries = []

    with ThreadPoolExecutor(
        max_workers=category_workers
    ) as executor:

        futures = {
            executor.submit(
                crawl_one_cafef_category,
                category=category,
                content_id=content_id,
                start_date=start_date,
                end_date=end_date,
                max_pages=max_pages,
            ): category
            for category, content_id
            in CAFEF_CATEGORIES.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="CafeF categories",
        ):
            category = futures[future]

            try:
                result = future.result()

                all_links.extend(result["links"])

                category_summaries.append(
                    {
                        "category": category,
                        "content_id": result["content_id"],
                        "n_links_raw": len(result["links"]),
                        "last_page": result["last_page"],
                        "status": "success",
                    }
                )

            except Exception as error:
                print(
                    f"\nLỗi chuyên mục {category}: "
                    f"{type(error).__name__}: {error}"
                )

                category_summaries.append(
                    {
                        "category": category,
                        "content_id": (
                            CAFEF_CATEGORIES[category]
                        ),
                        "n_links_raw": 0,
                        "last_page": None,
                        "status": (
                            f"error:{type(error).__name__}"
                        ),
                    }
                )

    links_df = pd.DataFrame(all_links)
    summary_df = pd.DataFrame(category_summaries)

    if links_df.empty:
        return links_df, summary_df

    links_df = (
        links_df
        .dropna(subset=["url_normalized"])
        .sort_values(
            ["published_at", "discovered_at"],
            na_position="last",
        )
        .drop_duplicates(
            subset=["url_normalized"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return links_df, summary_df

# === Cell 6 ===
def crawl_one_cafef_category(
    category,
    content_id,
    start_date,
    end_date,
    max_pages=None,
    max_empty_pages=2,
    max_repeated_pages=3,
):
    """
    Cào một chuyên mục CafeF.

    max_pages=None:
        Không giới hạn cứng số trang. Crawler tự dừng theo ngày,
        trang trống hoặc trang bị lặp.
    """
    session = create_session()

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    category_links = []
    page = 1
    empty_pages = 0
    repeated_pages = 0
    previous_page_urls = None

    while True:
        # Chỉ áp dụng giới hạn nếu người dùng truyền max_pages
        if max_pages is not None and page > max_pages:
            print(
                f"CafeF — {category}: đạt giới hạn "
                f"{max_pages} trang."
            )
            break

        articles, status = fetch_cafef_listing_page(
            session=session,
            content_id=content_id,
            category=category,
            page=page,
        )

        if not articles:
            empty_pages += 1

            print(
                f"CafeF — {category} — trang {page}: "
                f"không có dữ liệu ({status}); "
                f"empty={empty_pages}/{max_empty_pages}"
            )

            if empty_pages >= max_empty_pages:
                break

            page += 1
            time.sleep(random.uniform(0.8, 1.5))
            continue

        empty_pages = 0

        # --------------------------------------------------------
        # Kiểm tra CafeF có trả lặp lại cùng một trang hay không
        # --------------------------------------------------------
        current_page_urls = tuple(
            sorted(
                row["url_normalized"]
                for row in articles
                if row.get("url_normalized")
            )
        )

        if current_page_urls == previous_page_urls:
            repeated_pages += 1
        else:
            repeated_pages = 0

        previous_page_urls = current_page_urls

        if repeated_pages >= max_repeated_pages:
            print(
                f"CafeF — {category}: nội dung trang bị lặp "
                f"{max_repeated_pages} lần; dừng tại trang {page}."
            )
            break

        # --------------------------------------------------------
        # Xác định ngày của trang hiện tại
        # --------------------------------------------------------
        valid_dates = [
            pd.Timestamp(row["published_at"])
            for row in articles
            if not pd.isna(row.get("published_at"))
        ]

        articles_in_range = [
            row
            for row in articles
            if (
                pd.isna(row.get("published_at"))
                or (
                    start_date
                    <= pd.Timestamp(
                        row["published_at"]
                    ).normalize()
                    <= end_date
                )
            )
        ]

        category_links.extend(articles_in_range)

        oldest_date = (
            min(valid_dates)
            if valid_dates
            else pd.NaT
        )

        newest_date = (
            max(valid_dates)
            if valid_dates
            else pd.NaT
        )

        print(
            f"CafeF — {category} — trang {page}: "
            f"tìm thấy={len(articles)}, "
            f"giữ={len(articles_in_range)}, "
            f"mới nhất={newest_date}, "
            f"cũ nhất={oldest_date}"
        )

        # Timeline CafeF đi từ mới đến cũ.
        # Khi bài cũ nhất của trang đã nhỏ hơn start_date,
        # không cần tải thêm trang.
        if (
            not pd.isna(oldest_date)
            and oldest_date.normalize() < start_date
        ):
            print(
                f"CafeF — {category}: đã đi qua "
                f"start_date={start_date.date()}; dừng."
            )
            break

        page += 1
        time.sleep(random.uniform(0.4, 0.9))

    session.close()

    return {
        "category": category,
        "content_id": content_id,
        "links": category_links,
        "last_page": page,
    }


def crawl_cafef_links_by_date(
    start_date,
    end_date,
    max_pages=None,
    category_workers=4,
):
    """
    Cào đồng thời các chuyên mục CafeF.

    max_pages=None:
        Không giới hạn cứng số trang.

    category_workers:
        Số chuyên mục được cào đồng thời.
        Nên dùng 3–5 để tránh gửi quá nhiều request.
    """
    all_links = []
    category_summaries = []

    with ThreadPoolExecutor(
        max_workers=category_workers
    ) as executor:

        futures = {
            executor.submit(
                crawl_one_cafef_category,
                category=category,
                content_id=content_id,
                start_date=start_date,
                end_date=end_date,
                max_pages=max_pages,
            ): category
            for category, content_id
            in CAFEF_CATEGORIES.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="CafeF categories",
        ):
            category = futures[future]

            try:
                result = future.result()

                all_links.extend(result["links"])

                category_summaries.append(
                    {
                        "category": category,
                        "content_id": result["content_id"],
                        "n_links_raw": len(result["links"]),
                        "last_page": result["last_page"],
                        "status": "success",
                    }
                )

            except Exception as error:
                print(
                    f"\nLỗi chuyên mục {category}: "
                    f"{type(error).__name__}: {error}"
                )

                category_summaries.append(
                    {
                        "category": category,
                        "content_id": (
                            CAFEF_CATEGORIES[category]
                        ),
                        "n_links_raw": 0,
                        "last_page": None,
                        "status": (
                            f"error:{type(error).__name__}"
                        ),
                    }
                )

    links_df = pd.DataFrame(all_links)
    summary_df = pd.DataFrame(category_summaries)

    if links_df.empty:
        return links_df, summary_df

    links_df = (
        links_df
        .dropna(subset=["url_normalized"])
        .sort_values(
            ["published_at", "discovered_at"],
            na_position="last",
        )
        .drop_duplicates(
            subset=["url_normalized"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return links_df, summary_df

# === Cell 8 ===
def find_first_existing_column(columns, candidates):
    columns_lower = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    return None


def read_old_metadata(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print("Không tìm thấy master cũ:", file_path)
        return pd.DataFrame(
            columns=["source", "url_normalized", "published_at"]
        )

    if file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(
            file_path,
            low_memory=False,
            on_bad_lines="skip",
        )

    url_col = find_first_existing_column(
        df.columns,
        ["url_normalized", "url", "link"],
    )

    date_col = find_first_existing_column(
        df.columns,
        ["published_at", "published_date", "date", "pub_date"],
    )

    source_col = find_first_existing_column(
        df.columns,
        ["source", "news_source", "site"],
    )

    category_col = find_first_existing_column(
        df.columns,
        ["category", "section"],
    )

    metadata = pd.DataFrame(index=df.index)

    if url_col:
        metadata["url_normalized"] = (
            df[url_col].map(normalize_url)
        )
    else:
        metadata["url_normalized"] = None

    if date_col:
        metadata["published_at"] = (
            df[date_col].map(parse_vietnamese_datetime)
        )
    else:
        metadata["published_at"] = pd.NaT

    if source_col:
        metadata["source"] = (
            df[source_col]
            .astype(str)
            .str.lower()
            .str.strip()
        )
    else:
        # Suy luận nguồn từ URL nếu master không có cột source
        metadata["source"] = metadata["url_normalized"].apply(
            lambda x: (
                "cafef"
                if isinstance(x, str) and "cafef.vn" in x
                else "vietstock"
                if isinstance(x, str) and "vietstock.vn" in x
                else "unknown"
            )
        )

    if category_col:
        metadata["category"] = df[category_col]
    else:
        metadata["category"] = None

    metadata = metadata.dropna(
        subset=["url_normalized"]
    ).drop_duplicates(
        subset=["url_normalized"],
        keep="last",
    )

    return metadata.reset_index(drop=True)


old_metadata = read_old_metadata(MASTER_FILE)

print("Số URL cũ:", f"{len(old_metadata):,}")

display(
    old_metadata.groupby("source").agg(
        n_articles=("url_normalized", "nunique"),
        first_date=("published_at", "min"),
        last_date=("published_at", "max"),
    )
)

# === Cell 10 ===
DEFAULT_START_DATE = pd.Timestamp("2025-06-26")


def infer_crawl_start(source_name):
    source_data = old_metadata[
        old_metadata["source"]
        .astype(str)
        .str.contains(source_name, case=False, na=False)
    ]

    last_date = source_data["published_at"].max()

    if pd.isna(last_date):
        print(
            f"Không xác định được ngày cuối của {source_name}; "
            f"dùng mặc định {DEFAULT_START_DATE.date()}"
        )
        return DEFAULT_START_DATE

    start_date = (
        pd.Timestamp(last_date).normalize()
        - pd.Timedelta(days=OVERLAP_DAYS)
    )

    return start_date


CAFEF_FROM = infer_crawl_start("cafef")
VIETSTOCK_FROM = infer_crawl_start("vietstock")

OLD_URLS = set(
    old_metadata["url_normalized"]
    .dropna()
    .astype(str)
)

print("CafeF:", CAFEF_FROM.date(), "→", CRAWL_TO.date())
print("Vietstock:", VIETSTOCK_FROM.date(), "→", CRAWL_TO.date())
print("Số URL đã có:", f"{len(OLD_URLS):,}")

# === Cell 13 ===
CAFEF_CATEGORIES = {
    "vimo": 18833,
    "nganhang": 18834,
    "doanhnghiep": 18836,
    "chungkhoan": 18831,
    "batdongsan": 18835,
    "xahoi": 188112,
    "taichinhquocte": 18832,
    "hanghoa": 18839,
    "kinhteso": 188127,
}


def fetch_cafef_listing_page(
    session,
    content_id,
    category,
    page,
):
    url = (
        f"https://cafef.vn/timelinelist/"
        f"{content_id}/{page}.chn"
    )

    try:
        response = session.get(
            url,
            timeout=30,
            headers={"Referer": "https://cafef.vn/"},
        )

        if response.status_code != 200:
            return [], f"http_{response.status_code}"

        soup = BeautifulSoup(
            response.content,
            "lxml",
        )

        items = soup.select(
            "div.tlitem.box-category-item"
        )

        articles = []

        for item in items:
            a_tag = item.select_one("h3 > a")

            if not a_tag or not a_tag.get("href"):
                continue

            href = a_tag.get("href").strip()

            if href.startswith("http"):
                link = href
            else:
                link = "https://cafef.vn" + href

            title = (
                a_tag.get("title")
                or a_tag.get_text(" ", strip=True)
            )

            sapo_tag = item.select_one(
                "p.sapo.box-category-sapo"
            )

            time_tag = item.select_one(
                "span.time.time-ago"
            )

            published_raw = (
                time_tag.get("title", "")
                if time_tag
                else ""
            )

            published_at = parse_vietnamese_datetime(
                published_raw
            )

            articles.append(
                {
                    "source": "cafef",
                    "category": category,
                    "content_id": content_id,
                    "listing_page": page,
                    "title_listing": clean_text(title),
                    "description_listing": clean_text(
                        sapo_tag.get_text(" ", strip=True)
                        if sapo_tag
                        else ""
                    ),
                    "published_raw": published_raw,
                    "published_at": published_at,
                    "url": link,
                    "url_normalized": normalize_url(link),
                    "discovered_at": pd.Timestamp.now(),
                    "crawl_batch_id": BATCH_ID,
                }
            )

        return articles, "success"

    except Exception as error:
        return [], f"request_error:{type(error).__name__}"


def crawl_one_cafef_category(
    category,
    content_id,
    start_date,
    end_date,
    max_pages=None,
    max_empty_pages=2,
    max_repeated_pages=3,
):
    """
    Cào một chuyên mục CafeF.

    max_pages=None:
        Không giới hạn cứng số trang. Crawler tự dừng theo ngày,
        trang trống hoặc trang bị lặp.
    """
    session = create_session()

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    category_links = []
    page = 1
    empty_pages = 0
    repeated_pages = 0
    previous_page_urls = None

    while True:
        # Chỉ áp dụng giới hạn nếu người dùng truyền max_pages
        if max_pages is not None and page > max_pages:
            print(
                f"CafeF — {category}: đạt giới hạn "
                f"{max_pages} trang."
            )
            break

        articles, status = fetch_cafef_listing_page(
            session=session,
            content_id=content_id,
            category=category,
            page=page,
        )

        if not articles:
            empty_pages += 1

            print(
                f"CafeF — {category} — trang {page}: "
                f"không có dữ liệu ({status}); "
                f"empty={empty_pages}/{max_empty_pages}"
            )

            if empty_pages >= max_empty_pages:
                break

            page += 1
            time.sleep(random.uniform(0.8, 1.5))
            continue

        empty_pages = 0

        # --------------------------------------------------------
        # Kiểm tra CafeF có trả lặp lại cùng một trang hay không
        # --------------------------------------------------------
        current_page_urls = tuple(
            sorted(
                row["url_normalized"]
                for row in articles
                if row.get("url_normalized")
            )
        )

        if current_page_urls == previous_page_urls:
            repeated_pages += 1
        else:
            repeated_pages = 0

        previous_page_urls = current_page_urls

        if repeated_pages >= max_repeated_pages:
            print(
                f"CafeF — {category}: nội dung trang bị lặp "
                f"{max_repeated_pages} lần; dừng tại trang {page}."
            )
            break

        # --------------------------------------------------------
        # Xác định ngày của trang hiện tại
        # --------------------------------------------------------
        valid_dates = [
            pd.Timestamp(row["published_at"])
            for row in articles
            if not pd.isna(row.get("published_at"))
        ]

        articles_in_range = [
            row
            for row in articles
            if (
                pd.isna(row.get("published_at"))
                or (
                    start_date
                    <= pd.Timestamp(
                        row["published_at"]
                    ).normalize()
                    <= end_date
                )
            )
        ]

        category_links.extend(articles_in_range)

        oldest_date = (
            min(valid_dates)
            if valid_dates
            else pd.NaT
        )

        newest_date = (
            max(valid_dates)
            if valid_dates
            else pd.NaT
        )

        print(
            f"CafeF — {category} — trang {page}: "
            f"tìm thấy={len(articles)}, "
            f"giữ={len(articles_in_range)}, "
            f"mới nhất={newest_date}, "
            f"cũ nhất={oldest_date}"
        )

        # Timeline CafeF đi từ mới đến cũ.
        # Khi bài cũ nhất của trang đã nhỏ hơn start_date,
        # không cần tải thêm trang.
        if (
            not pd.isna(oldest_date)
            and oldest_date.normalize() < start_date
        ):
            print(
                f"CafeF — {category}: đã đi qua "
                f"start_date={start_date.date()}; dừng."
            )
            break

        page += 1
        time.sleep(random.uniform(0.4, 0.9))

    session.close()

    return {
        "category": category,
        "content_id": content_id,
        "links": category_links,
        "last_page": page,
    }


def crawl_cafef_links_by_date(
    start_date,
    end_date,
    max_pages=None,
    category_workers=4,
):
    """
    Cào đồng thời các chuyên mục CafeF.

    max_pages=None:
        Không giới hạn cứng số trang.

    category_workers:
        Số chuyên mục được cào đồng thời.
        Nên dùng 3–5 để tránh gửi quá nhiều request.
    """
    all_links = []
    category_summaries = []

    with ThreadPoolExecutor(
        max_workers=category_workers
    ) as executor:

        futures = {
            executor.submit(
                crawl_one_cafef_category,
                category=category,
                content_id=content_id,
                start_date=start_date,
                end_date=end_date,
                max_pages=max_pages,
            ): category
            for category, content_id
            in CAFEF_CATEGORIES.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="CafeF categories",
        ):
            category = futures[future]

            try:
                result = future.result()

                all_links.extend(result["links"])

                category_summaries.append(
                    {
                        "category": category,
                        "content_id": result["content_id"],
                        "n_links_raw": len(result["links"]),
                        "last_page": result["last_page"],
                        "status": "success",
                    }
                )

            except Exception as error:
                print(
                    f"\nLỗi chuyên mục {category}: "
                    f"{type(error).__name__}: {error}"
                )

                category_summaries.append(
                    {
                        "category": category,
                        "content_id": (
                            CAFEF_CATEGORIES[category]
                        ),
                        "n_links_raw": 0,
                        "last_page": None,
                        "status": (
                            f"error:{type(error).__name__}"
                        ),
                    }
                )

    links_df = pd.DataFrame(all_links)
    summary_df = pd.DataFrame(category_summaries)

    if links_df.empty:
        return links_df, summary_df

    links_df = (
        links_df
        .dropna(subset=["url_normalized"])
        .sort_values(
            ["published_at", "discovered_at"],
            na_position="last",
        )
        .drop_duplicates(
            subset=["url_normalized"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return links_df, summary_df

# === Cell 14 ===
def get_cafef_article_detail(row):
    session = create_session()
    url = row["url"]

    base_result = row.to_dict()

    try:
        response = session.get(
            url,
            timeout=30,
            headers={"Referer": "https://cafef.vn/"},
        )

        base_result["http_status"] = (
            response.status_code
        )

        if response.status_code != 200:
            base_result["parse_status"] = (
                f"http_{response.status_code}"
            )
            return base_result

        soup = BeautifulSoup(
            response.content,
            "lxml",
        )

        title_tag = soup.select_one(
            "meta[property='og:title']"
        )

        title = (
            title_tag.get("content", "")
            if title_tag
            else ""
        )

        if not title:
            heading = soup.select_one("h1")
            title = (
                heading.get_text(" ", strip=True)
                if heading
                else row.get("title_listing", "")
            )

        sapo_tag = (
            soup.select_one("h2.sapo[data-role='sapo']")
            or soup.select_one("h2.sapo")
            or soup.select_one(".sapo")
        )

        content_div = (
            soup.select_one("div.detail-content.afcbc-body")
            or soup.select_one("div.detail-content")
            or soup.select_one("[data-role='content']")
        )

        if content_div:
            for unwanted in content_div.select(
                "script, style, iframe, "
                ".relate-container, .link-content-footer"
            ):
                unwanted.decompose()

            paragraphs = [
                clean_text(p.get_text(" ", strip=True))
                for p in content_div.select("p")
            ]

            paragraphs = [
                p for p in paragraphs
                if len(p) >= 20
            ]

            content = "\n".join(paragraphs)

            if not content:
                content = clean_text(
                    content_div.get_text("\n", strip=True)
                )
        else:
            content = ""

        author_tag = (
            soup.select_one(".author")
            or soup.select_one("[data-role='author']")
        )

        base_result.update(
            {
                "title": clean_text(title),
                "description": clean_text(
                    sapo_tag.get_text(" ", strip=True)
                    if sapo_tag
                    else row.get(
                        "description_listing", ""
                    )
                ),
                "content": clean_text(content),
                "author": clean_text(
                    author_tag.get_text(" ", strip=True)
                    if author_tag
                    else ""
                ),
                "crawl_at": pd.Timestamp.now(),
                "parse_status": (
                    "success"
                    if content
                    else "empty_content"
                ),
            }
        )

        base_result["article_id"] = make_article_id(
            base_result["url_normalized"]
        )

        base_result["content_hash"] = (
            make_content_hash(content)
            if content
            else None
        )

        return base_result

    except requests.Timeout:
        base_result["parse_status"] = "timeout"
        base_result["crawl_at"] = pd.Timestamp.now()
        return base_result

    except Exception as error:
        base_result["parse_status"] = (
            f"parse_error:{type(error).__name__}"
        )
        base_result["crawl_at"] = pd.Timestamp.now()
        return base_result

# === Cell 15 ===
cafef_links = crawl_cafef_links_by_date(
    start_date=CAFEF_FROM,
    end_date=CRAWL_TO,
    max_pages=CAFEF_MAX_PAGES,
)

print("\nSố link CafeF trong khoảng cập nhật:", len(cafef_links))

if not cafef_links.empty:
    # Cào URL mới và URL nằm trong khoảng overlap.
    # Việc cào lại overlap giúp nhận biết bài được chỉnh sửa.
    cafef_links["already_in_master"] = (
        cafef_links["url_normalized"].isin(OLD_URLS)
    )

    display(
        cafef_links[
            [
                "category",
                "published_at",
                "title_listing",
                "already_in_master",
            ]
        ].head()
    )

    save_parquet_safe(
        cafef_links,
        UPDATE_DIR / f"cafef_links_{RUN_DATE}.parquet",
    )

    cafef_results = []

    with ThreadPoolExecutor(
        max_workers=DETAIL_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                get_cafef_article_detail,
                row,
            ): index
            for index, row in cafef_links.iterrows()
        }

        for completed, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc="CafeF detail",
            ),
            start=1,
        ):
            result = future.result()
            cafef_results.append(result)

            if completed % CHECKPOINT_EVERY == 0:
                checkpoint_df = pd.DataFrame(
                    cafef_results
                )

                save_parquet_safe(
                    checkpoint_df,
                    CHECKPOINT_DIR
                    / f"cafef_checkpoint_{RUN_DATE}.parquet",
                )

    cafef_detail = pd.DataFrame(cafef_results)

    save_parquet_safe(
        cafef_detail,
        UPDATE_DIR
        / f"cafef_detail_increment_{RUN_DATE}.parquet",
    )

else:
    cafef_detail = pd.DataFrame()
    print("Không có bài CafeF trong khoảng cập nhật.")

# === Cell 16 ===
VIETSTOCK_CHANNELS = {
    "chung-khoan": 144,
    "doanh-nghiep": 733,
    "bat-dong-san": 763,
    "tai-chinh": 734,
    "hang-hoa": 2,
    "kinh-te": 5307,
    "the-gioi": 736,
    "phan-tich": 579,
    "dong-duong": 1317,
}


VIETSTOCK_API = (
    "https://vietstock.vn/"
    "StartPage/ChannelContentPage"
)


# Số chuyên mục chạy đồng thời.
# Khuyến nghị từ 3–5.
VIETSTOCK_CATEGORY_WORKERS = 4

# === Cell 17 ===
def fetch_vietstock_listing_page(
    session,
    channel_id,
    category,
    page,
    start_date,
    end_date,
):
    """
    Lấy một trang danh sách bài viết Vietstock.
    """

    payload = {
        "channelID": str(channel_id),
        "page": str(page),
        "fromdate": pd.Timestamp(
            start_date
        ).strftime("%Y-%m-%d"),
        "todate": pd.Timestamp(
            end_date
        ).strftime("%Y-%m-%d"),
    }

    headers = {
        "Content-Type": (
            "application/x-www-form-urlencoded; "
            "charset=UTF-8"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Referer": (
            f"https://vietstock.vn/"
            f"{category}.htm"
        ),
        "Origin": "https://vietstock.vn",
    }

    try:
        response = session.post(
            VIETSTOCK_API,
            headers=headers,
            data=payload,
            timeout=30,
        )

        if response.status_code != 200:
            return [], f"http_{response.status_code}"

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        articles = []

        for h4 in soup.find_all("h4"):
            a_tag = h4.find("a")

            if not a_tag:
                continue

            href = a_tag.get("href")

            if not href:
                continue

            href = href.strip()

            if href.startswith("http"):
                link = href
            elif href.startswith("/"):
                link = "https://vietstock.vn" + href
            else:
                link = "https://vietstock.vn/" + href

            title = clean_text(
                a_tag.get_text(" ", strip=True)
            )

            url_normalized = normalize_url(link)

            if not url_normalized:
                continue

            articles.append(
                {
                    "source": "vietstock",
                    "category": category,
                    "channel_id": channel_id,
                    "listing_page": page,
                    "title_listing": title,
                    "url": link,
                    "url_normalized": url_normalized,
                    "discovered_at": pd.Timestamp.now(),
                    "crawl_batch_id": BATCH_ID,
                }
            )

        # Loại trùng ngay trong cùng một trang
        if articles:
            page_df = pd.DataFrame(articles)

            page_df = page_df.drop_duplicates(
                subset=["url_normalized"],
                keep="last",
            )

            articles = page_df.to_dict("records")

        return articles, "success"

    except requests.Timeout:
        return [], "timeout"

    except Exception as error:
        return (
            [],
            f"request_error:{type(error).__name__}",
        )

# === Cell 18 ===
def crawl_one_vietstock_category(
    category,
    channel_id,
    start_date,
    end_date,
    max_pages=None,
    max_empty_pages=2,
    max_repeated_pages=2,
):
    """
    Cào toàn bộ link của một chuyên mục Vietstock.

    max_pages=None:
        Không giới hạn cứng số trang.

    Điều kiện tự dừng:
        1. Gặp max_empty_pages trang trống liên tiếp.
        2. Nội dung trang bị lặp lại max_repeated_pages lần.
        3. Đạt max_pages nếu max_pages được cung cấp.
    """

    session = create_session()

    category_links = []

    page = 1
    empty_pages = 0
    repeated_pages = 0
    consecutive_no_new_pages = 0

    previous_page_urls = None
    seen_urls = set()

    stop_reason = None
    last_status = None

    try:
        while True:
            if (
                max_pages is not None
                and page > max_pages
            ):
                stop_reason = (
                    f"max_pages_{max_pages}"
                )

                print(
                    f"Vietstock — {category}: "
                    f"đạt giới hạn {max_pages} trang."
                )
                break

            articles, status = (
                fetch_vietstock_listing_page(
                    session=session,
                    channel_id=channel_id,
                    category=category,
                    page=page,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

            last_status = status

            if not articles:
                empty_pages += 1

                print(
                    f"Vietstock — {category} — "
                    f"trang {page}: không có dữ liệu "
                    f"({status}); "
                    f"empty={empty_pages}/"
                    f"{max_empty_pages}"
                )

                if empty_pages >= max_empty_pages:
                    stop_reason = (
                        f"empty_pages_{empty_pages}"
                    )
                    break

                page += 1

                time.sleep(
                    random.uniform(0.8, 1.5)
                )
                continue

            empty_pages = 0

            current_page_urls = tuple(
                sorted(
                    row["url_normalized"]
                    for row in articles
                    if row.get("url_normalized")
                )
            )

            # Kiểm tra trang hiện tại có giống
            # hoàn toàn trang trước đó không
            if current_page_urls == previous_page_urls:
                repeated_pages += 1
            else:
                repeated_pages = 0

            previous_page_urls = current_page_urls

            if repeated_pages >= max_repeated_pages:
                stop_reason = (
                    f"repeated_pages_{repeated_pages}"
                )

                print(
                    f"Vietstock — {category}: "
                    f"nội dung trang bị lặp "
                    f"{repeated_pages} lần; "
                    f"dừng tại trang {page}."
                )
                break

            new_articles = [
                row
                for row in articles
                if row["url_normalized"]
                not in seen_urls
            ]

            new_urls = {
                row["url_normalized"]
                for row in new_articles
            }

            seen_urls.update(new_urls)
            category_links.extend(new_articles)

            # Một cơ chế chống vòng lặp bổ sung:
            # trang có dữ liệu nhưng không tạo URL mới
            if not new_articles:
                consecutive_no_new_pages += 1
            else:
                consecutive_no_new_pages = 0

            print(
                f"Vietstock — {category} — "
                f"trang {page}: "
                f"tìm thấy={len(articles)}, "
                f"mới={len(new_articles)}, "
                f"lũy kế={len(category_links):,}"
            )

            if consecutive_no_new_pages >= 2:
                stop_reason = (
                    "consecutive_no_new_pages"
                )

                print(
                    f"Vietstock — {category}: "
                    "2 trang liên tiếp không có URL mới; "
                    f"dừng tại trang {page}."
                )
                break

            page += 1

            time.sleep(
                random.uniform(0.4, 0.9)
            )

    except Exception as error:
        stop_reason = (
            f"error:{type(error).__name__}"
        )

        print(
            f"Vietstock — {category}: "
            f"{type(error).__name__}: {error}"
        )

    finally:
        session.close()

    if stop_reason is None:
        stop_reason = "completed"

    return {
        "category": category,
        "channel_id": channel_id,
        "links": category_links,
        "n_links_raw": len(category_links),
        "last_page": page,
        "last_status": last_status,
        "stop_reason": stop_reason,
    }

# === Cell 19 ===
def crawl_vietstock_links_by_date(
    start_date,
    end_date,
    max_pages=None,
    category_workers=4,
):
    """
    Cào nhiều chuyên mục Vietstock song song.

    max_pages=None:
        Không giới hạn cứng số trang.

    category_workers:
        Số chuyên mục chạy đồng thời.
        Khuyến nghị 3–5.
    """

    start_date = pd.Timestamp(
        start_date
    ).normalize()

    end_date = pd.Timestamp(
        end_date
    ).normalize()

    all_links = []
    category_summaries = []

    with ThreadPoolExecutor(
        max_workers=category_workers
    ) as executor:

        futures = {
            executor.submit(
                crawl_one_vietstock_category,
                category=category,
                channel_id=channel_id,
                start_date=start_date,
                end_date=end_date,
                max_pages=max_pages,
            ): category
            for category, channel_id
            in VIETSTOCK_CHANNELS.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Vietstock categories",
        ):
            category = futures[future]

            try:
                result = future.result()

                all_links.extend(
                    result["links"]
                )

                category_summaries.append(
                    {
                        "category": category,
                        "channel_id": (
                            result["channel_id"]
                        ),
                        "n_links_raw": (
                            result["n_links_raw"]
                        ),
                        "last_page": (
                            result["last_page"]
                        ),
                        "last_status": (
                            result["last_status"]
                        ),
                        "stop_reason": (
                            result["stop_reason"]
                        ),
                        "status": "success",
                    }
                )

            except Exception as error:
                print(
                    f"\nLỗi chuyên mục {category}: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                category_summaries.append(
                    {
                        "category": category,
                        "channel_id": (
                            VIETSTOCK_CHANNELS[
                                category
                            ]
                        ),
                        "n_links_raw": 0,
                        "last_page": None,
                        "last_status": None,
                        "stop_reason": (
                            f"error:"
                            f"{type(error).__name__}"
                        ),
                        "status": "failed",
                    }
                )

    links_df = pd.DataFrame(all_links)

    summary_df = pd.DataFrame(
        category_summaries
    )

    if not summary_df.empty:
        summary_df = (
            summary_df
            .sort_values("category")
            .reset_index(drop=True)
        )

    if links_df.empty:
        return links_df, summary_df

    links_df = (
        links_df
        .dropna(
            subset=["url_normalized"]
        )
        .sort_values(
            ["category", "listing_page",
             "discovered_at"],
            na_position="last",
        )
        .drop_duplicates(
            subset=["url_normalized"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return links_df, summary_df

# === Cell 20 ===
def extract_meta_content(
    soup,
    key,
    value,
):
    tag = soup.find(
        "meta",
        attrs={key: value},
    )

    if tag:
        return clean_text(
            tag.get("content", "")
        )

    return ""

# === Cell 21 ===
def get_vietstock_article_detail(row):
    session = create_session()

    url = row["url"]
    base_result = row.to_dict()
    try:
        response = session.get(
            url,
            timeout=30,
            headers={
                "Referer": (
                    f"https://vietstock.vn/"
                    f"{row['category']}.htm"
                ),
            },
        )

        base_result["http_status"] = (
            response.status_code
        )

        if response.status_code != 200:
            base_result["parse_status"] = (
                f"http_{response.status_code}"
            )
            base_result["crawl_at"] = (
                pd.Timestamp.now()
            )

            return base_result

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        # -----------------------------------------
        # Tiêu đề
        # -----------------------------------------
        title = (
            extract_meta_content(
                soup,
                "property",
                "og:title",
            )
        )

        if not title:
            h1 = (
                soup.select_one("h1")
                or soup.select_one(
                    "[itemprop='headline']"
                )
            )

            title = (
                clean_text(
                    h1.get_text(" ", strip=True)
                )
                if h1
                else row.get(
                    "title_listing",
                    "",
                )
            )

        # -----------------------------------------
        # Mô tả / sapo
        # -----------------------------------------
        description = (
            extract_meta_content(
                soup,
                "property",
                "og:description",
            )
            or extract_meta_content(
                soup,
                "name",
                "description",
            )
        )

        if not description:
            sapo_tag = (
                soup.select_one(".sapo")
                or soup.select_one(
                    ".article-sapo"
                )
                or soup.select_one(
                    "[itemprop='description']"
                )
            )

            if sapo_tag:
                description = clean_text(
                    sapo_tag.get_text(
                        " ",
                        strip=True,
                    )
                )

        # -----------------------------------------
        # Ngày xuất bản
        # -----------------------------------------
        date_raw = ""

        time_tag = (
            soup.select_one(
                "time[itemprop='datePublished']"
            )
            or soup.select_one(
                "meta[property='article:"
                "published_time']"
            )
            or soup.select_one(
                ".meta .date"
            )
            or soup.select_one(
                ".date"
            )
        )

        if time_tag:
            date_raw = clean_text(
                time_tag.get("datetime")
                or time_tag.get("content")
                or time_tag.get_text(
                    " ",
                    strip=True,
                )
            )

        if not date_raw:
            meta_div = soup.find(
                "div",
                class_="meta",
            )

            if meta_div:
                date_raw = clean_text(
                    meta_div.get_text(
                        " ",
                        strip=True,
                    )
                )

        published_at = (
            parse_vietnamese_datetime(
                date_raw
            )
        )

        # -----------------------------------------
        # Nội dung chính
        # -----------------------------------------
        content_div = (
            soup.find(
                "div",
                id="vst_detail",
            )
            or soup.select_one(
                "[itemprop='articleBody']"
            )
            or soup.select_one(
                ".article-content"
            )
            or soup.select_one(
                ".content-detail"
            )
        )

        if content_div:
            for unwanted in content_div.select(
                "script, style, iframe, "
                "figure figcaption, "
                ".tags, .related-news, "
                ".social-sharing, "
                ".ads, .advertisement"
            ):
                unwanted.decompose()

            paragraphs = [
                clean_text(
                    paragraph.get_text(
                        " ",
                        strip=True,
                    )
                )
                for paragraph
                in content_div.find_all("p")
            ]

            paragraphs = [
                paragraph
                for paragraph in paragraphs
                if len(paragraph) >= 20
            ]

            content = "\n".join(
                paragraphs
            )

            if not content:
                content = clean_text(
                    content_div.get_text(
                        "\n",
                        strip=True,
                    )
                )
        else:
            content = ""

        tags = []

        tag_div = (
            soup.select_one(
                "div.tags.mt20"
            )
            or soup.select_one(
                "div.tags"
            )
            or soup.select_one(
                ".article-tags"
            )
        )

        if tag_div:
            tags = list(
                dict.fromkeys(
                    clean_text(
                        tag.get_text(
                            " ",
                            strip=True,
                        )
                    )
                    for tag in tag_div.find_all("a")
                    if clean_text(
                        tag.get_text(
                            " ",
                            strip=True,
                        )
                    )
                )
            )

        # -----------------------------------------
        # Tác giả
        # -----------------------------------------
        author_tag = (
            soup.select_one(
                "[itemprop='author']"
            )
            or soup.select_one(
                ".author"
            )
            or soup.select_one(
                ".article-author"
            )
        )

        author = clean_text(
            author_tag.get_text(
                " ",
                strip=True,
            )
            if author_tag
            else ""
        )

        # API Vietstock đôi khi có thể trả URL
        # nằm ngoài khoảng ngày yêu cầu.
        date_in_range = (
            pd.isna(published_at)
            or (
                VIETSTOCK_FROM
                <= published_at.normalize()
                <= CRAWL_TO
            )
        )

        base_result.update(
            {
                "title": clean_text(title),
                "description": clean_text(
                    description
                ),
                "content": clean_text(
                    content
                ),
                "author": author,
                "tags": tags,
                "published_raw": date_raw,
                "published_at": published_at,
                "date_in_range": date_in_range,
                "crawl_at": pd.Timestamp.now(),
                "parse_status": (
                    "success"
                    if len(clean_text(content)) >= 100
                    else "empty_content"
                ),
            }
        )

        base_result["article_id"] = (
            make_article_id(
                base_result[
                    "url_normalized"
                ]
            )
        )

        base_result["content_hash"] = (
            make_content_hash(content)
            if content
            else None
        )

        return base_result

    except requests.Timeout:
        base_result["parse_status"] = (
            "timeout"
        )
        base_result["crawl_at"] = (
            pd.Timestamp.now()
        )

        return base_result

    except Exception as error:
        base_result["parse_status"] = (
            f"parse_error:"
            f"{type(error).__name__}"
        )
        base_result["error_message"] = (
            str(error)[:500]
        )
        base_result["crawl_at"] = (
            pd.Timestamp.now()
        )

        return base_result

    finally:
        session.close()

# === Cell 22 ===
vietstock_links, vietstock_category_summary = (
    crawl_vietstock_links_by_date(
        start_date=VIETSTOCK_FROM,
        end_date=CRAWL_TO,

        # Không giới hạn cứng số trang
        max_pages=None,

        # Số chuyên mục chạy cùng lúc
        category_workers=(
            VIETSTOCK_CATEGORY_WORKERS
        ),
    )
)


print(
    "\nSố link Vietstock trong khoảng cập nhật:",
    f"{len(vietstock_links):,}",
)


if not vietstock_category_summary.empty:
    display(
        vietstock_category_summary
    )

    save_parquet_safe(
        vietstock_category_summary,
        UPDATE_DIR
        / (
            f"vietstock_category_summary_"
            f"{RUN_DATE}.parquet"
        ),
    )

# === Cell 23 ===
if not vietstock_links.empty:
    vietstock_links[
        "already_in_master"
    ] = (
        vietstock_links[
            "url_normalized"
        ].isin(OLD_URLS)
    )

    display(
        vietstock_links[
            [
                "category",
                "listing_page",
                "title_listing",
                "already_in_master",
            ]
        ].head(10)
    )

    print(
        "\nSố URL hoàn toàn mới:",
        (
            ~vietstock_links[
                "already_in_master"
            ]
        ).sum(),
    )

    print(
        "Số URL thuộc khoảng overlap:",
        vietstock_links[
            "already_in_master"
        ].sum(),
    )

    save_parquet_safe(
        vietstock_links,
        UPDATE_DIR
        / f"vietstock_links_{RUN_DATE}.parquet",
    )

    vietstock_results = []

    with ThreadPoolExecutor(
        max_workers=DETAIL_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                get_vietstock_article_detail,
                row,
            ): index
            for index, row
            in vietstock_links.iterrows()
        }

        for completed, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Vietstock detail",
            ),
            start=1,
        ):
            try:
                result = future.result()

            except Exception as error:
                index = futures[future]

                source_row = (
                    vietstock_links
                    .loc[index]
                    .to_dict()
                )

                source_row.update(
                    {
                        "parse_status": (
                            f"future_error:"
                            f"{type(error).__name__}"
                        ),
                        "error_message": (
                            str(error)[:500]
                        ),
                        "crawl_at": (
                            pd.Timestamp.now()
                        ),
                    }
                )

                result = source_row

            vietstock_results.append(
                result
            )

            if (
                completed
# COLAB COMMAND:                 % CHECKPOINT_EVERY
                == 0
            ):
                checkpoint_df = pd.DataFrame(
                    vietstock_results
                )

                save_parquet_safe(
                    checkpoint_df,
                    CHECKPOINT_DIR
                    / (
                        f"vietstock_checkpoint_"
                        f"{RUN_DATE}.parquet"
                    ),
                )

    vietstock_detail_all = pd.DataFrame(
        vietstock_results
    )

    # Lưu toàn bộ kết quả trước khi lọc ngày,
    # giúp kiểm tra lỗi parser ngày.
    save_parquet_safe(
        vietstock_detail_all,
        UPDATE_DIR
        / (
            f"vietstock_detail_all_status_"
            f"{RUN_DATE}.parquet"
        ),
    )

    vietstock_detail = (
        vietstock_detail_all.copy()
    )

    # Chỉ loại bài khi xác định chắc chắn
    # ngày nằm ngoài khoảng.
    if (
        "date_in_range"
        in vietstock_detail.columns
    ):
        vietstock_detail = (
            vietstock_detail[
                vietstock_detail[
                    "date_in_range"
                ].fillna(True)
            ]
            .copy()
            .reset_index(drop=True)
        )

    save_parquet_safe(
        vietstock_detail,
        UPDATE_DIR
        / (
            f"vietstock_detail_increment_"
            f"{RUN_DATE}.parquet"
        ),
    )

    print("\n===== VIETSTOCK DETAIL =====")

    display(
        vietstock_detail.groupby(
            ["category", "parse_status"],
            dropna=False,
        )
        .size()
        .rename("n_articles")
        .reset_index()
    )

    if "published_at" in vietstock_detail.columns:
        display(
            vietstock_detail.groupby(
                "category"
            ).agg(
                n_articles=(
                    "url_normalized",
                    "nunique",
                ),
                first_date=(
                    "published_at",
                    "min",
                ),
                last_date=(
                    "published_at",
                    "max",
                ),
                median_content_length=(
                    "content",
                    lambda values: (
                        values.fillna("")
                        .astype(str)
                        .str.len()
                        .median()
                    ),
                ),
            )
            .reset_index()
        )

else:
    vietstock_detail_all = pd.DataFrame()
    vietstock_detail = pd.DataFrame()

    print(
        "Không có bài Vietstock "
        "trong khoảng cập nhật."
    )

# === Cell 24 ===
