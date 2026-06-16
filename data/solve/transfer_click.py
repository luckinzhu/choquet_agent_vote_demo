"""
Append Tencent clickbait records into the toy_data.csv task dataset.

This script:
1. Loads the existing toy dataset.
2. Loads the older Tencent clickbait dataset.
3. Transforms old rows to the toy dataset schema.
4. Appends the transformed rows without modifying existing rows.
5. Writes the merged result back to toy_data.csv.

Run with Python 3.x:
    python merge_tencent_into_toy_data.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


OLD_DATASET_PATH = Path(r"H:\clickbait_data\multimodel_clickbait\wangyi\wangyi_nc.csv")
NEW_DATASET_PATH = Path(
    r"F:\PY_projects\03_STCLABS\Choquet\choquet_agent_vote_demo\data\raw_data\clickbait\wangyi_ot.csv"
)

TARGET_COLUMNS = ["task_name", "task_description", "label", "text"]

# The toy dataset uses task metadata. Reuse the existing clickbait description
# from toy_data.csv when present; otherwise fall back to this description.
DEFAULT_CLICKBAIT_DESCRIPTION = (
    "clickbait detection: decide whether a headline uses misleading curiosity gap, "
    "sensational wording, or click-inducing intention"
)

# Explicit semantic mapping from old Tencent columns to the target dataset columns.
# Keep Tencent title as title instead of duplicating it into text.
COLUMN_MAPPING = {
    "title": "text",
    # "content": "content",
    "label": "label",
}

# These old Tencent identifiers are not useful for the toy task dataset and can
# add noise. The text column is also excluded because it duplicates title.
EXCLUDED_COLUMNS = {"id", "folder_name", "foldername", "text"}

ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV file while tolerating common UTF and Chinese encodings."""
    if not path.exists():
        return pd.DataFrame(columns=TARGET_COLUMNS), "utf-8-sig"

    last_error: Exception | None = None
    for encoding in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df, encoding
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=TARGET_COLUMNS), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode {path} with encodings {ENCODINGS_TO_TRY}: {last_error}",
    )


def first_non_empty(values: Iterable[object], default: str) -> str:
    """Return the first non-empty value from an iterable, or a default."""
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value)
    return default


def get_clickbait_description(new_df: pd.DataFrame) -> str:
    """Use the existing clickbait task description if it exists in toy_data.csv."""
    required_columns = {"task_name", "task_description"}
    if not required_columns.issubset(new_df.columns):
        return DEFAULT_CLICKBAIT_DESCRIPTION

    clickbait_rows = new_df["task_name"].astype("string").str.lower().eq("clickbait")
    if not clickbait_rows.any():
        return DEFAULT_CLICKBAIT_DESCRIPTION

    return first_non_empty(
        new_df.loc[clickbait_rows, "task_description"],
        DEFAULT_CLICKBAIT_DESCRIPTION,
    )


def coerce_to_existing_dtype(series: pd.Series, target_dtype: object) -> pd.Series:
    """Best-effort conversion to match an existing toy_data.csv column dtype."""
    try:
        if pd.api.types.is_integer_dtype(target_dtype):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.isna().any():
                return numeric.astype("Int64")
            return numeric.astype(target_dtype)

        if pd.api.types.is_float_dtype(target_dtype):
            return pd.to_numeric(series, errors="coerce").astype(target_dtype)

        if pd.api.types.is_bool_dtype(target_dtype):
            return series.astype("boolean")

        if pd.api.types.is_datetime64_any_dtype(target_dtype):
            return pd.to_datetime(series, errors="coerce")

        return series.astype(target_dtype)
    except (TypeError, ValueError):
        return series


def transform_old_dataset(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Transform Tencent records to the toy dataset schema and keep old extras."""
    new_columns = TARGET_COLUMNS
    transformed = pd.DataFrame(index=old_df.index)

    # Preserve the column order of toy_data.csv first.
    for new_column in new_columns:
        old_column = next(
            (source for source, target in COLUMN_MAPPING.items() if target == new_column),
            None,
        )

        if old_column and old_column in old_df.columns:
            transformed[new_column] = old_df[old_column]
        elif new_column == "task_name":
            transformed[new_column] = "clickbait"
        elif new_column == "task_description":
            transformed[new_column] = get_clickbait_description(new_df)
        else:
            transformed[new_column] = pd.NA

        if new_column in new_df.columns:
            transformed[new_column] = coerce_to_existing_dtype(
                transformed[new_column],
                new_df[new_column].dtype,
            )

    return transformed


def preserve_text_from_title(new_df: pd.DataFrame) -> pd.DataFrame:
    """Move title into text only when text is missing, then use text as final column."""
    if "title" not in new_df.columns:
        return new_df

    new_df = new_df.copy()
    if "text" not in new_df.columns:
        new_df["text"] = new_df["title"]
        return new_df

    text_is_empty = new_df["text"].isna() | new_df["text"].astype("string").str.strip().eq("")
    title_has_value = new_df["title"].notna() & new_df["title"].astype("string").str.strip().ne("")
    new_df.loc[text_is_empty & title_has_value, "text"] = new_df.loc[
        text_is_empty & title_has_value,
        "title",
    ]
    return new_df


def main() -> None:
    """Load, transform, append, and save the merged dataset."""
    new_df, new_encoding = read_csv_robust(NEW_DATASET_PATH)
    old_df, old_encoding = read_csv_robust(OLD_DATASET_PATH)
    new_df = preserve_text_from_title(new_df)

    transformed_old_df = transform_old_dataset(old_df, new_df)

    final_columns = TARGET_COLUMNS

    # Align both dataframes to the fixed output schema, then append old rows.
    new_aligned = new_df.reindex(columns=final_columns)
    transformed_aligned = transformed_old_df.reindex(columns=final_columns)
    merged_df = pd.concat([new_aligned, transformed_aligned], ignore_index=True)

    # Write without the pandas index. Existing rows remain first; old rows are appended.
    NEW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(NEW_DATASET_PATH, index=False, encoding=new_encoding)

    print(f"Loaded new dataset: {NEW_DATASET_PATH} ({len(new_df)} rows, {new_encoding})")
    print(f"Loaded old dataset: {OLD_DATASET_PATH} ({len(old_df)} rows, {old_encoding})")
    print(f"Saved merged dataset: {NEW_DATASET_PATH} ({len(merged_df)} rows)")
    print(f"Final columns: {', '.join(final_columns)}")


if __name__ == "__main__":
    main()
