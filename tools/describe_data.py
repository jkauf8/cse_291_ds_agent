# tools/describe_data.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def numeric_summary_for_col(s: pd.Series) -> Dict[str, Any]:
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return {"count": 0}
    return {
        "count": int(s_nonnull.size),
        "mean": round(float(s_nonnull.mean()), 6),
        "median": round(float(s_nonnull.median()), 6),
        "std": round(float(s_nonnull.std()), 6) if s_nonnull.size > 1 else 0.0,
        "min": round(float(s_nonnull.min()), 6),
        "25%": round(float(s_nonnull.quantile(0.25)), 6),
        "75%": round(float(s_nonnull.quantile(0.75)), 6),
        "max": round(float(s_nonnull.max()), 6),
    }


def detect_outliers(
    s: pd.Series,
    factor: float = 1.5,
    return_examples: int = 5
) -> Dict[str, Any]:
    """
    IQR outlier detection on non-null values.
    Returns dict with:
      - count: number of outlier rows
      - percent: percent of non-null rows that are outliers (rounded 2dp)
      - lower_bound, upper_bound
      - examples: list of dicts {index, value} (first `return_examples` hits)
    """
    s_nonnull = s.dropna()
    n = len(s_nonnull)
    if n < 4:
        return {"count": 0, "percent": 0.0, "lower_bound": None, "upper_bound": None, "examples": []}

    q1 = s_nonnull.quantile(0.25)
    q3 = s_nonnull.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # No spread -> no outliers by IQR
        return {"count": 0, "percent": 0.0, "lower_bound": None, "upper_bound": None, "examples": []}

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    outlier_mask = (s_nonnull < lower) | (s_nonnull > upper)
    outliers = s_nonnull[outlier_mask]
    count = int(outliers.size)
    percent = round((count / n * 100), 2) if n > 0 else 0.0

    # examples: provide first few by their original index and value
    examples = []
    for idx, val in outliers.head(return_examples).items():
        examples.append({"index": int(idx) if (isinstance(idx, (int, np.integer))) else str(idx), "value": float(val)})

    return {
        "count": count,
        "percent": percent,
        "lower_bound": round(float(lower), 6),
        "upper_bound": round(float(upper), 6),
        "examples": examples,
    }


def top_correlations(numeric_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    if numeric_df.shape[1] < 2:
        return []
    corr = numeric_df.corr().abs()
    pairs = (
        corr.unstack()
        .reset_index()
        .rename(columns={"level_0": "col1", "level_1": "col2", 0: "abs_corr"})
    )
    pairs = pairs[pairs["col1"] != pairs["col2"]]
    pairs["pair_key"] = pairs.apply(lambda r: tuple(sorted([r["col1"], r["col2"]])), axis=1)
    pairs = pairs.drop_duplicates(subset=["pair_key"])
    pairs = pairs.sort_values(by="abs_corr", ascending=False)
    out = []
    for _, row in pairs.head(top_n).iterrows():
        out.append({"pair": [row["pair_key"][0], row["pair_key"][1]], "abs_corr": round(float(row["abs_corr"]), 6)})
    return out


# --- paste into tools/describe_data.py, replacing the old describe_data(...) function ---
def describe_data(
    df: pd.DataFrame,
    dataset_name: Optional[str] = None,
    *,
    do_detect_outliers: bool = True,        # renamed flag (was `detect_outliers`)
    outlier_iqr_factor: float = 1.5,
    outlier_examples: int = 5,
    top_correlation: int = 5,
    sample_rows: int = 3
) -> Dict[str, Any]:
    """
    Compact and deterministic dataset description.

    Important points:
      - DOES NOT drop rows globally: missing values are preserved.
      - Numeric stats are computed on non-null values.
      - Outlier detection uses an IQR method and is deterministic (configurable factor).
      - Returns outlier counts and examples so the user can inspect.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    n_rows, n_cols = df.shape
    columns = df.columns.tolist()
    dtypes = {c: str(df[c].dtype) for c in columns}

    # Missing counts (based on original df)
    missing_counts = df.isnull().sum().to_dict()

    # Numeric and categorical separation
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()
    categorical_cols = [c for c in columns if c not in numeric_cols]

    # Numeric summary
    numeric_summary: Dict[str, Any] = {}
    for col in numeric_cols:
        numeric_summary[col] = numeric_summary_for_col(df[col])

    # Categorical summary (top values)
    categorical_summary: Dict[str, Any] = {}
    for col in categorical_cols:
        s = df[col].dropna().astype("object")
        if s.empty:
            categorical_summary[col] = {"unique": 0, "top_values": []}
        else:
            vc = s.value_counts().head(5)
            top_vals = [{"value": str(v), "count": int(c)} for v, c in vc.items()]
            categorical_summary[col] = {"unique": int(s.nunique()), "top_values": top_vals}

    # Top correlations
    top_corrs = top_correlations(numeric_df, top_n=top_correlation)

    # Sample rows
    sample_rows_list = df.head(sample_rows).to_dict(orient="records")

    # Suggested targets (numeric with > 5 non-null values and non-zero std)
    suggested_targets: List[str] = []
    for col in numeric_cols:
        s_nonnull = df[col].dropna()
        if s_nonnull.size >= 5 and float(s_nonnull.std()) > 0:
            suggested_targets.append(col)
    suggested_features = {t: [c for c in columns if c != t] for t in suggested_targets}

    # Data quality metrics
    total_cells = n_rows * n_cols
    total_missing = int(sum(missing_counts.values()))
    total_missing_percentage = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
    duplicate_row_count = int(df.duplicated().sum())

    # Issues / warnings (kept simple)
    critical_issues = []
    warnings = []
    if total_missing_percentage > 50:
        critical_issues.append(f"very_high_missing: {total_missing_percentage}%")
    elif total_missing_percentage > 10:
        warnings.append(f"moderate_missing: {total_missing_percentage}%")
    if duplicate_row_count > max(1, int(n_rows * 0.05)):
        warnings.append(f"duplicate_rows: {duplicate_row_count}")

    # Quality score heuristic
    quality_score = 10.0
    quality_score -= min(total_missing_percentage / 5, 4)
    quality_score -= min(len(critical_issues) * 2, 4)
    quality_score -= min(len(warnings) * 0.5, 2)
    quality_score = max(0.0, round(quality_score, 1))

    data_issues = {
        "quality_score": quality_score,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "total_missing_percentage": total_missing_percentage,
        "duplicate_row_count": duplicate_row_count,
    }

    missing_percentages = {col: round((count / n_rows * 100), 2) for col, count in missing_counts.items()}
    columns_with_high_missing = [col for col, pct in missing_percentages.items() if pct > 20]

    missing_analysis = {
        "percentages": missing_percentages,
        "columns_with_high_missing": columns_with_high_missing,
        "total_missing_cells": total_missing,
    }

    # Outlier detection (IQR) per numeric column
    outliers_info: Dict[str, Any] = {}
    if do_detect_outliers and numeric_cols:
        for col in numeric_cols:
            try:
                outliers_info[col] = detect_outliers(
                    df[col],
                    factor=outlier_iqr_factor,
                    return_examples=outlier_examples
                )
            except Exception as e:
                # don't crash the whole describe if outlier detection has an issue;
                # record the exception for debugging.
                outliers_info[col] = {"error": f"outlier detection failed: {str(e)}"}

    distributions = {
        "outliers": outliers_info
    }

    text_summary = (
        f"Dataset '{dataset_name or 'unnamed'}': {n_rows} rows × {n_cols} columns. "
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} non-numeric columns. "
        f"Suggested targets: {', '.join(suggested_targets) if suggested_targets else 'None'}."
    )

    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "shape": {"rows": int(n_rows), "columns": int(n_cols)},
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": {k: int(v) for k, v in missing_counts.items()},
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "top_correlations": top_corrs,
        "sample_rows": sample_rows_list,
        "suggested_targets": suggested_targets,
        "suggested_features": suggested_features,
        "text_summary": text_summary,
        "data_issues": data_issues,
        "missing_analysis": missing_analysis,
        "distribution_analysis": distributions,
    }

    return result
