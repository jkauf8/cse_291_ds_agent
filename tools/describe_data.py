# tools/describe_data.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def describe_data(df: pd.DataFrame, dataset_name: str = None) -> Dict[str, Any]:
    """
    Produce a deterministic, machine-readable summary of a pandas DataFrame.

    Returns a dictionary with keys:
      - dataset, shape, columns, dtypes, missing_values
      - numeric_summary (per numeric column)
      - categorical_summary (per non-numeric column)
      - top_correlations (list of top pairs by abs corr)
      - sample_rows (first 3 rows as list of dicts)
      - suggested_targets (numeric columns with non-zero variance)
      - suggested_features (mapping from target -> candidate features list)
      - text_summary (short human-friendly one-line summary)

    Notes:
      - All numeric stats are rounded deterministically to 6 decimal places.
      - The function does not modify the input DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    n_rows, n_cols = df.shape
    columns = df.columns.tolist()
    dtypes = {c: str(df[c].dtype) for c in columns}
    missing_counts = df.isnull().sum().to_dict()

    # Numeric summary
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_cols = numeric_df.columns.tolist()
    numeric_summary = {}
    for col in numeric_cols:
        s = numeric_df[col].dropna()
        if s.empty:
            numeric_summary[col] = {"count": 0}
            continue
        numeric_summary[col] = {
            "count": int(s.size),
            "mean": round(float(s.mean()), 6),
            "median": round(float(s.median()), 6),
            "std": round(float(s.std()), 6) if s.size > 1 else 0.0,
            "min": round(float(s.min()), 6),
            "25%": round(float(s.quantile(0.25)), 6),
            "75%": round(float(s.quantile(0.75)), 6),
            "max": round(float(s.max()), 6),
        }

    # Categorical (non-numeric) summary
    categorical_cols = [c for c in columns if c not in numeric_cols]
    categorical_summary = {}
    for col in categorical_cols:
        s = df[col].astype("object").dropna()
        if s.empty:
            categorical_summary[col] = {"unique": 0, "top_values": []}
            continue
        vc = s.value_counts().sort_values(ascending=False)
        vc.index = vc.index.infer_objects()
        top_vals = [{"value": str(v), "count": int(c)} for v, c in vc.head(5).items()]
        categorical_summary[col] = {"unique": int(s.nunique()), "top_values": top_vals}

    # Correlations (top pairs by absolute Pearson)
    top_correlations = []
    if len(numeric_cols) >= 2:
        try:
            corr = numeric_df.corr().abs()
            pairs = (
                corr.unstack()
                .reset_index()
                .rename(columns={"level_0": "col1", "level_1": "col2", 0: "abs_corr"})
            )
            pairs = pairs[pairs['col1'] != pairs['col2']]
            # canonical pair key to avoid duplicates
            pairs['pair_key'] = pairs.apply(lambda r: tuple(sorted([r['col1'], r['col2']])), axis=1)
            pairs = pairs.drop_duplicates(subset=['pair_key'])
            pairs = pairs.sort_values(by='abs_corr', ascending=False)
            for _, row in pairs.head(5).iterrows():
                top_correlations.append({
                    "pair": [row['pair_key'][0], row['pair_key'][1]],
                    "abs_corr": round(float(row['abs_corr']), 6)
                })
        except Exception:
            top_correlations = []

    # Sample rows
    sample_rows = df.head(3).to_dict(orient='records')

    # Suggested targets: numeric columns with at least 5 non-null and non-zero std
    suggested_targets: List[str] = []
    for col in numeric_cols:
        s = numeric_df[col].dropna()
        if s.size >= 5 and float(s.std()) > 0:
            suggested_targets.append(col)

    # Suggested features: for each suggested target, provide all other columns (planner may filter later)
    suggested_features = {t: [c for c in columns if c != t] for t in suggested_targets}

    # REPLACE the data_quality section with this:

    # NEW: DATA ISSUES SUMMARY (replaces data_quality)
    total_cells = n_rows * n_cols
    total_missing = sum(missing_counts.values())
    total_missing_percentage = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0
    
    duplicate_row_count = int(df.duplicated().sum())
    
    # Identify constant columns
    constant_columns = []
    binary_columns = []
    high_cardinality_columns = []
    
    for col in columns:
        unique_count = df[col].nunique()
        if unique_count == 1:
            constant_columns.append(col)
        elif unique_count == 2:
            binary_columns.append(col)
        elif unique_count > n_rows * 0.8:  # >80% unique values
            high_cardinality_columns.append(col)

    # CRITICAL ISSUES (must fix)
    critical_issues = []
    if constant_columns:
        critical_issues.append(f"constant_columns: {', '.join(constant_columns)}")
    if total_missing_percentage > 20:
        critical_issues.append(f"high_missing_data: {total_missing_percentage}%")
    if duplicate_row_count > n_rows * 0.1:  # >10% duplicates
        critical_issues.append(f"high_duplicate_rows: {duplicate_row_count}")

    # WARNINGS (should check)
    warnings = []
    if total_missing_percentage > 5:
        warnings.append(f"moderate_missing_data: {total_missing_percentage}%")
    if duplicate_row_count > 0:
        warnings.append(f"duplicate_rows: {duplicate_row_count}")
    if high_cardinality_columns:
        warnings.append(f"high_cardinality_columns: {', '.join(high_cardinality_columns[:3])}")  # Limit to first 3
    
    # Add distribution warnings
    if 'outlier_counts' in locals():
        for col, count in outlier_counts.items():
            if count > n_rows * 0.05:  # >5% outliers
                warnings.append(f"high_outliers_{col}: {count} outliers")
            elif count > 0:
                warnings.append(f"moderate_outliers_{col}: {count} outliers")
    
    if 'skewed_columns' in locals():
        for col, skew_val in skewed_columns.items():
            if abs(skew_val) > 2.0:
                warnings.append(f"high_skew_{col}: {skew_val:.2f}")
            elif abs(skew_val) > 1.0:
                warnings.append(f"moderate_skew_{col}: {skew_val:.2f}")

    # QUALITY SCORE CALCULATION (0-10 scale)
    quality_score = 10.0
    
    # Deduct for critical issues (severe)
    quality_score -= len(critical_issues) * 3
    
    # Deduct for warnings (moderate)
    quality_score -= len(warnings) * 0.5
    
    # Deduct for missing data
    quality_score -= min(total_missing_percentage / 5, 3)  # Max -3 points
    
    # Deduct for high duplicates
    if duplicate_row_count > 0:
        duplicate_percentage = (duplicate_row_count / n_rows) * 100
        quality_score -= min(duplicate_percentage / 10, 2)  # Max -2 points
    
    quality_score = max(0, min(10, round(quality_score, 1)))
    
    data_issues = {
        "quality_score": quality_score,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "total_missing_percentage": total_missing_percentage,
        "duplicate_row_count": duplicate_row_count
    }
    missing_percentages = {col: round((count / n_rows * 100), 2) for col, count in missing_counts.items()}
    columns_with_high_missing = [col for col, percentage in missing_percentages.items() if percentage > 20]
    
    missing_analysis = {
        "percentages": missing_percentages,
        "columns_with_high_missing": columns_with_high_missing,
        "total_missing_cells": int(total_missing)
    }

    # NEW: TARGETED DISTRIBUTION ANALYSIS FOR HOUSING & COFFEE DATA
    skewed_columns = {}
    outlier_counts = {}
    zero_counts = {}
    outlier_details = {}

    for col in numeric_cols:
        s = numeric_df[col].dropna()
        if len(s) < 2:  # Need at least 2 values
            continue
        
        # 1. IDENTIFY DATA TYPE FOR SMART ANALYSIS
        is_likely_id = (
            col.lower().endswith(('_id', 'id')) or
            s.nunique() == len(s)  # All values unique
        )
        
        is_likely_categorical = (
            s.nunique() <= 10 and s.dtype in ['int64', 'int32']  # Small integer range
        )
        
        is_likely_continuous = (
            s.nunique() > 20 and  # Reasonable diversity
            not is_likely_id and
            not is_likely_categorical
        )
        
        # 2. SKEWNESS (only for continuous data)
        if is_likely_continuous:
            skew_val = float(s.skew())
            if abs(skew_val) > 0.5:
                skewed_columns[col] = round(skew_val, 6)
        
        # 3. OUTLIER DETECTION (smart based on data type)
        if is_likely_id:
            # ID columns - skip outlier detection
            outlier_counts[col] = 0
        elif is_likely_categorical:
            # Categorical codes - use percentile method only
            p05 = s.quantile(0.05)
            p95 = s.quantile(0.95)
            outliers = s[(s < p05) | (s > p95)]
            outlier_counts[col] = len(outliers)
        else:
            # Continuous data - use robust IQR method
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                # Domain-aware bounds for housing data
                if col == 'price':  # Housing prices can vary widely
                    lower_bound = Q1 - 2.5 * IQR
                    upper_bound = Q3 + 2.5 * IQR
                elif col in ['area', 'unit_price']:  # Prices/areas can have high variance
                    lower_bound = Q1 - 2.0 * IQR  
                    upper_bound = Q3 + 2.0 * IQR
                else:  # Standard IQR for other numeric columns
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                
                outliers = s[(s < lower_bound) | (s > upper_bound)]
                outlier_counts[col] = len(outliers)
                
                outlier_details[col] = {
                    "outlier_percentage": round(len(outliers) / len(s) * 100, 2),
                    "method": "domain_aware_iqr"
                }
            else:
                outlier_counts[col] = 0
        
        # 4. ZERO COUNTS (meaningful analysis)
        zero_count = int((s == 0).sum())
        if zero_count > 0:
            # Only report if zeros might be meaningful
            if not is_likely_id and zero_count < len(s):  # Not all zeros
                zero_counts[col] = zero_count

    distributions = {
        "skewed_columns": skewed_columns,
        "outlier_counts": outlier_counts,
        "outlier_details": outlier_details,
        "zero_counts": zero_counts
    }


    text_summary = (
        f"Dataset '{dataset_name or 'unnamed'}': {n_rows} rows × {n_cols} columns. "
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} non-numeric columns. "
        f"Suggested targets: {', '.join(suggested_targets) if suggested_targets else 'None'}."
    )

    result = {
        "dataset": dataset_name,
        "shape": {"rows": int(n_rows), "columns": int(n_cols)},
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": {k: int(v) for k, v in missing_counts.items()},
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "top_correlations": top_correlations,
        "sample_rows": sample_rows,
        "suggested_targets": suggested_targets,
        "suggested_features": suggested_features,
        "text_summary": text_summary,
        "data_issues": data_issues,
        "missing_analysis": missing_analysis,
        "distribution_analysis": distributions,
    }

    return result
