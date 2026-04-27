from __future__ import annotations

import csv
import logging
from io import StringIO

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

# Warning threshold: warn if dataset exceeds this but still load it
LARGE_DATASET_WARNING = 100_000


def load_csv(uploaded_file, max_rows: int | None = None, max_size_mb: float = 200) -> pd.DataFrame:
    """Load a CSV file with automatic encoding and delimiter detection.

    Args:
        uploaded_file: File-like object with .getvalue() method
        max_rows: Maximum rows to load. None means no limit (loads all rows).
        max_size_mb: Maximum file size in MB.

    Returns:
        pandas DataFrame with the loaded data.
    """
    raw_bytes = uploaded_file.getvalue()

    if len(raw_bytes) > max_size_mb * 1024 * 1024:
        raise ValueError(
            f"Fichier trop volumineux ({len(raw_bytes) / 1024 / 1024:.1f} Mo). "
            f"Taille maximale : {max_size_mb} Mo."
        )

    df = None
    last_encoding_error = None

    for encoding in ENCODING_FALLBACKS:
        try:
            text = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            last_encoding_error = encoding
            continue

        sniffer_sample = text[:8192]
        try:
            dialect = csv.Sniffer().sniff(sniffer_sample)
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        try:
            df = pd.read_csv(StringIO(text), sep=delimiter)
            break
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(StringIO(text))
                break
            except pd.errors.ParserError:
                continue

    if df is None:
        raise ValueError(
            f"Impossible de parser le fichier CSV. "
            f"Derniere tentative d'encodage echouee : {last_encoding_error}."
        )

    if df.empty or df.shape[1] == 0:
        raise ValueError("Le fichier CSV est vide ou ne contient aucune colonne valide.")

    total_rows = len(df)

    # Optimize memory usage for large datasets
    if total_rows > 10_000:
        df = _optimize_dtypes(df)

    # Warn for large datasets instead of silently sampling
    if total_rows > LARGE_DATASET_WARNING:
        logger.warning(
            "Dataset volumineux : %d lignes chargees. L'analyse peut etre plus lente.",
            total_rows,
        )

    # Apply max_rows limit if specified (but default is None = no limit)
    if max_rows is not None and total_rows > max_rows:
        logger.warning(
            "Dataset tronque de %d a %d lignes (max_rows=%d).",
            total_rows,
            max_rows,
            max_rows,
        )
        df = df.head(max_rows).reset_index(drop=True)

    return df


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "object":
            num_unique = df[col].nunique()
            if num_unique / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")
        elif col_type in ("int64", "int32"):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype("uint8")
                elif c_max < 65535:
                    df[col] = df[col].astype("uint16")
                elif c_max < 4294967295:
                    df[col] = df[col].astype("uint32")
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype("int8")
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype("int16")
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype("int32")
        elif col_type == "float64":
            df[col] = df[col].astype("float32")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    for col in df.columns:
        if col not in datetime_cols:
            try:
                parsed = pd.to_datetime(df[col], format="mixed")
                if parsed.notna().sum() / len(df) > 0.8:
                    datetime_cols.append(col)
            except (ValueError, TypeError):
                pass

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


def get_sample_data(df: pd.DataFrame, n: int = 5) -> str:
    return df.head(n).to_string()


def get_column_stats(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    stats = {
        "name": col,
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "missing": int(series.isnull().sum()),
        "missing_pct": round(series.isnull().sum() / len(series) * 100, 2),
    }

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        stats.update(
            {
                "type": "numeric",
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": float(desc["min"]),
                "q25": float(desc["25%"]),
                "median": float(desc["50%"]),
                "q75": float(desc["75%"]),
                "max": float(desc["max"]),
                "skew": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "zeros": int((series == 0).sum()),
                "zeros_pct": round((series == 0).sum() / len(series) * 100, 2),
            }
        )
    else:
        vc = series.value_counts()
        stats.update(
            {
                "type": "categorical",
                "unique": int(series.nunique()),
                "top_values": vc.head(10).to_dict(),
                "top_pct": (vc.head(10) / len(series) * 100).round(2).to_dict(),
            }
        )

    return stats
