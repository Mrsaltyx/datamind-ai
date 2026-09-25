"""Helpers de typage robustes pandas 2.x / 3.x.

Depuis pandas 2.3 (option infer_string) et pandas 3, les colonnes de
chaines portent un dtype `str` distinct de `object`. Tout code testant
uniquement `is_object_dtype` devient aveugle aux colonnes string.
"""

from __future__ import annotations

import pandas as pd


def is_categorical_dtype(dtype) -> bool:
    """True pour object / str / string / category."""
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


def is_categorical(series: pd.Series) -> bool:
    return is_categorical_dtype(series.dtype)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes categorielles, robustes au dtype str de pandas 3."""
    return [c for c in df.columns if is_categorical_dtype(df[c].dtype)]
