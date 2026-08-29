"""Deterministic entity resolution over labelled records."""

from .resolve import (STOPWORDS, canonical, fold_duplicates, fold_key, looks_like_handle,
                      stem)

__all__ = ["STOPWORDS", "canonical", "fold_duplicates", "fold_key", "looks_like_handle", "stem"]
